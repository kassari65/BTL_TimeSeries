import pandas as pd
import numpy as np
import cv2
import h5py
import logging
import warnings
import json
import xarray as xr
from tqdm import tqdm
import zipfile
import os

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
warnings.filterwarnings(action='ignore', message='Mean of empty slice')

# --- CONFIGURATION ---
LON_MIN, LON_MAX = 100.0, 125.0
LAT_MIN, LAT_MAX = 0.0, 25.0
SPATIAL_RES = 0.25 
IMG_SIZE = 100

def coord_to_pixel(lat, lon):
    x = int((lon - LON_MIN) / SPATIAL_RES)
    y = int((LAT_MAX - lat) / SPATIAL_RES)
    return x, y

def load_and_preprocess_ibtracs(csv_path):
    logging.info(f"Nạp dữ liệu IBTrACS: {csv_path}")
    df = pd.read_csv(csv_path, skiprows=[1], low_memory=False)
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'])
    df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
    df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
    df['USA_WIND'] = pd.to_numeric(df['USA_WIND'], errors='coerce')
    
    df = df[
        (df['ISO_TIME'].dt.year.between(2021, 2025)) &
        (df['ISO_TIME'].dt.month.between(6, 12)) &
        (df['ISO_TIME'].dt.hour.isin([0, 6, 12, 18])) &
        (df['LON'].between(LON_MIN, LON_MAX)) &
        (df['LAT'].between(LAT_MIN, LAT_MAX)) &
        (df['NATURE'] != 'ET')
    ]

    radii_cols = ['USA_R34_NE', 'USA_R34_SE', 'USA_R34_SW', 'USA_R34_NW']
    for col in radii_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['RADIUS_DEG'] = df[radii_cols].mean(axis=1) / 60.0

    def impute_radius(row):
        if pd.notna(row['RADIUS_DEG']): return row['RADIUS_DEG']
        wind = row['USA_WIND']
        if pd.isna(wind) or wind < 34: return 1.5
        elif 34 <= wind < 64: return 2.0
        else: return 2.5      
    df['RADIUS_DEG'] = df.apply(impute_radius, axis=1)
    return df

def process_era5_and_labels(storm_df, output_h5, scaler_json):
    logging.info("Bắt đầu tiền xử lý ERA5 và sinh nhãn...")
    
    # Để tính Global Mean/Std
    sum_vals = np.zeros(13, dtype=np.float64)
    sum_sq_vals = np.zeros(13, dtype=np.float64)
    total_pixels = 0
    
    years = [2021, 2022, 2023, 2024, 2025]
    
    with h5py.File(output_h5, 'w') as f_h5:
        for year in years:
            logging.info(f"Đang xử lý năm {year}...")
            
            # Load nc files
            surf_path = f"ERA5_Data/surface_{year}.nc"
            pres_path = f"ERA5_Data/pressure_{year}.nc"
            
            if not os.path.exists(surf_path) or not os.path.exists(pres_path):
                logging.warning(f"Không tìm thấy file dữ liệu ERA5 cho năm {year}. Bỏ qua.")
                continue
                
            # Xử lý trường hợp CDS API trả về file zip ẩn dưới đuôi .nc
            if zipfile.is_zipfile(surf_path):
                logging.info(f"Giải nén {surf_path}...")
                with zipfile.ZipFile(surf_path, 'r') as z:
                    nc_filename = z.namelist()[0]
                    z.extract(nc_filename, path='ERA5_Data')
                    temp_surf = f'ERA5_Data/temp_surf_{year}.nc'
                    if os.path.exists(temp_surf): os.remove(temp_surf)
                    os.rename(os.path.join('ERA5_Data', nc_filename), temp_surf)
                ds_surf = xr.open_dataset(temp_surf)
            else:
                ds_surf = xr.open_dataset(surf_path)
                
            if zipfile.is_zipfile(pres_path):
                logging.info(f"Giải nén {pres_path}...")
                with zipfile.ZipFile(pres_path, 'r') as z:
                    nc_filename = z.namelist()[0]
                    z.extract(nc_filename, path='ERA5_Data')
                    temp_pres = f'ERA5_Data/temp_pres_{year}.nc'
                    if os.path.exists(temp_pres): os.remove(temp_pres)
                    os.rename(os.path.join('ERA5_Data', nc_filename), temp_pres)
                ds_pres = xr.open_dataset(temp_pres)
            else:
                ds_pres = xr.open_dataset(pres_path)
            
            # Extract time values
            # Xarray times are datetime64. We need to intersect with our target times
            target_times = pd.date_range(start=f'{year}-06-01', end=f'{year}-12-31 18:00', freq='6h')
            
            # Get valid times that exist in the datasets
            time_dim = 'valid_time' if 'valid_time' in ds_surf.dims else 'time'
            avail_times = pd.to_datetime(ds_surf[time_dim].values)
            valid_times = target_times.intersection(avail_times)
            
            ds_surf = ds_surf.sel({time_dim: valid_times}).fillna(0)
            ds_pres = ds_pres.sel({time_dim: valid_times}).fillna(0)
            
            try:
                p500 = ds_pres.sel(pressure_level=500)
                p850 = ds_pres.sel(pressure_level=850)
            except KeyError:
                p500 = ds_pres.sel(level=500)
                p850 = ds_pres.sel(level=850)
                
            N = len(valid_times)
            
            inputs_array = np.zeros((N, 13, IMG_SIZE, IMG_SIZE), dtype=np.float32)
            labels_array = np.zeros((N, IMG_SIZE, IMG_SIZE), dtype=np.float32)
            
            for i, current_time in enumerate(tqdm(valid_times, desc=f"Rendering {year}")):
                # Lấy numpy arrays của từng channel (100, 100)
                
                s_ds = ds_surf.isel({time_dim: i})
                p5 = p500.isel({time_dim: i})
                p8 = p850.isel({time_dim: i})
                
                # Trích xuất 13 features
                try:
                    channels = [
                        s_ds['msl'].values, s_ds['sst'].values, s_ds['u10'].values, s_ds['v10'].values, s_ds['tcwv'].values,
                        p5['r'].values, p5['u'].values, p5['v'].values, p5['z'].values,
                        p8['r'].values, p8['u'].values, p8['v'].values, p8['z'].values
                    ]
                except KeyError as e:
                    # Nếu tên biến bị khác (vd u10 -> u, v10 -> v...) thì catch ở đây
                    logging.error(f"Missing variable: {e} in dataset. Variables present: {list(s_ds.keys())}")
                    raise e
                    
                frame_tensor = np.stack(channels, axis=0)
                
                # Đảm bảo shape là (13, 100, 100)
                if frame_tensor.shape[1:] != (IMG_SIZE, IMG_SIZE):
                    frame_tensor = frame_tensor[:, :IMG_SIZE, :IMG_SIZE]
                    
                inputs_array[i] = frame_tensor
                
                # Cập nhật Mean/Std Accumulators
                # Tính tổng và tổng bình phương theo không gian (H, W) -> ra mảng (16,)
                sum_vals += np.sum(frame_tensor, axis=(1, 2))
                sum_sq_vals += np.sum(frame_tensor ** 2, axis=(1, 2))
                
                # Tạo mask label
                mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
                active_storms = storm_df[storm_df['ISO_TIME'] == current_time]          
                for _, storm in active_storms.iterrows():
                    px_x, px_y = coord_to_pixel(storm['LAT'], storm['LON'])
                    px_radius = max(1, int(storm['RADIUS_DEG'] / SPATIAL_RES))
                    cv2.circle(mask, (px_x, px_y), px_radius, 1.0, -1)
                
                if not active_storms.empty:
                    mask = cv2.GaussianBlur(mask, (5, 5), 0)
                    if mask.max() > 0: mask /= mask.max()       
                
                labels_array[i] = mask
                
            total_pixels += N * IMG_SIZE * IMG_SIZE
            
            # Không tạo Sliding Window nữa để tránh quá tải RAM (OOM) và ổ cứng
            if N > 0:
                # Tạo Group theo năm và Dataset
                grp = f_h5.create_group(str(year))
                grp.create_dataset('inputs', data=inputs_array, compression="gzip")
                grp.create_dataset('labels', data=labels_array, compression="gzip")
            else:
                logging.warning(f"Năm {year} không có dữ liệu (N={N}).")
            
            # Clean up memory
            if 'p500' in locals(): del p500
            if 'p850' in locals(): del p850
            if 's_ds' in locals(): del s_ds
            if 'p5' in locals(): del p5
            if 'p8' in locals(): del p8
            ds_surf.close()
            ds_pres.close()
            del ds_surf, ds_pres
            import gc
            gc.collect()
            
            # Xoá file temp nếu có (Bỏ qua lỗi khóa file của Windows)
            try:
                if zipfile.is_zipfile(surf_path):
                    os.remove(f'ERA5_Data/temp_surf_{year}.nc')
                if zipfile.is_zipfile(pres_path):
                    os.remove(f'ERA5_Data/temp_pres_{year}.nc')
            except Exception as e:
                logging.warning(f"Không thể xoá file tạm thời (Windows File Lock): {e}")
            
    # Tính Mean và Std cuối cùng
    logging.info("Tính toán Mean và Std từ toàn bộ tập dữ liệu...")
    mean_arr = sum_vals / total_pixels
    variance = (sum_sq_vals / total_pixels) - (mean_arr ** 2)
    
    # Prevent negative variance due to floating point precision
    variance = np.maximum(variance, 0)
    std_arr = np.sqrt(variance)
    
    # Tránh chia cho 0 nếu Std = 0
    std_arr[std_arr == 0] = 1e-6
    
    scaler = {
        "mean": mean_arr.tolist(),
        "std": std_arr.tolist(),
        "channels": [
            "msl", "sst", "u10", "v10", "tcwv",
            "r_500", "u_500", "v_500", "z_500",
            "r_850", "u_850", "v_850", "z_850"
        ]
    }
    
    with open(scaler_json, 'w') as f:
        json.dump(scaler, f, indent=4)
        
    logging.info(f"Đã lưu scaler vào {scaler_json}")
    logging.info(f"HOÀN TẤT: Dữ liệu được lưu vào {output_h5}")

if __name__ == "__main__":
    ibtracs_csv = "ibtracs.WP.list.v04r01.csv"
    output_h5 = "SCS_Typhoon_Dataset_Raw.h5"
    scaler_file = "scaler.json"
    
    if os.path.exists(output_h5):
        logging.info(f"Tìm thấy file {output_h5} cũ. Xoá để tạo file mới với cấu trúc chuẩn...")
        os.remove(output_h5)
        
    storm_df = load_and_preprocess_ibtracs(ibtracs_csv)
    process_era5_and_labels(storm_df, output_h5, scaler_file)
