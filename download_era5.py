import cdsapi
import time
import os

URL = "https://cds.climate.copernicus.eu/api"
KEY = "79218e52-0a03-450f-b115-5fdc4ab09ee2"
c = cdsapi.Client(url=URL, key=KEY)

# Biển Đông
bounding_box = [25, 100, 0, 125]

years = ['2021', '2022', '2023', '2024', '2025']
months = ['06', '07', '08', '09', '10', '11', '12']
days = [str(i).zfill(2) for i in range(1, 32)]
times = ['00:00', '06:00', '12:00', '18:00']

os.makedirs('ERA5_Data', exist_ok=True)

for year in years:

    print(f"\n{'='*50}")
    print(f"BẮT ĐẦU TẢI DỮ LIỆU NĂM {year}")
    print(f"{'='*50}")

    file_surface = f"ERA5_Data/surface_{year}.nc"
    file_pressure = f"ERA5_Data/pressure_{year}.nc"

    # =====================================================
    # 1. SURFACE LEVELS
    # =====================================================

    print("[1/2] Đang tải SURFACE variables...")

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'data_format': 'netcdf',

            'variable': [

                # Áp suất
                'mean_sea_level_pressure',

                # Nhiệt độ biển
                'sea_surface_temperature',

                # Gió bề mặt
                '10m_u_component_of_wind',
                '10m_v_component_of_wind',

                # Hơi nước toàn cột khí quyển
                'total_column_water_vapour',

                # Mưa
                'total_precipitation',
            ],

            'year': year,
            'month': months,
            'day': days,
            'time': times,
            'area': bounding_box,
        },
        file_surface
    )

    print("⏳ Nghỉ 10 giây...")
    time.sleep(10)

    # =====================================================
    # 2. PRESSURE LEVELS
    # =====================================================

    print("[2/2] Đang tải PRESSURE variables...")

    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'data_format': 'netcdf',

            'variable': [
                #độ ẩm
                'relative_humidity'

                # Độ xoáy
                'vorticity',

                # Gió tầng cao
                'u_component_of_wind',
                'v_component_of_wind',
                # độ cao của một mặt áp suất trong khí quyển
                'geopotential',

            ],

            'pressure_level': [
                '500',
                '850'
            ],

            'year': year,
            'month': months,
            'day': days,
            'time': times,
            'area': bounding_box,
        },
        file_pressure
    )

    print(f"✅ Hoàn tất năm {year}")

    if year != years[-1]:
        print("⏳ Chờ 30 giây tránh rate limit...")
        time.sleep(30)

print("\n🎉 HOÀN TẤT TOÀN BỘ ERA5 PIPELINE")
