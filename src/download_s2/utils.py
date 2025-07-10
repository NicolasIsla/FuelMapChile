import os
import zipfile
import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from datetime import datetime
import shutil
import glob
import rasterio
from rasterio.enums import Resampling
from time import sleep

def rescale_and_copy(path_band, output_band, scale):
    with rasterio.open(path_band) as src:
        data = src.read(
            1,
            out_shape=(int(src.height * scale), int(src.width * scale)),
            resampling=Resampling.bilinear
        )
        transform = src.transform * src.transform.scale(
            src.width / data.shape[1], src.height / data.shape[0]
        )
        with rasterio.open(
            output_band, "w",
            driver="JP2OpenJPEG",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=src.crs,
            transform=transform
        ) as dst:
            dst.write(data, 1)

def download_with_retries(url, zip_path, session, max_retries=3):
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    for attempt in range(max_retries):
        try:
            r = session.get(url, stream=True, timeout=60)
            r.raise_for_status()
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
            return
        except Exception:
            if attempt < max_retries - 1:
                sleep(5)
            else:
                raise

def get_keycloak(username: str, password: str) -> str:
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    r = requests.post(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data=data,
    )
    r.raise_for_status()
    return r.json()["access_token"]

from datetime import datetime, timedelta

def generate_date_pairs(start_date, end_date, n):
    """
    Generate n pairs of equally spaced dates between start_date and end_date.

    Args:
        start_date: str, format 'YYYY-MM-DD'
        end_date: str, format 'YYYY-MM-DD'
        n: int, number of pairs to generate

    Returns:
        List of tuples (start, end) as strings in 'YYYY-MM-DD' format
    """
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    total_days = (end - start).days
    step = total_days / n

    pairs = []
    for i in range(n):
        t0 = start + timedelta(days=i * step)
        t1 = start + timedelta(days=(i + 1) * step)
        pairs.append((t0.strftime("%Y-%m-%d"), t1.strftime("%Y-%m-%d")))

    return pairs
bands_10m = ['B02', 'B03', 'B04', 'B08']
bands_20m = ['B01', 'B05', 'B06', 'B07', 'B11', 'B12']
bands_60m = ['B8A', 'B09']
def download_and_process_sentinel_tile(tile, start_date, end_date, base_path, copernicus_user, copernicus_password):


    


    # === Build query
    query_url = (
        f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
        f"$filter=Collection/Name eq 'SENTINEL-2' and "
        f"contains(Name, '{tile}') and "
        f"not contains(Name, 'N0300') and "
        f"ContentDate/Start ge {start_date}T00:00:00.000Z and "
        f"ContentDate/Start le {end_date}T23:59:59.999Z"
        f"&$count=True&$top=1000"
    )

    response = requests.get(query_url)
    json_ = response.json()
    p = pd.DataFrame.from_dict(json_["value"])
    p["geometry"] = p["GeoFootprint"].apply(shape)
    productDF = gpd.GeoDataFrame(p).set_geometry("geometry")

    productDF = productDF[~productDF["Name"].str.contains("L1C")]
    print(f"✅ Total L2A tiles encontrados: {len(productDF)}")

    productDF["identifier"] = productDF["Name"].str.split(".").str[0]
    productDF["Date"] = pd.to_datetime(productDF["ContentDate"].apply(lambda x: x["Start"]))
    productDF["Tile"] = productDF["identifier"].str.split("_").str[5]
    productDF["Version_num"] = productDF["identifier"].str.split("_").str[3].str[1:].astype(int)

    productDF = productDF.sort_values(by=["Tile", "Date", "Version_num"], ascending=[True, True, False])
    productDF = productDF.drop_duplicates(subset=["Tile", "Date"], keep="first")
    productDF = productDF.sort_values(by=["Version_num", "Date"], ascending=[False, False])
    best_product = productDF.iloc[0]

    identifier = best_product['identifier']
    file_id = best_product['Id']
    tile_name = best_product['Tile']
    date_str = best_product['Date'].strftime('%Y-%m-%d')

    identifier_folder = os.path.join(base_path, tile_name, date_str, identifier)
    zip_path = os.path.join(identifier_folder, f"{identifier}.zip")

    safe_exists = any(f.endswith(".jp2") for f in os.listdir(identifier_folder)) if os.path.exists(identifier_folder) else False
    if safe_exists:
        print(f"🔁 Skipping {identifier}, JP2 already exists.")
        return

    os.makedirs(identifier_folder, exist_ok=True)
    token = get_keycloak(copernicus_user, copernicus_password)
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})

    url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({file_id})/$value"
    redirect = session.get(url, allow_redirects=False)
    while redirect.status_code in (301, 302, 303, 307):
        url = redirect.headers["Location"]
        redirect = session.get(url, allow_redirects=False)

    download_with_retries(url, zip_path, session)

    temp_extract_path = os.path.join(identifier_folder, "temp_extract")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_path)
    os.remove(zip_path)

    safe_dirs = [d for d in os.listdir(temp_extract_path) if d.endswith(".SAFE")]
    if not safe_dirs:
        print("❌ No .SAFE folder found.")
        return

    safe_path = os.path.join(temp_extract_path, safe_dirs[0])
    granule_folder = os.path.join(safe_path, "GRANULE")
    granule_subfolder = os.listdir(granule_folder)[0]
    img_data_path = os.path.join(granule_folder, granule_subfolder, "IMG_DATA")

    for r, b_list, scale in [("R10m", bands_10m, 1), ("R20m", bands_20m, 2), ("R60m", bands_60m, 6)]:
        res_path = os.path.join(img_data_path, r)
        if os.path.exists(res_path):
            for b in b_list:
                for f in glob.glob(os.path.join(res_path, f"*{b}_{r[-3:]}.jp2")):
                    output_name = os.path.basename(f).replace(f"_{r[-3:]}", "_10m") if scale > 1 else os.path.basename(f)
                    dst = os.path.join(identifier_folder, output_name)
                    if scale == 1:
                        shutil.copy2(f, dst)
                    else:
                        rescale_and_copy(f, dst, scale)

    shutil.rmtree(temp_extract_path, ignore_errors=True)
    print(f"✅ Procesamiento completo: {identifier_folder}")