import os
import zipfile
import requests
import shutil
import glob
from datetime import datetime, timedelta
from shapely.geometry import shape
import pandas as pd
import geopandas as gpd
from omegaconf import DictConfig
import hydra

from utils import get_keycloak, rescale_and_copy, bands_10m, bands_20m, bands_60m

def generate_date_pairs(start_date: str, end_date: str, n: int):
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    step = (end - start).days / n
    pairs = []
    for i in range(n):
        t0 = start + timedelta(days=i * step)
        t1 = start + timedelta(days=(i + 1) * step)
        pairs.append((t0.strftime("%Y-%m-%d"), t1.strftime("%Y-%m-%d")))
    return pairs

def download_and_process(tile, start_date, end_date, base_path, user, password):
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
    productDF = pd.DataFrame.from_dict(response.json()["value"])
    productDF["geometry"] = productDF["GeoFootprint"].apply(shape)
    productDF = gpd.GeoDataFrame(productDF).set_geometry("geometry")
    productDF = productDF[~productDF["Name"].str.contains("L1C")]

    if productDF.empty:
        print(f"No data found for {tile} from {start_date} to {end_date}")
        return

    productDF["identifier"] = productDF["Name"].str.split(".").str[0]
    productDF["Date"] = pd.to_datetime(productDF["ContentDate"].apply(lambda x: x["Start"]))
    productDF["Tile"] = productDF["identifier"].str.split("_").str[5]
    productDF["Version_num"] = productDF["identifier"].str.split("_").str[3].str[1:].astype(int)
    productDF = productDF.sort_values(by=["Tile", "Date", "Version_num"], ascending=[True, True, False])
    productDF = productDF.drop_duplicates(subset=["Tile", "Date"], keep="first")
    best_product = productDF.sort_values(by=["Version_num", "Date"], ascending=[False, False]).iloc[0]

    identifier = best_product["identifier"]
    file_id = best_product["Id"]
    tile_name = best_product["Tile"]
    date_str = best_product["Date"].strftime("%Y-%m-%d")

    identifier_folder = os.path.join(base_path, tile_name, date_str, identifier)
    zip_path = os.path.join(identifier_folder, f"{identifier}.zip")
    if os.path.exists(identifier_folder) and any(f.endswith(".jp2") for f in os.listdir(identifier_folder)):
        print(f"Skipping {identifier}, already processed.")
        return

    os.makedirs(identifier_folder, exist_ok=True)
    token = get_keycloak(user, password)
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})
    url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({file_id})/$value"
    r = session.get(url, allow_redirects=False)
    while r.status_code in (301, 302, 303, 307):
        url = r.headers["Location"]
        r = session.get(url, allow_redirects=False)
    r = session.get(url, stream=True)
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    temp_path = os.path.join(identifier_folder, "temp_extract")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_path)
    os.remove(zip_path)

    safe_dirs = [d for d in os.listdir(temp_path) if d.endswith(".SAFE")]
    if not safe_dirs:
        print("❌ No .SAFE folder found.")
        return

    safe_path = os.path.join(temp_path, safe_dirs[0])
    granule_folder = os.path.join(safe_path, "GRANULE", os.listdir(os.path.join(safe_path, "GRANULE"))[0], "IMG_DATA")

    for r, b_list, scale in [("R10m", bands_10m, 1), ("R20m", bands_20m, 2), ("R60m", bands_60m, 6)]:
        res_path = os.path.join(granule_folder, r)
        if os.path.exists(res_path):
            for b in b_list:
                for f in glob.glob(os.path.join(res_path, f"*{b}_{r[-3:]}.jp2")):
                    name = os.path.basename(f).replace(f"_{r[-3:]}", "_10m") if scale > 1 else os.path.basename(f)
                    dst = os.path.join(identifier_folder, name)
                    if scale == 1:
                        shutil.copy2(f, dst)
                    else:
                        rescale_and_copy(f, dst, scale)
    shutil.rmtree(temp_path, ignore_errors=True)

@hydra.main(version_base=None, config_path="../../configs/download", config_name="sentinel2")
def main(cfg: DictConfig):
    date_ranges = generate_date_pairs(cfg.start_date, cfg.end_date, cfg.n)
    for start, end in date_ranges:
        download_and_process(cfg.tile, start, end, cfg.base_path, cfg.user, cfg.password)
        print(f"Processed {cfg.tile} from {start} to {end}")

if __name__ == "__main__":
    main()