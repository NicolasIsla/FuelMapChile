'''
El siguiente es el script utilizado para el procesamiento de las imágenes 
satelitales Sentinel S2 en conjunto con los polígonos de parcelas de cultivos,
para formar tensores (band, time, x, y) con la agregación de productos Sentinel 2
en parches de 256x256 y matrices (x,y) con las anotaciones de cultivos asociadas
a la ubicación geográfica.

Utiliza las funciones definidas en el archivo aux_functions.py

Loopea a través de los tiles encontrados en `s2_path` y para cada tile
genera todos los patches de 256x256 asociados, guardando la metadata actualzada
cada 5 patches generados.
'''
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd 

import rasterio
import geopandas as gpd


from utils import (
    get_id,
    get_crs,
    get_labels_in_tile,
    get_annotation_raster,
    create_patch_tensor_rasterio,
    update_metadata_file,
)


def process_tile(
    tile_name,
    class_mapping,
    s2_path,
    labels_path,
    s2_out_path,
    annotations_out_path,
    metadata_path,
    patch_size,
    grid_padding,
    verbose=False,
    ):
    start = time.time()
    print(f"Formateando tile {tile_name}...")
    sentinel_crs = get_crs(s2_path.glob(f"*{tile_name}*"))

    #Parcelas en tile
    labels_gdf = get_labels_in_tile(
        labels_path=labels_path,
        tile_name=tile_name,
        class_mapping=class_mapping,
        crs=sentinel_crs,
    )

    #Crear tensor con bandas y tiempo, en el patch n-ésimo
    count=0
    metadata_rows = []
    array_size = 10980
    patch_size = 256
    final_n = (array_size//patch_size + 1)**2
    for patch_n in range(0, final_n):
        id = get_id(tile_name, patch_n)
        if verbose: print(f"\tFormateando patch {patch_n} (id={id})...")

        time_series_tensor, raster_data = create_patch_tensor_rasterio(
            products_paths=s2_path.glob(f"*{tile_name}*"),
            patch_n=patch_n,
            patch_size=patch_size,
            padding=grid_padding,
        )
        # Crear el raster de etiquetado
        annotation_raster = get_annotation_raster(
                raster_data, 
                labels_gdf
        )

        # Guardar resultados
        np.save(
            s2_out_path / f"S2_{"{:05}".format(id)}.npy",
           time_series_tensor 
        )
        np.save(
            annotations_out_path / f"ParcelIDs_{"{:05}".format(id)}.npy",
            annotation_raster.astype(np.int16)
        )

        # Guardar metadata cada 5 patches (por si hay detención forzosa)
        metadata_rows.append({
            "id": id,
            "tile_name": tile_name,
            "patch_n": patch_n,
            "parcel_cover": (annotation_raster > 0).sum() / annotation_raster.size,
            "dates_S2": {
                i: pd.to_datetime(date).strftime("%Y%m%d")
                for i, date in enumerate(raster_data.dates)
            },
            "process_num_S2": {
                i: processn
                for i, processn in enumerate(raster_data.process_nums)
            },
            "geometry": raster_data.bounds,
        })
        count = (count + 1)%5
        if count == 0 :
            if verbose: print("Tiempo de ejecución acumulado: ", round((time.time() - start)/60, 2), "[m]")
            #Almacenar metadata
            update_metadata_file(
                new_rows=metadata_rows,
                path=metadata_path,
                crs=sentinel_crs)

    #Almacenar metadata faltante al terminar el tile
    update_metadata_file(
        new_rows=metadata_rows,
        path=metadata_path,
        crs=sentinel_crs)

    end = time.time()
    print("El tiempo de ejecución de la tile es:",
          (end-start)/60 , "m")



@hydra.main(version_base=None, config_path="../../configs/download", config_name="patches_S2")
def main(cfg: DictConfig):

    # se definen las direcciones de los archivos a trabajar
    in_path = Path(cfg.in_path) # dirección del directorio con los datos a procesar.
    s2_path = in_path / "products"
    labels_path = in_path / "combustibles.gpkg"
    assert labels_path.exists(), "No existe archivo con labels"

    out_path = Path(cfg.out_path) # dirección del directorio donde se almacenarán los datos procesados siguiendo el formato de https://huggingface.co/datasets/IGNF/PASTIS-HD/tree/main.
    metadata_path = out_path / f"metadata_{cfg.tile}.geojson" #dirección de la metadata producida en el procesamiento.
    s2_out_path = out_path / "DATA_S2" #dirección de los tensores de imágenes 4D producidos.
    annotations_out_path = out_path / "ANNOTATIONS" #dirección de las matrices 2D con los labels producidos.
    for path in [s2_out_path, annotations_out_path]:
        if not path.exists(): os.makedirs(path)

    # definición mapeo hcat4_code -> crop label class
    class_mapping_path = Path("class_mapping.csv")
    class_mapping = (
        pd.read_csv(class_mapping_path, index_col=0)
        .iloc[:,0]
        .to_dict()
    )

    process_tile(
        cfg.tile,
        class_mapping,
        s2_path,
        labels_path,
        s2_out_path,
        annotations_out_path,
        metadata_path,
        cfg.patch_size,
        cfg.grid_padding,
        verbose=cfg.verbose,
    )

if __name__ == "__main__":
    main()

