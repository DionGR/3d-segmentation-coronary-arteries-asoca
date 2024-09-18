DATAROOT_PATH = ""
DATALIST_AUTOSEG3D_PATH_IMAGECAS = "data/datasets/imagecas/datalist_auto3d.json"
ANALYZER_OUT_PATH_IMAGECAS = "data/data_analysis/imagecas/stats.yaml"

config_imagecas = {
    "name": "IMAGECAS_CTCA",
    "task": "segmentation",  
    "modality": "CT", 
}

analyzer_cfg_imagecas = {
    "dataroot": DATAROOT_PATH, 
    "datalist": DATALIST_AUTOSEG3D_PATH_IMAGECAS, 
    "output_path": ANALYZER_OUT_PATH_IMAGECAS,
    "worker": 4,
}