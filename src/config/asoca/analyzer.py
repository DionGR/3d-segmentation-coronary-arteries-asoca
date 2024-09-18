DATAROOT_PATH = ""
DATALIST_AUTOSEG3D_PATH_ASOCA = "data/datasets/asoca/datalist_auto3d.json"
ANALYZER_OUT_PATH_ASOCA = "data/data_analysis/asoca/stats.yaml"

config_asoca = {
    "name": "ASOCA_CTCA",
    "task": "segmentation",  
    "modality": "CT", 
}

analyzer_cfg_asoca = {
    "dataroot": DATAROOT_PATH, 
    "datalist": DATALIST_AUTOSEG3D_PATH_ASOCA, 
    "output_path": ANALYZER_OUT_PATH_ASOCA,
    "worker": 4,
}
