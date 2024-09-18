from monai.apps.auto3dseg import (
    DataAnalyzer,
)

from config.imagecas.analyzer import analyzer_cfg_imagecas


""" 
Data Analysis for ImageCAS Dataset using Auto3dSeg 
Executed on IDUN for all the cases in our ImageCAS subset
"""

analyser = DataAnalyzer(**analyzer_cfg_imagecas)

datastat = analyser.get_all_case_stats(key="training",
                                       transform_list=None)