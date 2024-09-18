from monai.transforms import (
    LoadImageD,
    SpacingD,
    ScaleIntensityRangeD,
    OrientationD,
    RandCropByPosNegLabeld,
    EnsureChannelFirstD,
    RandAffineD,
    EnsureTypeD,
    AsDiscreteD,
    ActivationsD,
    KeepLargestConnectedComponentD,
)

import numpy as np

from utils.utils import get_device


""" Transforms Configurations """
KEYS_TR = ["image", "label"]
KEYS_PR = ["pred"]

# Intensity Range
A_MIN = -175
A_MAX = 604
B_MIN = 0.0
B_MAX = 1.0

# Sizing
SPATIAL_SIZE = (128, 128, 128)
PIX_DIM = (0.414, 0.414, 0.5)

# Misc
PRED_THRESHOLD = 0.75
BATCH_SIZE = 2

""" Data Transforms """

# Transforms that apply to all the phases

key_transforms = [        
        LoadImageD(keys=KEYS_TR, image_only=False),
        EnsureChannelFirstD(keys=KEYS_TR, channel_dim='no_channel'),
        ScaleIntensityRangeD(keys=KEYS_TR[0], a_min=A_MIN, a_max=A_MAX, b_min=B_MIN, b_max=B_MAX, clip=True),
        OrientationD(keys=KEYS_TR, axcodes="RAS"),
        SpacingD(keys=KEYS_TR, pixdim=PIX_DIM, mode=("bilinear", "nearest")),
    ]

""" For Training/Validation Phase """

train_transforms = key_transforms.copy() \
                        +         \
                    [
                        RandCropByPosNegLabeld(keys=KEYS_TR, image_key=KEYS_TR[0], label_key=KEYS_TR[1], 
                               spatial_size=SPATIAL_SIZE, pos=2, neg=1, num_samples=BATCH_SIZE,
                               image_threshold=0),
                        
                        RandAffineD(keys=KEYS_TR,
                                mode=('bilinear', 'nearest'),
                                prob=0.25,
                                spatial_size=SPATIAL_SIZE,
                                rotate_range=(0, 0, np.pi/12),
                                scale_range=(0.1, 0.1, 0.1)),
                    ]

val_transforms = key_transforms.copy()

train_post_transforms = [
                            ActivationsD(keys=KEYS_PR, sigmoid=True),
                            AsDiscreteD(keys=KEYS_PR, threshold=PRED_THRESHOLD),
                            KeepLargestConnectedComponentD(keys=KEYS_PR, applied_labels=[1], num_components=3),
                        ]

val_post_transforms = [
                        EnsureTypeD(keys=KEYS_PR),
                        ActivationsD(keys=KEYS_PR, sigmoid=True),
                        AsDiscreteD(keys=KEYS_PR, threshold=PRED_THRESHOLD),
                        KeepLargestConnectedComponentD(keys=KEYS_PR, applied_labels=[1], num_components=3),
                      ]

""" For EDA Phase """

eda_transforms = [
        LoadImageD(keys=KEYS_TR, image_only=False),
        EnsureChannelFirstD(keys=KEYS_TR, channel_dim='no_channel'),
        EnsureTypeD(keys=KEYS_TR),
    ]