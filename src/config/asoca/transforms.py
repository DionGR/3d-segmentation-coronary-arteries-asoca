from monai.transforms import (
    LoadImageD,
    SpacingD,
    EnsureTypeD,
    EnsureTypeD,
    ScaleIntensityRangeD,
    OrientationD,
    RandCropByPosNegLabeld,
    RandAffineD,
    EnsureTypeD,
    AsDiscreteD,
    ActivationsD,
    KeepLargestConnectedComponentD,
    InvertD,
    SaveImageD,
    Compose,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd
)

import numpy as np


""" Transforms Configurations """
KEYS_TR = ["image", "label"]
KEYS_PR = ["pred"]

# Intensity Range
A_MIN = 150
A_MAX = 625
B_MIN = 0.0
B_MAX = 1.0

# Sizing
SPATIAL_SIZE = (128, 128, 128)
PIX_DIM = (0.48, 0.48, 0.62)

# Misc
PRED_THRESHOLD = 0.75
BATCH_SIZE = 2

""" Data Transforms """

# Transforms that apply to all the phases
key_transforms = [        
        LoadImageD(keys=KEYS_TR, image_only=True, ensure_channel_first=True), 
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
                                prob=0.5,
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

""" For Inference Phase """
undoable_transforms = [
        OrientationD(keys=KEYS_TR[0], axcodes="RAS"),
        SpacingD(keys=KEYS_TR[0], pixdim=PIX_DIM, mode=("bilinear")),
]

test_transforms = [        
        LoadImageD(keys=KEYS_TR[0], image_only=False, ensure_channel_first=True), 
        ScaleIntensityRangeD(keys=KEYS_TR[0], a_min=A_MIN, a_max=A_MAX, b_min=B_MIN, b_max=B_MAX, clip=True),
    ] + undoable_transforms.copy()

test_post_transforms = [
    InvertD(keys="pred", 
        transform=Compose(undoable_transforms.copy()), 
        orig_keys="image", 
        meta_keys="pred_meta", 
        orig_meta_keys="image_meta_dict", 
        meta_key_postfix="meta_dict", 
        nearest_interp=False, 
        to_tensor=True),
    EnsureTypeD(keys=KEYS_PR),
    ActivationsD(keys=KEYS_PR, sigmoid=True),
    AsDiscreteD(keys=KEYS_PR, threshold=PRED_THRESHOLD),
    KeepLargestConnectedComponentD(keys=KEYS_PR, applied_labels=[1], num_components=3),
    
    SaveImageD(keys=KEYS_PR, 
               meta_keys=KEYS_TR[0], 
               output_dir="data/preds/out_raw/", 
               output_postfix="seg",
               resample=False,
               separate_folder=False),
    ]

""" For EDA Phase """

eda_transforms = [
        LoadImageD(keys=KEYS_TR, image_only=True, ensure_channel_first=True),
        EnsureTypeD(keys=KEYS_TR),
    ]