from monai.losses import DiceFocalLoss
from monai.inferers import SlidingWindowInferer 

from config.asoca.transforms import train_transforms, val_transforms, SPATIAL_SIZE, BATCH_SIZE

""" Data Loading Parameters """
loader_params = dict(num_workers=8,
                        train_transforms=train_transforms, 
                        val_transforms=val_transforms
                    )

btcv_model_path = "data/checkpoints/btcv/btcv_base.pt"

""" Training Parameters """
train_params = dict(
    img_size=SPATIAL_SIZE,
    in_channels=1,
    out_channels=1,
    feature_size=48,
    use_checkpoint=True,
    use_v2=True,
)

MAX_EPOCHS = 250

loss_function = DiceFocalLoss(sigmoid=True, squared_pred=True)
inferer = SlidingWindowInferer(roi_size=SPATIAL_SIZE, sw_batch_size=BATCH_SIZE, overlap=0.75)

# Optimizer Parameters
optimizer_params = dict(
    lr=2e-4,
    weight_decay=1e-5,
)
