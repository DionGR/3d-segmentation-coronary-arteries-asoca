from utils.utils import get_device
from utils.data_loaders.asoca_loader import ASOCALoader
from config.asoca.transforms import train_post_transforms, val_post_transforms
from config.asoca.swin_unetr import loader_params, train_params, inferer, btcv_model_path, imagecas_model_path, optimizer_params, loss_function, MAX_EPOCHS, USE_IMAGECAS

import torch
import gc
import time 

from monai.networks.nets import SwinUNETR
from monai.handlers import StatsHandler, CheckpointSaver, MeanDice, from_engine, EarlyStopHandler, HausdorffDistance, TensorBoardStatsHandler, TensorBoardImageHandler, LrScheduleHandler, ValidationHandler
from monai.inferers import SimpleInferer
from monai.engines import SupervisedEvaluator
from monai.engines import SupervisedTrainer
from monai.transforms import Compose

from monai.optimizers import LinearLR

gc.collect()
torch.cuda.empty_cache()
device = get_device()


""" Data Loading """
ASOCALoader = ASOCALoader(**loader_params)
train_loader, val_loader = ASOCALoader.get_train_val_dataloaders()

""" Model Initialization """
model = SwinUNETR(**train_params).to(device)

if USE_IMAGECAS:
    model.load_state_dict(torch.load(imagecas_model_path), strict=False)
    print("Using pretrained ImageCAS backbone weights!")
else:
    weight = torch.load(btcv_model_path)
    model.load_from(weights=weight)
    print("Using pretrained BTCV backbone weights!")

torch.backends.cudnn.benchmark = True


""" Optimizer and Scheduler"""
optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params)
lr_scheduler = LinearLR(optimizer, num_iter=6, end_lr=2e-4)

""" Validation """
val_handlers = [
    EarlyStopHandler(trainer=None, 
                     patience=25, 
                     score_function=lambda x: x.state.metrics["val_mean_dice"]
                     ),
    
    StatsHandler(name="val_log", 
                 output_transform=lambda x: None),
    
    TensorBoardStatsHandler(log_dir="data/tensorboard/asoca/training/", 
                            output_transform=lambda x: None,
                            global_epoch_transform=lambda x: trainer.state.epoch),
    
    TensorBoardImageHandler(
        log_dir="data/tensorboard/asoca/training/",
        batch_transform=from_engine(["image", "label"]),
        output_transform=from_engine(["pred"])
        ),
    
    CheckpointSaver(save_dir="data/checkpoints/asoca/", 
                    save_dict={"model": model},
                    save_key_metric=True,
                    key_metric_save_state=True,
                    key_metric_name="val_mean_dice",
                    ),
]

evaluator = SupervisedEvaluator(
    device=device,
    val_data_loader=val_loader,
    network=model,
    inferer=inferer,
    postprocessing=Compose(val_post_transforms),
    val_handlers=val_handlers,
    key_val_metric={
        "val_mean_dice": MeanDice(include_background=True, output_transform=from_engine(["pred", "label"])),
        "train_hd95": HausdorffDistance(include_background=True, percentile=95, output_transform=from_engine(["pred", "label"]))
    },
    amp=True,
)

""" Training """
train_handlers = [
    LrScheduleHandler(lr_scheduler=lr_scheduler, 
                      print_lr=True),
    
    ValidationHandler(validator=evaluator, 
                      interval=10,
                      epoch_level=True),

    StatsHandler(name="train_log", 
                 tag_name="train_loss", 
                 output_transform=from_engine(["loss"], first=True)),
    
    TensorBoardStatsHandler(
        log_dir="data/tensorboard/asoca/training/", 
        tag_name="train_loss", 
        output_transform=from_engine(["loss"], first=True)
    ),
]
    
trainer = SupervisedTrainer(
    device=device,
    max_epochs=MAX_EPOCHS,
    train_data_loader=train_loader,
    network=model,
    optimizer=optimizer,
    loss_function=loss_function,
    inferer=SimpleInferer(),
    postprocessing=Compose(train_post_transforms),
    key_train_metric={ 
                      "train_mean_dice": MeanDice(include_background=True, output_transform=from_engine(["pred", "label"])),
                      },
    train_handlers=train_handlers,
    amp=True,
)
val_handlers[0].set_trainer(trainer=trainer)

""" Actual Training """

try:
    train_start = time.time()
    trainer.run()
except KeyboardInterrupt:
    pass
finally:
    train_end = time.time()
    total = train_end - train_start
    if total > 120:
        print(f"Training time: {total/60} minutes", flush=True)
        with open("data/exec_log.txt", "a") as f:
            date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write(f"[{date}] Training time: {total/60} minutes\n")