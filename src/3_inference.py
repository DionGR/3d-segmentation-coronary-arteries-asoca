import torch
from utils.utils import get_device
from monai.networks.nets import SwinUNETR
from monai.transforms import Compose
from monai.engines import SupervisedEvaluator
from utils.data_loaders.asoca_loader import ASOCALoader
from config.asoca.transforms import test_transforms, test_post_transforms
from config.asoca.swin_unetr import train_params, inferer
from monai.handlers import StatsHandler, TensorBoardImageHandler, from_engine
import gc 

# Assuming the device setup and data loader configuration are the same
gc.collect()
torch.cuda.empty_cache()
device = get_device()

# Load data
ASOCALoader = ASOCALoader(train_transforms=[], val_transforms=[], test_transforms=test_transforms)
test_loader = ASOCALoader.get_test_dataloader()

# Initialize the model
model = SwinUNETR(**train_params).to(device)

# Load the model weights from checkpoint
torch.backends.cudnn.benchmark = True

model.load_state_dict(torch.load("data/checkpoints/asoca/best_model.pt"), strict=False)

model.eval()
model.to(device)

# Validation handlers for inference verification (like visual aid and stats logging)
val_handlers = [
    StatsHandler(name="test_log", 
                 output_transform=lambda x: None),
    
    TensorBoardImageHandler(
        log_dir="data/checkpoints/tensorboard/asoca/testing/",
        batch_transform=from_engine(["image"]),
        output_transform=from_engine(["pred"])
    )
]

# Set up the evaluator for inference
evaluator = SupervisedEvaluator(
    device=device,
    val_data_loader=test_loader,
    network=model,
    inferer=inferer,
    postprocessing=Compose(test_post_transforms),
    val_handlers=val_handlers,
    amp=True,
)

# Run evaluation/inference
try:
    evaluator.run()
except IndexError:
    pass
except Exception as e:
    print(f"An error occurred during inference: {e}")
finally:
    print("Inference completed!")
    