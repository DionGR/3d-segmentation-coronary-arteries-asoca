# SwinUNETRv2 for 3D Medical Image Segmentation - The ASOCA Challenge

In this project, we present a novel 3D medical image segmentation model, SwinUNETRv2, which is based on the Swin Transformer and UNETR. We evaluate the model on the ASOCA Challenge dataset, which is a 3D medical image segmentation dataset for the segmentation of the aorta and pulmonary artery. With very little pretraining and fine-tuning, our model achieves competitive performance on the ASOCA Challenge dataset, with a Dice Score of 0.83. 

## Requirements
- Pretraining and training the model requires a GPU with at least 22GB of VRAM (e.g. NVIDIA RTX4090).
- At least 32GB of RAM is recommended.


## Installation
1. Clone the repository
2. Install the required packages from the `requirements.txt` file:
`pip install -r requirements.txt`
3. Execute the notebooks in the numerical order they are in.

## Data
The ASOCA Challenge dataset is available at [this link](https://asoca.grand-challenge.org/). The dataset consists of 60 3D CT scans of the aorta and pulmonary artery, with corresponding segmentation masks. Out of the 60 scans, 36 are used for training, 4 for validation, and 20 for testing with the official ASOCA Challenge evaluation server and hidden labels.
