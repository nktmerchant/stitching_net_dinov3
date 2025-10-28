# stitching_net_dinov3

* This public repository is code for training a linear classifier on top of the class token produced by DINOv3 ViT-H+/16 distilled on the [StitchingNet dataset](https://www.kaggle.com/datasets/hyungjung/stitchingnet-dataset)
* The classifier is a linear layer of size (11, 1280), and we provide a checkpoint of its state dictionary under linear_probe_dinov3.pth 
