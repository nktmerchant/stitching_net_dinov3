# stitching_net_dinov3

* This public repository is code for training a linear classifier on top of the class token produced by DINOv3 ViT-H+/16 distilled on the [StitchingNet dataset](https://www.kaggle.com/datasets/hyungjung/stitchingnet-dataset)
* The classifier is a linear layer of size (11, 1280), and we provide a checkpoint of its state dictionary under linear_probe_dinov3.pth
* We provide a W&B report detailing the training set up, train and validation loss curves, and example predicitions under [this link](https://wandb.ai/nktmerchant-supermodel-research/linear-probe-stitchingnet-dinov3-cls-token/reports/Linear-probe-of-DINOv3-class-token-on-the-StitchingNet-dataset--VmlldzoxNDg2NDE1Mw?accessToken=hk0df1hw2tyd1ltrgywiv9qxymi556anq1pcwofctm2joeralnip2ykb10s3oga2)
