<div align="center">
  <h1> Devil's on the Edges: <br> Selective Quad Attention for Scene Graph Generation</h1>
</div>
<div align="center">
  <h3><a href=https://hesedjds.github.io/>Deunsol Jung</a> &nbsp;&nbsp;&nbsp;&nbsp; <a href=.>Sanghyun Kim</a> &nbsp;&nbsp;&nbsp;&nbsp; <a href=http://aimi.postech.ac.kr/members/>Won Hwa Kim</a> &nbsp;&nbsp;&nbsp;&nbsp; <a href=http://cvlab.postech.ac.kr/~mcho/>Minsu Cho</a></h3>
</div>

<div align="center">
  <img src="assets/squat.png" alt="result"/>
</div>

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/pytorch-1.10.0-ee4c2c)](https://pytorch.org/)
[![PyTorch](https://img.shields.io/badge/python-3.9-3776ab)](https://www.python.org/)

This repo is the official implementation of the CVPR 2023 paper: [Devil's on the Edges: Selective Quad Attention for Scene Graph Generation](https://arxiv.org/abs/2304.03495).

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Dataset

Check [DATASET.md](DATASET.md) for instructions of dataset preprocessing.

## Training **(IMPORTANT)**

### Prepare Faster-RCNN Detector
- You can download the pretrained Faster R-CNN we used in the paper: 
  - [VG](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EQIy64T-EK9Er9y8kVCDaukB79gJwfSsEIbey9g0Xag6lg?e=wkKHJs), 
  - [OIv6](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EfGXxc9byEtEnYFwd0xdlYEBcUuFXBjYxNUXVGkgc-jkfQ?e=lSlqnz)
- put the checkpoint into the folder:
```
mkdir -p checkpoints/detection/pretrained_faster_rcnn/
# for VG
mv /path/vg_faster_det.pth checkpoints/detection/pretrained_faster_rcnn/
```

Then, you need to modify the pretrained weight parameter `MODEL.PRETRAINED_DETECTOR_CKPT` in configs yaml `configs/e2e_relSQUAT_[vg, oiv6].yaml` to the path of corresponding pretrained rcnn weight to make sure you load the detection weight parameter correctly.

### Scene Graph Generation Model
You can follow the following instructions to train your own, which takes 4 GPUs for train each SGG model. The results should be very close to the reported results given in paper.

We provide the one-click script for training our SQUAT model in `scripts/train.sh`. 
You can simply replace the `--config-file` options with config file below to reproduce our model. 

Or you can copy the following command to train
```
gpu_num=4 && python -m torch.distributed.launch --master_port 10028 --nproc_per_node=$gpu_num \
       tools/relation_train_net.py \
       --config-file "configs/e2e_relSQUAT_vg.yaml" \
        EXPERIMENT_NAME "SQUAT-3-3" \
        SOLVER.IMS_PER_BATCH $[3*$gpu_num] \
        TEST.IMS_PER_BATCH $[$gpu_num] \
        SOLVER.VAL_PERIOD 1000 \
        SOLVER.CHECKPOINT_PERIOD 1000 \
        MODEL.ROI_RELATION_HEAD.PREDICTOR SquatPredictor \
        MODEL.ROI_RELATION_HEAD.SQUAT_MODULE.RHO 0.7 \
        MODEL.ROI_RELATION_HEAD.SQUAT_MODULE.BETA 0.7
```

## Test
Similarly, we also provide the `scripts/test.sh` for directly produce the results from the checkpoint provide by us.
By replacing the parameter of `archive_dir` and `model_name` to the directory and file name for trained model weight and selected dataset name in `DATASETS.TEST`, you can directly eval the model on validation or test set.

## Model Zoo 
We provide the pre-trained model weights and config files which was used for training the each model. 
You should use the RHO and BETA as written in the table for evaluating the pre-trained models. 
### Visual Genome 
|      Task        | RHO | BETA | mR@20 | mR@50 | mR@100 |  R@20 | R@50  | R@100  |                      Link (Google Drive)                     |
| :--------------: |:---:|:---: | :---: | :---: |  :---: | :---: | :---: | :----: | :----------------------------------------------------------: |
| PredCls          | 0.5 | 1.0  | 25.64 | 30.87 | 33.41  | 48.01 | 55.67 | 57.94  |  [Link](https://drive.google.com/drive/folders/1_S90m0TIZxOD8qjyJtfnhn1AHiAW0Y-N?usp=drive_link)  |
| SGCls            | 0.5 | 1.0  | 14.35 | 17.47 | 18.87  | 28.87 | 32.92 | 34.26  |  [Link](https://drive.google.com/drive/folders/1zF-3eL9_993LdAK_f5xxLrV6kTvzxjsh?usp=drive_link)  |
| SGDet            | 0.35| 0.7  | 10.57 | 14.12 | 16.47  | 17.85 | 24.51 | 28.93  |  [Link](https://drive.google.com/drive/folders/1rkTxRiPP_EzmiRbUGY0BnFiri5v1Rn9J?usp=drive_link)  |


### OpenImages 
|  RHO | BETA | R@50  | rel | phr  | final_score | Link (Google Drive) | 
| :---: |:---:|:---: | :---: | :---: |  :---: | :---: | 
| 0.6 | 0.7 | 75.8 | 34.9 | 35.9 | 43.5 | [Link](https://drive.google.com/drive/folders/167eeNYRoCuJwrIvFSKlmRG-uwra2x0h9?usp=drive_link) |

## Citations

If you find this project helps your research, please kindly consider citing our papers in your publications.

```
@inproceedings{jung2023devil,
  title={Devil's on the Edges: Selective Quad Attention for Scene Graph Generation},
  author={Jung, Deunsol and Kim, Sanghyun and Kim, Won Hwa and Cho, Minsu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```


## Acknowledgment
This repository is developed on top of the scene graph benchmarking framwork develped by [SHTUPLUS](https://github.com/SHTUPLUS/PySGG)
