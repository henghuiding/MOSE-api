# Reliable Conditioned Memory Selection (RCMS)
In this repo, we provide a simple plug-and-play module **RCMS** to enhance the performance of the SAM2 model in complex scenarios, especially to improve the model's performance in the disappearing and reappearing scene.

## Install

```shell
git clone https://github.com/henghuiding/MOSE-api
cd MOSEv2/sam2_rcms

conda create -n sam2_rcms python=3.10 -y 
conda activate sam2_rcms
pip install torch==2.6.0 torchvision  # we recommend using pytorch version 2.6.0
pip install -e ".[dev]"
```
## Training
>**Important**: Before using the config below for training, please first replace the dataset path and model weight path in the config.

We use `SAM2-B+` as an example. Training is divided into two stages: the first stage uses 8 frames with the image encoder unfrozen for training.
```shell
python training/train.py \
    -c configs/sam2.1_training/sam2.1_hiera_b+_MOSEv2_8frames_stage1.yaml \
    --use-cluster 0 \
    --num-gpus 8
```
In the second stage, we use 16 frames (`LVT`) while freezing the image encoder and adopting the `MSS` strategy.
```shell
python training/train.py \
    -c configs/sam2.1_training/sam2.1_hiera_b+_MOSEv2_mss_lvt_16frames_stage2.yaml \
    --use-cluster 0 \
    --num-gpus 8
```

## Evaluation
We provide the pretrained weigths bellow:

| MODEL | <i>J</i> & <i>Ḟ</i> | <i>J</i> | <i>Ḟ</i> | <i>J</i>&<i>Ḟ</i><sub>d</sub> | <i>J</i>&<i>Ḟ</i><sub>r</sub> | <i>F</i> | <i>J</i>&<i>F</i> | Weights | Submission |
|-------|----|----|----|----|----|----|----|---------| ---------- |
| SAM2.1_B+_MOSEv2_MSS_LVT16 | 51.5 | 49.5 | 53.6 | 56.6 | 36.5 | 56.3 | 52.9 | [link](https://huggingface.co/FudanCVL/MOSEv2_baseline/blob/main/sam2.1_hiera_b%2B_MOSEv2_mss_lvt16.pt) | [link](https://huggingface.co/FudanCVL/MOSEv2_baseline/blob/main/sam2_b%2B_MOSEv2_rcms_mqf_mss_lvt_submission.zip) |
| SAM2.1_L_MOSEv2_MSS_LVT16 | 54.4 | 52.4 | 56.3 | 66.8 | 33.2 | 58.9 | 55.6 | [link](https://huggingface.co/FudanCVL/MOSEv2_baseline/blob/main/sam2.1_hiera_l_MOSEv2_mss_lvt16.pt) | [link](https://huggingface.co/FudanCVL/MOSEv2_baseline/blob/main/sam2_l_MOSEv2_rcms_mqf_mss_lvt_submission.zip) |

Use the following code for inference:
```
python tools/vos_inference_ddp.py \
    --sam2_cfg configs/sam2.1/sam2.1_hiera_b+_rcms_mqf_mss_lvt.yaml \
    --sam2_checkpoint /path/to/weights \
    --output_mask_dir example_output \
    --num_gpus 8 \
    --base_video_dir /path/to/image \
    --input_mask_dir /path/to/annotation
```

After obtaining the final output, please package it and upload it to the official [codabench server](https://www.codabench.org/competitions/10062/) for evaluation. The packaging process is as follows, note that the zip command must be executed in the root directory of the output path:

```
cd example_output
zip -r example_submission.zip *
```
## Advanced Usage of RCMS

If you want to use RCMS in your model (e.g., SAM2 variants), you only need to follow the code snippets below to make the modifications:

**Step 1: Add disappearance detection and memory selection logic**

In [`sam2_video_predictor.py`](sam2/sam2_video_predictor.py#L621-L639), around lines 621-639, add the disappearance detection and memory selection logic to handle object disappearance and promote high-quality frames to conditioning memory.

**Step 2 (Optional): Add quality score computation**

In [`sam2_video_predictor.py`](sam2/sam2_video_predictor.py#L818-L831), around lines 818-831, add the quality score computation for memory frames to enable better memory selection.
