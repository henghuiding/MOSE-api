# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time

import numpy as np
import torch
from tqdm import tqdm

from sam2.build_sam import build_sam2_video_predictor

# Only cuda supported
assert torch.cuda.is_available()
device = torch.device("cuda")

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

# Config and checkpoint
# sam2_checkpoint = "checkpoints/sam2.1_hiera_base_plus.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

# sam2_checkpoint = "/home/knying/yingkaining/project/tpami_mosev2/ourbaseline/SAM2BoT/sam2_logs/sam2.1_hiera_l_MOSEv2_finetune_final2/checkpoints/checkpoint.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Build video predictor with vos_optimized=True setting
predictor = build_sam2_video_predictor(
    model_cfg, sam2_checkpoint, device=device, vos_optimized=False, num_prev_frames_to_promote=0, low_quality_threshold=1.
)


# Initialize with video
video_dir = "/home/knying/yingkaining/project/tpami_mosev2/mose_annotation/prepare_mosev2_release/MOSEv2_release_final/valid/JPEGImages/8i47h67j"
# scan all the JPEG frame names in this directory
frame_names = [
    p
    for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
][:200]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
inference_state = predictor.init_state(video_path=video_dir, offload_video_to_cpu=True)


# Number of runs, warmup etc
warm_up, runs = 2, 10
verbose = True
num_frames = len(frame_names)
total, count = 0, 0
torch.cuda.empty_cache()

# 记录初始显存占用
initial_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)

# We will select an object with a click.
# See video_predictor_example.ipynb for more detailed explanation
ann_frame_idx, ann_obj_id = 0, 1
# Add a positive click at (x, y) = (210, 350)
# For labels, `1` means positive click
points = np.array([[210, 350]], dtype=np.float32)
labels = np.array([1], np.int32)
# 读取mask而不是使用点击
mask_path = "/home/knying/yingkaining/project/tpami_mosev2/mose_annotation/prepare_mosev2_release/MOSEv2_release_final/valid/Annotations/8i47h67j/00000.png"
import cv2
from PIL import Image
import numpy as np
mask = np.array(Image.open(mask_path).convert("P"))
# 获取像素值为1的位置作为mask
binary_mask = mask > 0

_, out_obj_ids, out_mask_logits = predictor.add_new_mask(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    mask=binary_mask
)

# 记录最大显存占用
max_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)

# Warmup and then average FPS over several runs
with torch.autocast("cuda", torch.bfloat16):
    with torch.inference_mode():
        for i in tqdm(range(runs), disable=not verbose, desc="Benchmarking"):
            start = time.time()
            # Start tracking
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in predictor.propagate_in_video(inference_state):
                pass

            end = time.time()
            total += end - start
            count += 1
            if i == warm_up - 1:
                print("Warmup FPS: ", count * num_frames / total)
                total = 0
                count = 0
            
            # 更新最大显存占用
            current_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
            max_memory = max(max_memory, current_memory)

print("FPS: ", count * num_frames / total)
print(f"最大显存占用: {max_memory:.2f} MB")
print(f"初始显存占用: {initial_memory:.2f} MB")
print(f"推理过程额外占用: {max_memory - initial_memory:.2f} MB")
