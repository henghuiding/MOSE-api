# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from dataclasses import dataclass
from typing import List

from training.dataset.vos_segment_loader import LazySegments

MAX_RETRIES = 1000


@dataclass
class SampledFramesAndObjects:
    frames: List[int]
    object_ids: List[int]


class VOSSampler:
    def __init__(self, sort_frames=True):
        # frames are ordered by frame id when sort_frames is True
        self.sort_frames = sort_frames

    def sample(self, video):
        raise NotImplementedError()


class RandomUniformSampler(VOSSampler):
    def __init__(
        self,
        num_frames,
        max_num_objects,
        reverse_time_prob=0.0,
        allow_repeat=False,  # 新增参数，控制是否允许 repeat
    ):
        self.num_frames = num_frames
        self.max_num_objects = max_num_objects
        self.reverse_time_prob = reverse_time_prob
        self.allow_repeat = allow_repeat

    def sample(self, video, segment_loader, epoch=None):

        for retry in range(MAX_RETRIES):
            if len(video.frames) < self.num_frames:
                if self.allow_repeat:
                    # 构造 123321123 这种 repeat 序列
                    base = video.frames
                    rev = base[::-1]
                    pattern = base + rev[1:-1]  # 去掉首尾重复
                    # pattern 可能比 num_frames 短，需要多次拼接
                    repeat_times = (self.num_frames + len(pattern) - 1) // len(pattern)
                    full_seq = (pattern * repeat_times)[:self.num_frames]
                    frames = full_seq
                else:
                    raise Exception(
                        f"Cannot sample {self.num_frames} frames from video {video.video_name} as it only has {len(video.frames)} annotated frames."
                    )
            else:
                start = random.randrange(0, len(video.frames) - self.num_frames + 1)
                frames = [video.frames[start + step] for step in range(self.num_frames)]

            if random.uniform(0, 1) < self.reverse_time_prob:
                # Reverse time
                frames = frames[::-1]

            # Get first frame object ids
            visible_object_ids = []
            loaded_segms = segment_loader.load(frames[0].frame_idx)
            if isinstance(loaded_segms, LazySegments):
                # LazySegments for SA1BRawDataset
                visible_object_ids = list(loaded_segms.keys())
            else:
                for object_id, segment in segment_loader.load(
                    frames[0].frame_idx
                ).items():
                    if segment.sum():
                        visible_object_ids.append(object_id)

            # First frame needs to have at least a target to track
            if len(visible_object_ids) > 0:
                break
            if retry >= MAX_RETRIES - 1:
                raise Exception("No visible objects")

        object_ids = random.sample(
            visible_object_ids,
            min(len(visible_object_ids), self.max_num_objects),
        )
        return SampledFramesAndObjects(frames=frames, object_ids=object_ids)


class EvalSampler(VOSSampler):
    """
    VOS Sampler for evaluation: sampling all the frames and all the objects in a video
    """

    def __init__(
        self,
    ):
        super().__init__()

    def sample(self, video, segment_loader, epoch=None):
        """
        Sampling all the frames and all the objects
        """
        if self.sort_frames:
            # ordered by frame id
            frames = sorted(video.frames, key=lambda x: x.frame_idx)
        else:
            # use the original order
            frames = video.frames
        object_ids = segment_loader.load(frames[0].frame_idx).keys()
        if len(object_ids) == 0:
            raise Exception("First frame of the video has no objects")

        return SampledFramesAndObjects(frames=frames, object_ids=object_ids)


class RandomGapSampler(VOSSampler):
    """
    Sampler that samples frames with random gaps (e.g., 1, 2, 3 frames)
    """
    
    def __init__(
        self,
        num_frames,
        max_num_objects,
        reverse_time_prob=0.0,
        # List of possible gaps between frames
        gap_options=[1, 2, 3],
        # Whether to ensure the first frame has visible objects
        ensure_first_frame_visible=True,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.max_num_objects = max_num_objects
        self.reverse_time_prob = reverse_time_prob
        self.gap_options = gap_options
        self.ensure_first_frame_visible = ensure_first_frame_visible

    def _sample_with_gaps(self, video_frames, start_idx):
        """Sample frames with random gaps starting from start_idx"""
        selected_indices = [start_idx]
        current_idx = start_idx
        
        while len(selected_indices) < self.num_frames:
            # Randomly choose a gap from the options
            gap = random.choice(self.gap_options)
            next_idx = current_idx + gap
            
            # Check if we're still within video bounds
            if next_idx >= len(video_frames):
                # If we exceed video length, try to fill with remaining frames
                remaining_frames = len(video_frames) - current_idx - 1
                if remaining_frames > 0:
                    # Use smaller gaps to fit remaining frames
                    for i in range(1, min(remaining_frames + 1, self.num_frames - len(selected_indices) + 1)):
                        if current_idx + i < len(video_frames):
                            selected_indices.append(current_idx + i)
                break
            
            selected_indices.append(next_idx)
            current_idx = next_idx
        
        return selected_indices[:self.num_frames]

    def _find_valid_start_position(self, video_frames, segment_loader):
        """Find a valid starting position that ensures enough frames can be sampled"""
        max_start = len(video_frames) - 1
        
        for retry in range(MAX_RETRIES):
            start_idx = random.randint(0, max_start)
            
            # Check if we can sample enough frames from this position
            selected_indices = self._sample_with_gaps(video_frames, start_idx)
            
            if len(selected_indices) >= self.num_frames:
                # Check if first frame has visible objects (if required)
                if self.ensure_first_frame_visible:
                    first_frame = video_frames[selected_indices[0]]
                    first_frame_segments = segment_loader.load(first_frame.frame_idx)
                    visible_objects = []
                    
                    if isinstance(first_frame_segments, dict):
                        for obj_id, segment in first_frame_segments.items():
                            if segment.sum() > 0:
                                visible_objects.append(obj_id)
                    
                    if len(visible_objects) > 0:
                        return selected_indices, visible_objects
                else:
                    # If we don't need to check first frame visibility, just return
                    return selected_indices, None
        
        raise Exception(f"Could not find valid starting position after {MAX_RETRIES} retries")

    def sample(self, video, segment_loader, epoch=None):
        """Sample frames with random gaps"""
        
        video_frames = video.frames
        
        if len(video_frames) < self.num_frames:
            raise Exception(
                f"Video {video.video_name} has only {len(video_frames)} frames, "
                f"cannot sample {self.num_frames} frames"
            )
        
        # Find valid starting position and sample frames
        selected_indices, visible_objects = self._find_valid_start_position(video_frames, segment_loader)
        selected_frames = [video_frames[i] for i in selected_indices]
        
        # Apply time reversal if needed
        if random.random() < self.reverse_time_prob:
            selected_frames = selected_frames[::-1]
        
        # Get visible objects from first frame
        if visible_objects is None:
            first_frame = selected_frames[0]
            first_frame_segments = segment_loader.load(first_frame.frame_idx)
            visible_objects = []
            
            if isinstance(first_frame_segments, dict):
                for obj_id, segment in first_frame_segments.items():
                    if segment.sum() > 0:
                        visible_objects.append(obj_id)
        
        if len(visible_objects) == 0:
            raise Exception("No visible objects in first frame")
        
        # Select objects
        selected_object_ids = random.sample(
            visible_objects,
            min(len(visible_objects), self.max_num_objects)
        )
        
        return SampledFramesAndObjects(
            frames=selected_frames, 
            object_ids=selected_object_ids
        )
