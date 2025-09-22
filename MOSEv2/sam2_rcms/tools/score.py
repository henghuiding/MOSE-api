'''
Inference code for MOSE
Modified from DETR (https://github.com/facebookresearch/detr)
'''

import json
import multiprocessing as mp
import os
import time
import argparse

from PIL import Image
import numpy as np
from metricsv2 import db_eval_boundary, db_eval_iou
from track_progress_rich import track_progress_rich

SPLIT = 'valid'

MAX_WORKERS = mp.cpu_count()

try:
    import psutil
    MAX_MEMORY_GB = psutil.virtual_memory().total / (1024**3)
except:
    MAX_MEMORY_GB = "unknown"
MAX_FRAMES_PER_CHUNK = 15

print(f"System Info:")
print(f"  Physical CPU cores: {MAX_WORKERS}")
print(f"  Total memory: {MAX_MEMORY_GB} GB")
print(f"  Max frames per chunk: {MAX_FRAMES_PER_CHUNK}")


def process_video_chunk(vid_name, meta_file, pred_dir, mask_dir, chunk_start, chunk_end, anno_id, idx):
    """Process a chunk of frames for a specific object in a video"""
    start_time = time.time()
    
    vid_meta = meta_file[vid_name]
    vid_len = vid_meta['length']
    h, w = vid_meta['height'], vid_meta['width']

    ann_path = os.path.join(pred_dir, vid_name)
    
    if not os.path.exists(f'{pred_dir}/{vid_name}'):
        return {
            'vid_name': vid_name,
            'anno_id': anno_id,
            'chunk_start': chunk_start,
            'chunk_end': chunk_end,
            'j_list': [],
            'f_list': [],
            'f_list_': [],
            'empty_gt': [],
            'error': f'{vid_name} not found',
            'time': time.time() - start_time
        }

    anno_names = [x for x in os.listdir(ann_path) if not x.startswith('.') and x.endswith('.png')]
    anno_names = sorted(anno_names, key=lambda x: int(x.split('.')[0]))
    
    # Get h,w from first frame if not provided
    if h is None or w is None:
        first_frame = anno_names[0].split('.')[0]
        first_gt = np.asarray(Image.open(f'{mask_dir}/{vid_name}/{first_frame}.png').convert('P'))
        h, w = first_gt.shape

    # Process chunk frames (1 to vid_len-1, excluding first and last)
    chunk_size = chunk_end - chunk_start
    gt_anns = np.zeros((chunk_size, h, w), dtype=np.uint8)
    pred_anns = np.zeros((chunk_size, h, w), dtype=np.uint8)

    for i, frame_idx in enumerate(range(chunk_start, chunk_end)):
        if frame_idx < 1 or frame_idx >= vid_len - 1:
            continue
        assert frame_idx < len(anno_names), f'frame_idx: {frame_idx}, len(anno_names): {len(anno_names)}, video: {vid_name}'
        frame_name = anno_names[frame_idx].split('.')[0]

        gt_anns[i] = np.asarray(
            Image.open(f'{mask_dir}/{vid_name}/{frame_name}.png').convert('P'))

        if os.path.exists(f'{pred_dir}/{vid_name}/{frame_name}.png'):
            pred_anns[i] = np.asarray(
                Image.open(f'{pred_dir}/{vid_name}/{frame_name}.png').convert('P'))

            if pred_anns[i].shape != (h, w):
                raise ValueError(f'Wrong shape: {vid_name}/{frame_name}')
        else:
            raise ValueError(f'Frame not found: {vid_name}/{frame_name}')

    # Calculate metrics for this chunk and object
    pred_mask = (pred_anns == int(anno_id)).astype('uint8')
    gt_masks = (gt_anns == int(anno_id)).astype('uint8')

    j_list, empty_gt = db_eval_iou(gt_masks, pred_mask)
    f_list_, f_list = db_eval_boundary(gt_masks, pred_mask, bound_th=0.1, bound_mode='area')
    
    return {
        'vid_name': vid_name,
        'anno_id': anno_id,
        'chunk_start': chunk_start,
        'chunk_end': chunk_end,
        'j_list': j_list.tolist(),
        'f_list': f_list.tolist(),
        'f_list_': f_list_.tolist(),
        'empty_gt': empty_gt.tolist(),
        'error': None,
        'time': time.time() - start_time
    }


def merge_chunk_results(chunk_results, vid_name, anno_id):
    """Merge results from multiple chunks for a single object"""
    # Filter and sort chunks for this video and object
    relevant_chunks = [r for r in chunk_results if r['vid_name'] == vid_name and r['anno_id'] == anno_id]
    relevant_chunks.sort(key=lambda x: x['chunk_start'])
    
    # Check for errors
    errors = [r['error'] for r in relevant_chunks if r['error'] is not None]
    if errors:
        return [0, 0, 0, None, None, None, None]
    
    # Merge all metrics
    j_list = []
    f_list = []
    f_list_ = []
    empty_gt = []
    
    for chunk in relevant_chunks:
        j_list.extend(chunk['j_list'])
        f_list.extend(chunk['f_list'])
        f_list_.extend(chunk['f_list_'])
        empty_gt.extend(chunk['empty_gt'])
    
    if not j_list:
        return [0, 0, 0, None, None, None, None]
    
    j_list = np.array(j_list)
    f_list = np.array(f_list)
    f_list_ = np.array(f_list_)
    empty_gt = np.array(empty_gt, dtype=bool)
    
    j, f = j_list.mean(), f_list.mean()
    f_ = f_list_.mean()
    
    # Handle empty frames (disappearance) and reappearance
    # Find where objects disappear (empty_gt is True) and where they reappear
    disappear_frames = np.where(~empty_gt[:-1] & empty_gt[1:])[0] + 1
    if empty_gt[0] == 1:
        disappear_frames = np.insert(disappear_frames, 0, 0)
    reappear_frames = np.where(empty_gt[:-1] & ~empty_gt[1:])[0] + 1
    
    # Calculate J&F for disappearance segments (from start to end of disappearance)
    disappear_j, disappear_f = None, None
    if len(disappear_frames) > 0:
        # For each disappearance segment
        disappear_segments = []
        for i, start_frame in enumerate(disappear_frames):
            # Find end of this disappearance (next reappearance or end of sequence)
            end_frame = reappear_frames[i] if i < len(reappear_frames) else len(empty_gt)
            
            # Get the frame right before disappearance starts
            if start_frame > 0:
                # Collect metrics for the entire disappearance segment
                segment_j = j_list[start_frame:end_frame].mean()  # Last visible frame
                segment_f = f_list[start_frame:end_frame].mean()
                disappear_segments.append((segment_j, segment_f))
        
        if disappear_segments:
            # Average over all disappearance segments
            disappear_j = np.mean([seg[0] for seg in disappear_segments])
            disappear_f = np.mean([seg[1] for seg in disappear_segments])
    
    # Calculate J&F for reappearance frames
    reappear_j, reappear_f = None, None
    if len(reappear_frames) > 0:
        # For each reappearance segment
        reappear_segments = []
        for i, start_frame in enumerate(reappear_frames):
            # Find end of this reappearance (next disappearance or end of sequence)
            end_frame = disappear_frames[i+1] if i+1 < len(disappear_frames) else len(empty_gt)
            
            # Collect metrics for the entire reappearance segment
            segment_j = j_list[start_frame:end_frame].mean()
            segment_f = f_list[start_frame:end_frame].mean()
            reappear_segments.append((segment_j, segment_f))
        
        if reappear_segments:
            # Average over all reappearance segments
            reappear_j = np.mean([seg[0] for seg in reappear_segments])
            reappear_f = np.mean([seg[1] for seg in reappear_segments])

    return [j, f, f_, disappear_j, disappear_f, reappear_j, reappear_f]


def main():
    parser = argparse.ArgumentParser(description="MOSE metric evaluation")
    parser.add_argument('--meta', type=str, required=True, help='Path to meta.json file')
    parser.add_argument('--ann', type=str, required=True, help='Path to ground truth Annotations directory')
    parser.add_argument('--pred', type=str, required=True, help='Path to prediction directory (res)')
    parser.add_argument('--output', type=str, required=True, help='Path to output directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    meta_json_path = args.meta
    mask_dir = args.ann
    pred_dir = args.pred
    output_dir = args.output
    debug = args.debug

    # If pred_dir contains only one subfolder, use that as pred_dir
    pred_folders = [x for x in os.listdir(pred_dir) if not x.startswith('.')]
    if len(pred_folders) == 1 and os.path.isdir(os.path.join(pred_dir, pred_folders[0])):
        pred_dir = os.path.join(pred_dir, pred_folders[0])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # meta.json is a file, not a directory
    with open(meta_json_path, 'r') as fp:
        meta_file = json.load(fp)['videos']

    # video_list is the list of video folders in mask_dir
    video_list = list(os.listdir(mask_dir))

    print('Checking Integrity...', end=' ')
    for vid in video_list:
        if meta_file[vid]['length'] == len(os.listdir(os.path.join(mask_dir, vid))):
            continue
        else:
            raise ValueError(f'Server-side error ({vid}) - Please contact the organizers')
    print('Done')

    video_num = len(video_list)
    print(f'Number of videos: {video_num}')

    # Prepare tasks for parallel processing - split videos into chunks
    tasks = []
    for vid_name in video_list:
        vid_meta = meta_file[vid_name]
        vid_len = vid_meta['length']
        objs = vid_meta['objects']
        
        # Split video into chunks (excluding first and last frame: 1 to vid_len-1)
        effective_frames = vid_len - 2
        num_chunks = (effective_frames + MAX_FRAMES_PER_CHUNK - 1) // MAX_FRAMES_PER_CHUNK
        
        for chunk_idx in range(num_chunks):
            chunk_start = 1 + chunk_idx * MAX_FRAMES_PER_CHUNK
            chunk_end = min(1 + (chunk_idx + 1) * MAX_FRAMES_PER_CHUNK, vid_len - 1)
            
            for anno_id in objs:
                tasks.append((vid_name, meta_file, pred_dir, mask_dir, chunk_start, chunk_end, anno_id))
    
    start_time = time.time()
    print(f'Starting metric evaluation with {len(tasks)} chunk tasks')
    
    # Process chunks in parallel using track_progress_rich
    chunk_results = track_progress_rich(
        process_video_chunk,
        tasks,
        nproc=MAX_WORKERS,
        description="Evaluating video chunks"
    )
    
    # Merge chunk results for each video-object combination
    output_dict = {}
    total_time = 0
    
    # Print header
    print(f"{'Video':<50}, {'J':>8}, {'F_new':>8}, {'F':>8}, {'disappear_J':>12}, {'disappear_F_new':>15}, {'reappear_J':>11}, {'reappear_F_new':>14}")
    
    for vid_name in video_list:
        vid_meta = meta_file[vid_name]
        objs = vid_meta['objects']
        
        for anno_id in objs:
            exp_name = f'{vid_name}_{anno_id}'
            merged_result = merge_chunk_results(chunk_results, vid_name, anno_id)
            output_dict[exp_name] = merged_result
            
            # Print results
            # merged_result: [j, f, f_, disappear_j, disappear_f, reappear_j, reappear_f]
            j, f, f_, disappear_j, disappear_f, reappear_j, reappear_f = merged_result
            disappear_j_str = round(disappear_j*100, 2) if disappear_j is not None else "N/A"
            disappear_f_str = round(disappear_f*100, 2) if disappear_f is not None else "N/A"
            reappear_j_str = round(reappear_j*100, 2) if reappear_j is not None else "N/A"
            reappear_f_str = round(reappear_f*100, 2) if reappear_f is not None else "N/A"
            print(f'{vid_name:>50}, {j*100:>8.2f}, {f*100:>8.2f}, {f_*100:>8.2f}, {disappear_j_str:>12}, {disappear_f_str:>15}, {reappear_j_str:>11}, {reappear_f_str:>14}')
    
    # Calculate total processing time
    for result in chunk_results:
        if result is not None:
            total_time += result['time']
    
    print('Metric evaluation finished')
    # Convert metrics to percentages (multiply by 100) and round to 2 decimal places
    formatted_output_dict = {}
    for key, value in output_dict.items():
        if isinstance(value, (list, tuple)) and len(value) >= 7:
            # Format the metrics: j, f, f_, disappear_j, disappear_f, reappear_j, reappear_f
            formatted_metrics = {
                'J&F_new': round((value[0] + value[1]) / 2 * 100, 2) if value[0] is not None and value[1] is not None else None,
                'J': round(value[0] * 100, 2) if value[0] is not None else None,
                'F_new': round(value[1] * 100, 2) if value[1] is not None else None,
                'disappear_J&F_new': round((value[3] + value[4]) / 2 * 100, 2) if value[3] is not None and value[4] is not None else None,
                'reappear_J&F_new': round((value[5] + value[6]) / 2 * 100, 2) if value[5] is not None and value[6] is not None else None,
                'F': round(value[2] * 100, 2) if value[2] is not None else None,
                'J&F': round((value[0] + value[2]) / 2 * 100, 2) if value[0] is not None and value[2] is not None else None,
            }
            formatted_output_dict[key] = formatted_metrics
        else:
            formatted_output_dict[key] = value
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        formatted_output_dict['total_time'] = total_time
        json.dump(formatted_output_dict, f, indent=4)

    j = [output_dict[x][0] for x in output_dict if x != 'total_time']
    f = [output_dict[x][1] for x in output_dict if x != 'total_time']
    f_ = [output_dict[x][2] for x in output_dict if x != 'total_time']
    disappear_j = [output_dict[x][3] for x in output_dict if x != 'total_time']
    disappear_j = [x for x in disappear_j if x is not None]
    disappear_f = [output_dict[x][4] for x in output_dict if x != 'total_time']
    disappear_f = [x for x in disappear_f if x is not None]
    reappear_j = [output_dict[x][5] for x in output_dict if x != 'total_time']
    reappear_j = [x for x in reappear_j if x is not None]
    reappear_f = [output_dict[x][6] for x in output_dict if x != 'total_time']
    reappear_f = [x for x in reappear_f if x is not None]

    J = round(np.mean(j) * 100, 2)
    F_new = round(np.mean(f) * 100, 2)
    JF_new = round((np.mean(j) + np.mean(f)) / 2 * 100, 2)
    disappear_JF_new = round((np.mean(disappear_j) + np.mean(disappear_f)) / 2 * 100, 2)
    reappear_JF_new = round((np.mean(reappear_j) + np.mean(reappear_f)) / 2 * 100, 2)
    F = round(np.mean(f_) * 100, 2)
    JF = round((np.mean(j) + np.mean(f_)) / 2 * 100, 2)

    print(f'J&F_new: {JF_new}')
    print(f'J: {J}')
    print(f'F_new: {F_new}')
    print(f'disappear_J&F_new: {disappear_JF_new}')
    print(f'reappear_J&F_new: {reappear_JF_new}')
    print(f'F: {F}')
    print(f'J&F: {JF}')
    print(f'Copy-Paste: {JF_new:.1f},{J:.1f},{F_new:.1f},{disappear_JF_new:.1f},{reappear_JF_new:.1f},{F:.1f},{JF:.1f}')
    
    with open(os.path.join(output_dir, 'scores.txt'), 'w') as fp:
        fp.write(f'J&F_new: {JF_new}\n')
        fp.write(f'J: {J}\n')
        fp.write(f'F_new: {F_new}\n')
        fp.write(f'disappear_J&F_new: {disappear_JF_new}\n')
        fp.write(f'reappear_J&F_new: {reappear_JF_new}\n')
        fp.write(f'F: {F}\n')
        fp.write(f'J&F: {JF}\n')
        fp.write(f'Copy-Paste: {JF_new:.1f},{J:.1f},{F_new:.1f},{disappear_JF_new:.1f},{reappear_JF_new:.1f},{F:.1f},{JF:.1f}\n')
    end_time = time.time()
    total_time = end_time - start_time
    print("time: %.4f s" %(total_time))


if __name__ == '__main__':
    main()
