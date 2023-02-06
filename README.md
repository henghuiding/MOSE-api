# MOSE: A New Dataset for Video Object Segmentation in Complex Scenes

[[Homepage]](https://henghuiding.github.io/MOSE/) &emsp; [[Arxiv]](https://arxiv.org/abs/2302.01872)

This repository contains information and tools for the [MOSE](https://henghuiding.github.io/MOSE/) dataset.


## Download

Get the dataset from: [OneDrive](https://entuedu-my.sharepoint.com/:f:/g/personal/liuc0058_e_ntu_edu_sg/EjXSfDF7QEZApAVpFJ5rfdABkHCf0k2Va6VDfUy7rpabNw?e=9BVkrz)


## File Structure

The dataset follows a similar structure as [DAVIS](https://davischallenge.org/) and [Youtube-VOS](https://youtube-vos.org/). The dataset consists of two parts: `JPEGImages` which holds the frame images, and `Annotations` which contains the corresponding segmentation masks. The frame images are numbered using five-digit numbers. Annotations are saved in color-pattlate mode PNGs like DAVIS.

Please note that while annotations for all frames in the training set are provided, annotations for the validation set will only include the first frame.

```
<train/valid.tar>
│
├── Annotations
│ │ 
│ ├── <video_name_1>
│ │ ├── 00000.png
│ │ ├── 00001.png
│ │ └── ...
│ │ 
│ ├── <video_name_2>
│ │ ├── 00000.png
│ │ ├── 00001.png
│ │ └── ...
│ │ 
│ ├── <video_name_...>
│ 
└── JPEGImages
  │ 
  ├── <video_name_1>
  │ ├── 00000.jpg
  │ ├── 00001.jpg
  │ └── ...
  │ 
  ├── <video_name_2>
  │ ├── 00000.jpg
  │ ├── 00001.jpg
  │ └── ...
  │ 
  └── <video_name_...>

```


## Evaluation

Please submit your results on CodaLab.

(Link will be released before 17th Feb. Stay tuned!)


## BibTeX
Please consider to cite MOSE if it helps your research.

```latex
@article{MOSE,
  title={MOSE: A New Dataset for Video Object Segmentation in Complex Scenes},
  author={Ding, Henghui and Liu, Chang and He, Shuting and Jiang, Xudong and Torr, Philip HS and Bai, Song},
  journal={arXiv preprint arXiv:2302.01872},
  year={2023}
}
```
