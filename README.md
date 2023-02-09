# MOSE: A New Dataset for Video Object Segmentation in Complex Scenes

**[🏠[Homepage]](https://henghuiding.github.io/MOSE/)** &emsp; **[📄[Arxiv]](https://arxiv.org/abs/2302.01872)**

This repository contains information and tools for the [MOSE](https://henghuiding.github.io/MOSE/) dataset.


## Download

***[🔥02.09.2023: Dataset has been released!]***

⬇️ Get the dataset from: 

 - ☁️ [OneDrive](https://entuedu-my.sharepoint.com/:f:/g/personal/liuc0058_e_ntu_edu_sg/EjXSfDF7QEZApAVpFJ5rfdABkHCf0k2Va6VDfUy7rpabNw?e=9BVkrz)
 - ☁️ [Google Drive](https://drive.google.com/drive/folders/1vChKHzbboP1k6wd6t95guxxURW3nIXBe?usp=sharing)
 - ☁️ Baidu Pan (Coming Soon)

📦 Or use [gdown](https://github.com/wkentaro/gdown):

 ```bash
 # train.tar.gz
 gdown https://drive.google.com/uc\?id\=10HYO-CJTaITalhzl_Zbz_Qpesh8F3gZR
 # valid.tar.gz
 gdown https://drive.google.com/uc\?id\=1yFoacQ0i3J5q6LmnTVVNTTgGocuPB_hR
 # test.tar.gz
 gdown https://drive.google.com/uc\?id\=186ZEo-FN_vdv3U7X7eGfqIKFpkKIMERl
 ```

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

## License
MOSE is licensed under a [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) License. The data of MOSE is released for non-commercial research purpose only.