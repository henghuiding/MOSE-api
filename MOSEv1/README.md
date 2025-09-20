# MOSE: A New Dataset for Video Object Segmentation in Complex Scenes

**[🏠[Homepage]](https://henghuiding.github.io/MOSE/)** &emsp; **[📄[Arxiv]](https://arxiv.org/abs/2302.01872)**

This repository contains information and tools for the [MOSE](https://henghuiding.github.io/MOSE/) dataset.


## Download

***[🔥02.09.2023: Dataset has been released!]***

⬇️ Get the dataset from: 

 - ☁️ [***OneDrive*** ](https://entuedu-my.sharepoint.com/:f:/g/personal/liuc0058_e_ntu_edu_sg/EjXSfDF7QEZApAVpFJ5rfdABkHCf0k2Va6VDfUy7rpabNw?e=9BVkrz) ***(Recommended)***
 - ☁️ [Google Drive](https://drive.google.com/drive/folders/1vChKHzbboP1k6wd6t95guxxURW3nIXBe?usp=sharing)
 - ☁️ [Baidu Pan](https://pan.baidu.com/s/116p3tQsUqObem8G8FOJ7cA) (Access Code: MOSE) 


📦 Or use [gdown](https://github.com/wkentaro/gdown):

 ```bash
 # train.tar.gz
 gdown 'https://drive.google.com/uc?id=ID_removed_to_avoid_overaccesses_get_it_by_yourself'
 
 # valid.tar.gz
 gdown 'https://drive.google.com/uc?id=ID_removed_to_avoid_overaccesses_get_it_by_yourself'
 
 # test set will be released when competition starts.
 ```

Please also check the SHA256 sum of the files to ensure the data intergrity:

```
3f805e66ecb576fdd37a1ab2b06b08a428edd71994920443f70d09537918270b train.tar.gz
884baecf7d7e85cd35486e45d6c474dc34352a227ac75c49f6d5e4afb61b331c valid.tar.gz
```


## Evaluation

***[🔥02.16.2023: Our [CodaLab competition](https://codalab.lisn.upsaclay.fr/competitions/10703) is on live now!]***

Please submit your results on 
 - 💯 [**CodaLab**](https://codalab.lisn.upsaclay.fr/competitions/10703).


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


## BibTeX
Please consider to cite MOSE if it helps your research.

```latex
@inproceedings{MOSE,
  title={{MOSE}: A New Dataset for Video Object Segmentation in Complex Scenes},
  author={Ding, Henghui and Liu, Chang and He, Shuting and Jiang, Xudong and Torr, Philip HS and Bai, Song},
  booktitle={ICCV},
  year={2023}
}
```

## License
MOSE is licensed under a [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) License. The data of MOSE is released for non-commercial research purpose only.
