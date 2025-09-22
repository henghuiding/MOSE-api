# MOSEv2: A More Challenging Dataset for Video Object Segmentation in Complex Scenes

🔥 [Evaluation Server](https://www.codabench.org/competitions/10062/) | 🏠 [Homepage](https://mose.video) | 📄 [Paper](https://arxiv.org/pdf/2508.05630) | 🔗 [GitHub](https://github.com/henghuiding/MOSE-api/tree/master/MOSEv2)

## 🔥🔥 Download
- 🤗 [Hugging Face](https://huggingface.co/datasets/FudanCVL/MOSEv2)
- ☁️ [Baidu Pan](https://pan.baidu.com/s/1QWzovOubI0Uvr2gEQChy7Q?pwd=p2m6) (pwd: p2m6)
- ☁️ [Google Drive](https://drive.google.com/drive/folders/1tb6duuZPrjfuHVvJLcWXItsqde3fup4n?usp=drive_link)
- ☁️ [OneDrive](https://1drv.ms/f/c/c2b61e01a0e33ea5/EvcaVXmxT0FCvkZOOa2fXvEBCwEhhwlNJt5jDdM2LMV59w?e=xSLFTf)

## 🔥🔥 Baseline: RCMS
We provide a strong baseline RCMS for MOSEv2 in [here](sam2_rcms). You can also integrate the RCMS migration into your SAM2 method by modifying a few lines of code.

## 🔥🔥 Evaluation

We provides the evaluation code in [here](sam2_rcms/tools/score.py).

## MOSEv2 Dataset

### Dataset Summary

MOSEv2 is a comprehensive video object segmentation dataset designed to advance VOS methods under real-world conditions. It consists of **5,024 videos** and **701,976 high-quality masks** for **10,074 objects** across **200 categories**.

### Dataset Description

Video object segmentation (VOS) aims to segment specified target objects throughout a video. Although state-of-the-art methods have achieved impressive performance (e.g., 90+% J&F) on existing benchmarks such as DAVIS and YouTube-VOS, these datasets primarily contain salient, dominant, and isolated objects, limiting their generalization to real-world scenarios. To advance VOS toward more realistic environments, coMplex video Object SEgmentation (MOSEv1) was introduced to facilitate VOS research in complex scenes. Building on the strengths and limitations of MOSEv1, we present MOSEv2, a significantly more challenging dataset designed to further advance VOS methods under real-world conditions.

MOSEv2 introduces significantly greater scene complexity compared to existing datasets, including:

- **More frequent object disappearance and reappearance**
- **Severe occlusions and crowding**
- **Smaller objects**
- **Adverse weather conditions** (rain, snow, fog)
- **Low-light scenes** (nighttime, underwater)
- **Multi-shot sequences**
- **Camouflaged objects**
- **Non-physical targets** (shadows, reflections)
- **Scenarios requiring external knowledge**

We benchmark 20 representative VOS methods under 5 different settings and observe consistent performance drops. For example, SAM2 drops from 76.4% on MOSEv1 to only 50.9% on MOSEv2. We further evaluate 9 video object tracking methods and find similar declines, demonstrating that MOSEv2 presents challenges across tasks. These results highlight that despite high accuracy on existing datasets, current VOS methods still struggle under real-world complexities.

### Benchmark Results

We evaluated 20 representative VOS methods and observed consistent performance drops compared to simpler datasets:
- **SAM2**: 76.4% (MOSEv1) → 50.9% (MOSEv2)
- Similar declines observed across 9 video object tracking methods




## BibTeX
If you find MOSE helpful to your research, please consider citing our papers.

```
@article{MOSEv2,
  title={{MOSEv2}: A More Challenging Dataset for Video Object Segmentation in Complex Scenes},
  author={Ding, Henghui and Ying, Kaining and Liu, Chang and He, Shuting and Jiang, Xudong and Jiang, Yu-Gang and Torr, Philip HS and Bai, Song},
  journal={arXiv preprint arXiv:2508.05630},
  year={2025}
}
@inproceedings{MOSE,
  title={{MOSE}: A New Dataset for Video Object Segmentation in Complex Scenes},
  author={Ding, Henghui and Liu, Chang and He, Shuting and Jiang, Xudong and Torr, Philip HS and Bai, Song},
  booktitle={ICCV},
  year={2023}
}
```
## License

MOSEv2 is licensed under a [CC BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/). The data of MOSEv2 is released for non-commercial research purpose only.