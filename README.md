<div align="center">
<h2>FMA-Net (CVPR 2024 Oral)</h2>

<div>    
    <a href='https://sites.google.com/view/geunhyukyouk/' target='_blank'>Geunhyuk Youk</a><sup>1</sup>&nbsp;
    <a href='https://sites.google.com/view/ozbro/' target='_blank'>Jihyong Oh</a><sup>† 2</sup>&nbsp;
    <a href='https://www.viclab.kaist.ac.kr/' target='_blank'>Munchurl Kim</a><sup>† 1</sup>
</div>
<div>
    <sup>†</sup>Co-corresponding authors</span>
</div>
<div>
    <sup>1</sup>Korea Advanced Institute of Science and Technology, South Korea
</div>
<div>
    <sup>2</sup>Chung-Ang University, South Korea
</div>

<div>
    <h4 align="center">
        <a href="https://kaist-viclab.github.io/fmanet-site/" target='_blank'>
        <img src="https://img.shields.io/badge/🐳-Project%20Page-blue">
        </a>
        <a href="https://arxiv.org/abs/2401.03707" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2401.03707-b31b1b.svg">
        </a>
        <a href="https://www.youtube.com/watch?v=kO7KavOH6vw" target='_blank'>
        <img src="https://img.shields.io/badge/Demo%20Video-%23FF0000.svg?logo=YouTube&logoColor=white">
        </a>
                <a href="https://www.youtube.com/watch?v=G6qqJXztJDM" target='_blank'>
        <img src="https://img.shields.io/badge/Presentation-%23FF0000.svg?logo=YouTube&logoColor=white">
        </a>
        <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/KAIST-VICLab/FMA-Net">
    </h4>
</div>

---

<div align="center">
    <h4>
        This repository is the official PyTorch implementation of "FMA-Net: Flow-Guided Dynamic Filtering and Iterative Feature Refinement with Multi-Attention for Joint Video Super-Resolution and Deblurring".
    </h4>
</div>
</div>

## 📧 News
- **Apr 19, 2024:** Codes of FMA-Net (including the training, testing code, and pretrained model) are released :fire:
- **Apr 05, 2024:** FMA-Net is selected for an ORAL presentation at CVPR 2024 (0.78% of 11,532 valid submissions)
- **Feb 27, 2024:** FMA-Net accepted to CVPR 2024 :tada:
- **Jan 14, 2024:** This repository is created

## 📝 TODO
- [x] Release FMA-Net code
- [x] Release pretrained FMA-Net model
- [x] Add data preprocessing scripts


<!-- **Reference**:   -->
## Reference
If you find FMA-Net useful, please consider citing:
```BibTeX
@inproceedings{youk2024fmanet,
  author    = {Geunhyuk Youk and Jihyong Oh and Munchurl Kim},
  title     = {FMA-Net: Flow-Guided Dynamic Filtering and Iterative Feature Refinement with Multi-Attention for Joint Video Super-Resolution and Deblurring},
  booktitle = {CVPR},
  year      = {2024},
 }
```

## Contents
- [Requirements](#requirements)
- [Data Preprocessing](#data-preprocessing)
- [Pretrained Model](#pretrained-model)
- [Training](#training)
- [Testing](#testing)
- [Results](#results)
- [License](#license)
- [Acknowledgement](#acknowledgement)

## Requirements
> - Python 3.9, PyTorch >= 1.9.1
> - Platforms: Ubuntu 22.04, cuda 11.8

## Data Preprocessing
> - Download [REDS](https://seungjunnah.github.io/Datasets/reds.html) dataset
> - Generate REDS4: run ./preprocessing/generate_reds4.py
> - Generate RAFT pseudo-GT optical flow: run ./preprocessing/generate_flow.py (or download the optical flow from [here](https://www.dropbox.com/scl/fo/qgzadp9cqnmzyvghjk4v6/AN-b711qSN5RvakS9VaIpUc?rlkey=6di5bb9um962l8uko1hpiyx1h&st=tzd13ym2&dl=0))

## Pretrained Model
Pre-trained model can be downloaded from [here](https://www.dropbox.com/scl/fo/4392nxna1wptrw06ktv6r/AIyy20JrXK_9CMcXHUQY7Ko?rlkey=n4hhgl7p2c63y3l6lkpqlthi0&st=mnmmvm9y&dl=0).
* *FMA-Net_REDS.zip*: trained on REDS dataset.

## Training
```bash
# download code
git clone https://github.com/KAIST-VICLab/FMA-Net
cd FMA-Net

# train FMA-Net on REDS dataset
python main.py --train --config_path experiment.cfg
```

## Testing
```bash
# test FMA-Net on REDS dataset
python main.py --test --config_path experiment.cfg

# test on your own datasets
python main.py --test_custom --config_path experiment.cfg
```

## Results
Please visit our [project page](https://kaist-viclab.github.io/fmanet-site/) and [demo video](https://www.youtube.com/watch?v=kO7KavOH6vw) for diverse visual results.

## License
The source codes including the checkpoint can be freely used for research and education only. Any commercial use should get formal permission from the principal investigator (Prof. Munchurl Kim, mkimee@kaist.ac.kr).

## Acknowledgement
This work was supported by the Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT): No. 2021-0-00087, Development of high-quality conversion technology for SD/HD low-quality media and No. RS2022-00144444, Deep Learning Based Visual Representational Learning and Rendering of Static and Dynamic Scenes.
