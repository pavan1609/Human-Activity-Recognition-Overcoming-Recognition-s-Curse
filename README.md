# Human Activity Recognition: Overcoming-the-curse-of-recognition-in-the-wild

<img loop src="teaser.gif" width="100%"/>

[![arXiv](https://img.shields.io/badge/arXiv-2304.05088-b31b1b.svg)](https://arxiv.org/abs/2304.05088)
[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

## Abstract
Though research has shown the complementarity of camera- and inertial-based data, datasets which offer both modalities remain scarce. In this paper, we introduce WEAR, an outdoor sports dataset for both vision- and inertial-based human activity recognition (HAR). The dataset comprises data from 18 participants performing a total of 18 different workout activities with untrimmed inertial (acceleration) and camera (egocentric video) data recorded at 10 different outside locations. Unlike previous egocentric datasets, WEAR provides a challenging prediction scenario marked by purposely introduced activity variations as well as an overall small information overlap across modalities. Provided benchmark results reveal that single-modality architectures each have different strengths and weaknesses in their prediction performance. Further, in light of the recent success of transformer-based temporal action localization models, we demonstrate their versatility by applying them in a plain fashion using vision, inertial and combined (vision + inertial) features as input. Results demonstrate both the applicability of vision-based transformers for inertial data and fusing both modalities by means of simple concatenation, with the combined approach (vision + inertial features) being able to produce the highest mean average precision and close-to-best F1-score. The code to reproduce experiments is publicly available [here](https://github.com/mariusbock/wear). An arXiv version of our paper is available [here](https://arxiv.org/abs/2304.05088).

## Download
The full dataset can be downloaded [here](https://bit.ly/wear_dataset)

The download folder is divided into 3 subdirectories
- **annotations (> 1MB)**: JSON-files containing annotations per-subject using the THUMOS14-style
- **processed (15GB)**: precomputed I3D, inertial and combined per-subject features
- **raw (130GB)**: Raw, per-subject video and inertial data
