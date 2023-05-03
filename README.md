# SeiT: Storage-efficient Vision Training

Official Pytorch implementation of SeiT | [Paper](https://arxiv.org/abs/2303.11114)

[Song Park](https://8uos.github.io/) &nbsp; [Sanghyuk Chun](https://sanghyukchun.github.io/home/) &nbsp; [Byeongho Heo](https://sites.google.com/view/byeongho-heo/home) &nbsp; [Wonjae Kim](https://wonjae.kim/) &nbsp; [Sangdoo Yun](https://sangdooyun.github.io/)

[NAVER AI LAB](https://naver-career.gitbook.io/en/teams/clova-cic)

## Abstract

We need billion-scale images to achieve more generalizable and ground-breaking vision models, as well as massive dataset storage to ship the images (e.g., the LAION-4B dataset needs 240TB storage space). However, it has become challenging to deal with unlimited dataset storage with limited storage infrastructure. A number of storage-efficient training methods have been proposed to tackle the problem, but they are rarely scalable or suffer from severe damage to performance. In this paper, we propose a storage-efficient training strategy for vision classifiers for large-scale datasets (e.g., ImageNet) that only uses 1024 tokens per instance without using the raw level pixels; our token storage only needs <1% of the original JPEG-compressed raw pixels. We also propose token augmentations and a Stem-adaptor module to make our approach able to use the same architecture as pixel-based approaches with only minimal modifications on the stem layer and the carefully tuned optimization settings. Our experimental results on ImageNet-1k show that our method significantly outperforms other storage-efficient training methods with a large gap. We further show the effectiveness of our method in other practical scenarios, storage-efficient pre-training, and continual learning.


## Usage

### Requirements
- Python3
- Pytorch (> 1.7)
- timm (0.5.4)

### Training
#### Downloading datasets
- The ImageNet-1k token datasets can be downloaded from [here]()
  - The tar file contains all the files required to train the ViT model; tokens, codebook and pre-defined token-synonym dictionary.
  - Download the file and place the extracted files under same directory for convinience (we will call it to DATA_DIR.)
  
#### Training examples
- We used 8 V100 GPUs to train ViT-B with ImageNet-1k Tokens.
```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --model deit_base_token_32 \
    --codebook-path DATA_DIR/codebook.ckpt \
    --train-token-file DATA_DIR/imagenet1k-train-token.data.bin.gz \
    --train-label-file DATA_DIR/imagenet1k-train-label.txt \
    --val-token-file DATA_DIR/imagenet1k-val-token.data.bin.gz \
    --val-label-file DATA_DIR/imagenet1k-val-label.txt \
    --token-synonym-dict DATA_DIR/synonyms.json \
    --output_dir path/to/save \
    --batch-size 128 \
    --dist-eval
```
  
  
### Evaluation
#### Preparing pre-trained weights
- The pre-trained weights of Tokenizer and ViT models can be downloaded from [here]().
  - Place the extracted files to under same directory for convinence (we will call it to WEIGHT_DIR.)
- The tokenizer source code is required to load the tokenizer weight (can be downloaded from [here](https://github.com/thuanz123/enhancing-transformers.git)).

#### Evaluation on images
- The testing images should be organized to an ImageNet1k-like structure, which can be loaded appropriately by pytorch ImageFolder dataset.

```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --use_env evaluate_on_images.py \
    --model deit_base_token_32 \
    --model-path WEIGHT_DIR/trained-vit.ckpt \    
    --codebook-path WEIGHT_DIR/codebook.ckpt \
    --tokenizer-path WEIGHT_DIR/tokenizer.ckpt \
    --tokenizer-code-path path/to/downloaded/tokenizer/repository \
    --data-path path/to/test/dataset \
    --dist-eval
```

## Acknowledgements

This repository is a forked version of DeiT: [facebookresearch/deit](https://github.com/facebookresearch/deit).

Also, we are heavily inspired by ViT-VQGAN: [thuanz123/enhancing-transformers(unofficial)](https://github.com/thuanz123/enhancing-transformers).

## Citation

```
@article{park2023seit,
    title={SeiT: Storage-Efficient Vision Training with Tokens Using 1% of Pixel Storage},
    author={Park, Song and Chun, Sanghyuk and Heo, Byeongho and Kim, Wonjae and Yun, Sangdoo},
    year={2023},
    journal={arXiv preprint arXiv:2303.11114},
}
```

## License
```
SeiT
Copyright (c) 2023-present NAVER Cloud Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
