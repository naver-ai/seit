# SeiT++: Masked Token Modeling Improves Storage-efficient Training (ECCV 2024)
Official Pytorch implementation of SeiT++ | [Paper](https://arxiv.org/abs/2312.10105)

 [Minhyun Lee](https://scholar.google.com/citations?user=2hUlCnQAAAAJ&hl=ko) &nbsp; [Song Park](https://8uos.github.io/) &nbsp; [Byeongho Heo](https://sites.google.com/view/byeongho-heo/home) &nbsp; [Dongyoon Han](https://sites.google.com/site/dyhan0920/) &nbsp; [Hyunjung Shim](https://scholar.google.com/citations?user=KB5XZGIAAAAJ&hl=en) 

[NAVER AI LAB](https://naver-career.gitbook.io/en/teams/clova-cic)

## Abstract

Recent advancements in Deep Neural Network (DNN) models have significantly improved performance across computer vision tasks. However, achieving highly generalizable and high-performing vision models requires expansive datasets, resulting in significant storage requirements. This storage challenge is a critical bottleneck for scaling up models. A recent breakthrough by SeiT proposed the use of Vector-Quantized (VQ) feature vectors (i.e., tokens) as network inputs for vision classification. This approach achieved 90% of the performance of a model trained on full-pixel images with only 1% of the storage. While SeiT needs labeled data, its potential in scenarios beyond fully supervised learning remains largely untapped. In this paper, we extend SeiT by integrating Masked Token Modeling (MTM) for self-supervised pre-training. Recognizing that self-supervised approaches often demand more data due to the lack of labels, we introduce TokenAdapt and ColorAdapt. These methods facilitate comprehensive token-friendly data augmentation, effectively addressing the increased data requirements of self-supervised learning. We evaluate our approach across various scenarios, including storage-efficient ImageNet-1k classification, fine-grained classification, ADE-20k semantic segmentation, and robustness benchmarks. Experimental results demonstrate consistent performance improvement in diverse experiments, validating the effectiveness of our method.


## Usage

### Requirements
- Python3
- Pytorch (> 1.7)
- timm (0.5.4)

### Training
#### Downloading datasets
- The ImageNet-1k token datasets can be downloaded from [here](https://github.com/naver-ai/seit/releases)
  - The tar file contains all the files required to train the ViT model; tokens, codebook and pre-defined token-synonym dictionary.
  - Download the file and place the extracted files under same directory for convinience

#### Training examples
- We used 8 V100 GPUs to train ViT-B with ImageNet-1k Tokens.
```
  DATA_DIR="path/data_dir"
  # Pre-training MTM
  OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --use_env main_mtm_pretrain.py \
	  --model mae_token_vit_base_patch16_dec512d4b_k2s2_vitvqgan_32 \
    --codebook-path ${DATA_DIR}/codebook.ckpt \
    --train-token-file ${DATA_DIR}/imagenet1k-train-token.data.bin.gz \
    --train-label-file ${DATA_DIR}/imagenet1k-train-label.txt \
	  --batch_size 128 \
	  --accum_iter 4 \
	  --blr 1.5e-4 \
	  --epochs 400 \
    --output_dir path/to/save \
    --tokenadapt-path tokenadapt.ckpt    
  # finetuning MTM
  OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --use_env main_mtm_finetune.py \
	  --model deit_base_token_32 \
    --codebook-path ${DATA_DIR}/codebook.ckpt \
    --train-token-file ${DATA_DIR}/imagenet1k-train-token.data.bin.gz \
    --train-label-file ${DATA_DIR}/imagenet1k-train-label.txt \
    --val-token-file ${DATA_DIR}/imagenet1k-val-token.data.bin.gz \
    --val-label-file ${DATA_DIR}/imagenet1k-val-label.txt \
    --token-synonym-dict tokens/codebook-synonyms.json \
	  --batch_size 128 \
	  --accum_iter 1 \
	  --print_freq 400 \
	  --epochs 100 \
    --output_dir path/to/save \
	  --blr 1.0e-3 --layer_decay 0.65 \
	  --dist_eval \
	  --finetune path/to/pre-trained model
```  

## Acknowledgements

This repository is heavily borrowed brom SeiT: [naver-ai/seit](https://github.com/naver-ai/seit).

## License
```
SeiT++
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
```
