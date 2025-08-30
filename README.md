# Adversarial Attacks on BLIP-2: A Gray-Box Approach for Image-to-Text Models

## Research Project – Based on and extending the work “I See Dead People: Gray-Box Adversarial Attack on Image-To-Text Models” (ECML-PKDD 2023)

This repository contains code and experiments for adversarial attacks on modern image-to-text systems, focusing on the BLIP-2 model.
Our implementation adapts and extends the gray-box attack framework from:
[I See Dead People: Gray-Box Adversarial Attack on Image-To-Text Models", ECML-PKDD Machine Learning and Cybersecurity Workshop, 2023](https://arxiv.org/abs/2306.07591).

### Abstract:
Image-to-text models such as BLIP-2 represent a new generation of captioning systems that combine strong vision encoders with large language models for accurate and fluent caption generation. Despite their advancements, these models remain vulnerable to adversarial perturbations.
We present a gray-box adversarial attack targeting BLIP-2, extending prior work on ViT-GPT2. Unlike text classification attacks, adversarial image captioning is challenging due to the open-ended output space of natural language. Our attack leverages only the image encoder for optimization, making it decoder-agnostic and effective across different captioning models.
We implement a Projected Gradient Descent (PGD)-based optimization with hybrid L2 and L∞ constraints, combined with semantic losses based on CLIP and Sentence-BERT (SBERT). This allows perturbations to be both semantically guided and visually imperceptible.
Experiments on the Flickr30k dataset show that our attack can successfully fool BLIP-2 into producing targeted captions, while maintaining strong imperceptibility under norm constraints.

## Prerequisites
    conda create -n blip2_env python=3.10.9
    pip install -r requirements.txt

## Download the Flickr30k dataset
1. Download dataset from this link: [dataset](https://www.kaggle.com/datasets/adityajn105/flickr30k)
2. Update the FLICKR_PATH variable in utils.py accordingly

## Run
    python attack_targeted.py --model=<model_name> --dataset=<dataset_name> --eps=<epsilon> --n_epochs=<num_epochs> --n_imgs=<n_images>

- The values used in the paper are:
  - model=blip2
  - dataset=flickr30k
  - eps= 0.12-0.2
  - n_epochs=300
  - images=1000

If you wish to cite this paper:
```
@article{lapid2023see,
  title={I See Dead People: Gray-Box Adversarial Attack on Image-To-Text Models},
  author={Lapid, Raz and Sipper, Moshe},
  journal={arXiv preprint arXiv:2306.07591},
  year={2023}
}
```
![alt text](https://github.com/razla/I-See-Dead-People-Gray-Box-Adversarial-Attack-on-Image-To-Text-Models/blob/main/figures/examples.png)
