# The Hard Positive Truth about Vision-Language Compositionality (ECCV 2024)

Code and datasets for "The Hard Positive Truth about Vision-Language Compositionality" at ECCV 2024 [[paper](https://amitakamath.github.io/Hard_Positive.pdf)]. 

This code is based on the code for the ICLR 2023 paper by Yuksekgonul et al, **When and why vision-language models behave like bags-of-words, and what to do about it?** [[paper](https://openreview.net/pdf?id=KRLUvxh8uaX)][[code](https://github.com/mertyg/vision-language-models-are-bows)].

Please shoot me an email at [kamatha@cs.washington.edu](mailto:kamatha@cs.washington.edu) if you have any questions about the code+data, or if you'd just like to chat about this (or related) work!

<p align="center">
<img src="figures/teaser.jpg" width="700">
</p>

Here's more information about the data, followed by instructions about how to use this repository to reproduce our results in the paper.

# Datasets
In our work, we put forward two evaluation benchmarks: `SWAP` and `REPLACE`. `REPLACE` replaces either an attribute or a relation in the original caption _c_ to obtain _c<sub>n</sub>_ and _c<sub>p</sub>_. `SWAP` swaps object-attribute associations in the original caption _c_ to obtain _c<sub>n</sub>_ and _c<sub>p</sub>_. 

<p align="center">
<img src="figures/dataset_figure.jpg" width="500">
</p>

We also programmatically generate 591,753 hard positives for finetuning from COCO data. CLIP is finetuned on these, in combination with 591,753 hard negatives and 591,753 original captions. 

## Benchmarks
`data/` contains the original caption, hard negative, and VG image paths for evaluation.

`swapped_data/` contains the hard positive, hard negative, and VG image paths for evaluation.

In each of these, `visual_genome_attribution.json` contains `SWAP` data, `vl_checklist_attributes.json` contains `REPLACE` attributes data, and `vl_checklist_relations.json` contains `REPLACE` relations data.

Each JSON file is a list of dictionaries which contain the image ID, correct caption, and incorrect caption for each example. The code to load and evaluate each dataset is in `dataset_zoo/aro_datasets.py`.

## Finetuning data

`hard_positives_ft/` contains the hard positive captions generated for _finetuning_. 

The `SWAP` hard positives are in `swap_hp.jsonl` and the `REPLACE` hard positives in `replace_hp.jsonl`. Each line corresponds to the corresponding original caption in COCO. In cases where the original COCO caption did not contain the word "and", we cannot swap while retaining meaning, and use "None" as a placeholder in the file (these are not used for finetuning).


# Reproducing the results
## Setting Up
Clone the repository, then create and activate a new conda environment and install the packages you'll need.
```
git clone https://github.com/amitakamath/hard_positives.git
cd hard_positives
conda create --name hard_positives
conda activate hard_positives
pip install -r requirements.txt
```

## Downloading the data
The data lives in `hard_positives/data` and `hard_positives/swapped_data` (your models will go to the former as they're downloaded). You can change this in `dataset_zoo/constants.py` and `model_zoo/constants.py`, if you're so inclined.  

## Running experiments with the models in the repo
This repository contains wrappers for several models (listed later in this section). You can run experiments with any of these models on any of the datasets in the repository to reproduce the paper results, or alternately write a wrapper for a new model and see how it does on our benchmarks. 

To get the results of `model_name` on `dataset_name`, run the following command:
```
python run_eval.py --dataset=$dataset --model-name=$model_name
```
This will print the Original Test Accuracy, Augmented Test Accuracy and Brittleness (refer to Section 4.1 of the paper for definitions), i.e. Table 1 of the paper. You can also print out more specific categories, e.g., the number of instances where _s(c) > s(c<sub>n</sub>) > s(c<sub>p</sub>)_ (currently commented out in the code), which are used to generate Table 2. 

Note: Table 1 of the paper averages `REPLACE` relations and attributes.

We currently support the following vision-language models: [OpenAI CLIP](https://github.com/openai/CLIP) models; [LAION OpenCLIP](https://github.com/mlfoundations/open_clip) models; [NegCLIP](https://github.com/mertyg/vision-language-models-are-bows); [CoCa](https://arxiv.org/abs/2205.01917); and [XVLM](https://github.com/zengyan-97/X-VLM). 

We further support the following hard-negative finetuned models: CREPE-finetuned models on different types of hard negatives (swap, replace, negate, and all); SVLC finetuned on positives only, rule-based negatives, LLM-generated and rule-based negatives, LLM-generated and rule-based negatives and positives; DAC finetuned on LLM positives and SVLC negatives, on SAM positives and SVLC negatives; Our model finetuned on hard positives, hard negatives, and both (of different types and at different ratios); and CLIP finetuned on just COCO data. These models are discussed in the paper. Here are the corresponding `model_name` values you would use in the command: 
```
openai-clip:ViT-B/32
openai-clip:ViT-B/16
openai-clip:ViT-L/14
openai-clip:RN50x16
openai-clip:RN50x64
openai-clip:RN101
laion-clip:roberta-ViT-B-32
laion-clip:coca_ViT-B-32
laion-clip:dc-ViT-B-32
laion-clip:dc-ViT-B-16
laion-clip:dc-ViT-L-14
laion-clip:ViT-H-14
laion-clip:ViT-g-14
laion-clip:ViT-bigG-14
NegCLIP
crepe_ft_swap
crepe_ft_replace
crepe_ft_negate
crepe_ft_all
svlc_pos
svlc_rb_neg
svlc_llm_and_rb_negs
svlc_llm_and_rb_negs_pos
dac_llm
dac_sam
xvlm-pretrained-16m
coco_crepe_negs_hard_pos
coco_crepe_negs
coco_crepe_pos
coco_crepe_negs_hard_pos_swap_only
coco_crepe_negs_hard_pos_replace_only
coco_control
coco_crepe_negs_hard_pos_25
coco_crepe_negs_hard_pos_50
coco_crepe_negs_hard_pos_75
```

## Generating hard positives
`generate_hard_positives.py` generates hard positive captions for _finetuning_ using LLAMA2 70B-Chat.


## Note about additional files
I'm leaving in the code from Yuksekgonul et al. that I don't directly use in my work, e.g. `main_retrieval.py`, in the belief that this would still be helpful to have around, in case you want to run the other code on the datasets I've added in.

# Citation
If you use this code or data, please consider citing our paper:
```
@inproceedings{
  kamath2024the,
  title={The Hard Positive Truth about Vision-Language Compositionality},
  author={Kamath, Amita and Hsieh, Cheng-Yu and Chang, Kai-Wei and Krishna, Ranjay},
  booktitle={ECCV},
  year={2024}
}
```
