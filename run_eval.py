
import pdb
import clip
import random
import argparse
import pandas as pd

from torch.utils.data import DataLoader
from model_zoo import get_model
from dataset_zoo import VG_Attribution, VL_Checklist_Relation, VL_Checklist_Attribute, VL_Checklist_Object, CREPE_COCO_Negate



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", default="swap", type=str, \
            choices=["swap", "replace_rel", "replace_att", "replace_obj", "negate"])
    parser.add_argument("model_name", default="openai-clip:ViT-B/32", type=str, \
            choices=["openai-clip:ViT-B/32", "openai-clip:ViT-L/14", \
            "openai-clip:ViT-B/16", "openai-clip:RN50x16", \
            "openai-clip:RN50x64", "openai-clip:RN101", \
            "xvlm-pretrained-16m", \
            "laion-clip:ViT-H-14", \
            "laion-clip:ViT-g-14", "laion-clip:ViT-bigG-14", \
            "laion-clip:roberta-ViT-B-32", "laion-clip:coca_ViT-B-32", \
            "laion-clip:dc-ViT-B-32", "laion-clip:dc-ViT-B-16", \
            "laion-clip:dc-ViT-L-14", \
            "NegCLIP", "crepe_ft_swap", \
            "crepe_ft_replace", "crepe_ft_negate", "crepe_ft_all", \
            "svlc_pos", "svlc_rb_neg", "svlc_llm_and_rb_negs", \
            "svlc_llm_and_rb_negs_pos", "dac_llm", \
            "dac_sam", "coco_crepe_negs_hard_pos", "coco_crepe_negs", \
            "coco_crepe_pos", "coco_crepe_negs_hard_pos_swap_only", \
            "coco_crepe_negs_hard_pos_replace_only", "coco_control", \
            "coco_crepe_negs_hard_pos_25", "coco_crepe_negs_hard_pos_50", \
            "coco_crepe_negs_hard_pos_75", \
            "coco_crepe_negs_hard_pos_v2", "coco_crepe_negs_hard_pos_v2_replace_only"])
    args = parser.parse_args()

    dataset_name = args.dataset_name
    model_name = args.model_name
    
    root_dir = "data"
    swapped_dir = "swapped_data"

    model, preprocess = get_model(model_name=model_name, device="cuda", root_dir=root_dir)

    print(model_name)
    print(dataset_name)
    
    if dataset_name == 'swap':
        dataset =  VG_Attribution(image_preprocess=preprocess, download=True, root_dir=root_dir)
        swapped_dataset = VG_Attribution(image_preprocess=preprocess, download=True, root_dir=swapped_dir)

    elif dataset_name == 'replace_rel':
        dataset = VL_Checklist_Relation(image_preprocess=preprocess, download=True, root_dir=root_dir)
        swapped_dataset = VL_Checklist_Relation(image_preprocess=preprocess, download=True, root_dir=swapped_dir)

    elif dataset_name == 'replace_att':
        dataset = VL_Checklist_Attribute(image_preprocess=preprocess, download=True, root_dir=root_dir)
        swapped_dataset = VL_Checklist_Attribute(image_preprocess=preprocess, download=True, root_dir=swapped_dir)
    
    elif dataset_name == 'replace_obj':
        dataset = VL_Checklist_Object(image_preprocess=preprocess, download=True, root_dir=root_dir)
        swapped_dataset = VL_Checklist_Object(image_preprocess=preprocess, download=True, root_dir=swapped_dir)

    elif dataset_name == 'negate':
        dataset = CREPE_COCO_Negate(image_preprocess=preprocess, download=True, root_dir=root_dir)
        swapped_dataset = CREPE_COCO_Negate(image_preprocess=preprocess, download=True, root_dir=swapped_dir)

    # Get original scores
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    orig_scores = model.get_retrieval_scores_batched(data_loader)
    
    #pdb.set_trace()
    if 'xvlm' in model_name:
        orig_scores = orig_scores[0]
    
    orig_acc = sum([1 for v in orig_scores if v[0][0]<v[0][1]])/len(orig_scores)
    # Get scores of the image and the correct, original caption
    # p(c)
    orig_scores_list = [v[0][1] for v in orig_scores]

    # Get modified scores
    data_loader = DataLoader(swapped_dataset, batch_size=16, shuffle=False)
    mod_scores = model.get_retrieval_scores_batched(data_loader)
    if 'xvlm' in model_name:
        mod_scores = mod_scores[0]

    mod_acc = sum([1 for v in mod_scores if v[0][0]<v[0][1]])/len(mod_scores)
    # Get scores of the image and the correct, swapped caption
    # p(c'')
    mod_scores_list = [v[0][1] for v in mod_scores]

    diffs = [abs(orig_scores_list[i] - mod_scores_list[i]) for i in range(len(mod_scores))]
    total_diff = sum(diffs)
    avg_diff = total_diff/len(diffs)

    # (c, c'') > c'
    print(len([1 for i in range(len(orig_scores)) if mod_scores[i][0][0] < mod_scores[i][0][1] and mod_scores[i][0][0] < orig_scores[i][0][1]]))

    # c > c'' > c'
    #print("new")
    #print(len([1 for i in range(len(orig_scores)) if  orig_scores[i][0][1] > mod_scores[i][0][1] and mod_scores[i][0][1] > mod_scores[i][0][0]]))

    # c > c' > c''
    print(len([1 for i in range(len(orig_scores)) if  orig_scores[i][0][1] > mod_scores[i][0][0] and mod_scores[i][0][0] > mod_scores[i][0][1]]))

    # c'' > c' > c
    print(len([1 for i in range(len(orig_scores)) if  mod_scores[i][0][1] > mod_scores[i][0][0] and mod_scores[i][0][0] > orig_scores[i][0][1]]))

    # c' > (c, c'')
    print(len([1 for i in range(len(orig_scores)) if mod_scores[i][0][0] > mod_scores[i][0][1] and mod_scores[i][0][0] > orig_scores[i][0][1]]))

    # c' == c or c' == c''
    print(len([1 for i in range(len(orig_scores)) if mod_scores[i][0][0] == mod_scores[i][0][1] or mod_scores[i][0][0] == orig_scores[i][0][1]]))

    # total_diff of abs(c-c'')
    #print(total_diff)

    # avg_diff
    #print(avg_diff)
    print()



if __name__ == '__main__':
    main()

