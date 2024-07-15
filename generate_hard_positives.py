import os
import argparse
import math
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from transformers import pipeline


SWAP_TEMPLATE = """Swap the words around the word "and" in a sentence without changing the meaning. Only respond with the changed sentence.

Input: three giraffes and two antelope
Output: two antelopes and three giraffes

Input: a blue and white stained glass clock shows the time
Output: a white and blue stained glass clock shows the time

Input: a mixture of rice and broccoli are put together
Output: a mixture of broccoli and rice are put together

Input: a bathroom with a sink, toilet and shower
Output: a bathroom with a sink, shower and toilet

Input: there is a man wearing glasses and holding a wine bottle
Output: there is a man holding a wine bottle and wearing glasses

Input: {example}
Output: """


REPLACE_TEMPLATE = """Replace one word in the input sentence with a synonym, without changing the meaning of the sentence. Only output the changed sentence.
Input: {example}
Output: """


REPLACE_VERB = """Replace one verb in this sentence with a synonym, without changing the meaning of the sentence. Only output the changed sentence.
Input: A person wearing a shirt and smiling.
Output: A person clothed in a shirt and smiling.

Input: {example}
Output: """

REPLACE_SPATIAL = """Replace a spatial preposition in this sentence with a synonym, without changing the meaning of the sentence. Only output the changed sentence.
Input: A city filled with tall buildings under a cloudy sky.
Output: A city filled with tall buildings beneath a cloudy sky.

Input: {example}
Output: """


def write_jsonl(file_path, examples):
    with open(file_path, "a") as f:
        for example in examples:
            f.write(json.dumps(example))
            f.write("\n")


def batches(examples, bs):
    for i in range(0, len(examples), bs):
        yield examples[i: i + bs]


def main(args):
    # check if output file exists
    if os.path.exists(args.output_file):
        raise ValueError("Output file exists. Set a new output file.")
    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # prepare model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.no_flash_attention:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, 
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, 
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )

    # prepare examples and prompts
    with open(args.input_file) as f:
        examples = json.load(f)
    
    if args.template == "swap":
        template = SWAP_TEMPLATE
    elif args.template == "replace":
        template = REPLACE_TEMPLATE
    elif args.template == "replace_verb":
        template = REPLACE_VERB
    elif args.template == "replace_spatial":
        template = REPLACE_SPATIAL
    else:
        raise ValueError

    if args.num_examples is not None:
        all_prompts = [template.format(example=example["caption"]).strip() for example in examples[:args.num_examples]]
    else:
        all_prompts = [template.format(example=example["caption"]).strip() for example in examples]

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
    )
    pipe.tokenizer.pad_token_id = model.config.eos_token_id
    pipe.tokenizer.padding_side = "left"

    for batch_prompt in tqdm(batches(all_prompts, args.batch_size), total=math.ceil(len(all_prompts) / args.batch_size)):
        outputs = pipe(batch_prompt, return_full_text=False, max_new_tokens=args.max_new_tokens)
        outputs = [output[0]["generated_text"] for output in outputs]
        write_jsonl(args.output_file, outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str)
    parser.add_argument('--output-file', type=str)
    parser.add_argument('--template', type=str)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-new-tokens', type=int, default=20)
    parser.add_argument('--model-name', type=str, default="meta-llama/Llama-2-70b-chat-hf")
    parser.add_argument('--no-flash-attention', action="store_true", default=False) 
    parser.add_argument('--num-examples', type=int, default=None)

    args = parser.parse_args()
    main(args)

