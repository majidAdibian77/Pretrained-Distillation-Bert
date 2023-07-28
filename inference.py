from transformers import AutoTokenizer, BertForMaskedLM, AutoConfig
import torch 

import argparse
import os
import yaml

from utils import add_mask
def infer(args, config):
    os.environ["CUDA_VISIBLE_DEVICES"]= f"{config['inference']['device'].split(':')[1]}"

    model_checkpoint = os.path.join(config["train"]["ckpt_path"], f'checkpoint-{args.restore_step}')
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = BertForMaskedLM.from_pretrained(model_checkpoint)
    model = model.eval().to(config["inference"]["device"])

    if args.auto_mask:
        text = add_mask(args.input_text, args.num_mask, tokenizer)
    else:
        text = args.input_text
    print("input text: " + f"{text}")

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k:v.to(config['inference']['device']) for k,v in inputs.items()}
    mask_token_indexes = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    logits = model(**inputs).logits
    mask_token_logits = logits[0, mask_token_indexes, :]
    top_3_tokens = torch.topk(mask_token_logits, 3, dim=1).indices

    input_ids = inputs["input_ids"][0]
    for i in range(3):
        for j, mask_token_index in enumerate(mask_token_indexes):
            input_ids[mask_token_index] = top_3_tokens[j][i]
        print(f"{i+1}: " + tokenizer.decode(input_ids, skip_special_tokens=True))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, type=str, help="path to config")
    parser.add_argument("--restore_step", required=True, type=int, help="model restore step")
    parser.add_argument("--input_text", required=True, type=str, help="input text to test pretrained model")
    parser.add_argument("-auto_mask", action="store_true", required=False, help="replace mask token automatikaly")
    parser.add_argument("--num_mask", type=int, required=False, help="number of mask token mut be added")

    args = parser.parse_args()

    config = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
    infer(args, config["pretrain"])

    """
    python3 inference.py --config_path "config.yaml" --restore_step 1000000 --input_text "بیشتر درامد کشور [MASK] از نفت است."
    python3 inference.py --config_path "config.yaml" --restore_step 1000000 --input_text "بیشتر درامد کشور ایران از نفت است." -auto_mask --num_mask 1
    """