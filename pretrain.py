
from transformers import AutoTokenizer, BertForMaskedLM, AutoConfig
from transformers import TrainingArguments, Trainer
import argparse
import os
import yaml

from data import read_data, tokenize_data

def main(args, config):
    
    model_name = f"google/bert_uncased_L-{config['model']['num_layers']}_H-{config['model']['hidden_size']}_A-{config['model']['hidden_size']//64}"
    tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased")
    config = AutoConfig(model_name)
    model = BertForMaskedLM.from_config(config)


    train_data = read_data(os.path.join('data/pquad','train.json'))
    val_data = read_data(os.path.join('data/pquad','val.json'))

    tokenized_train_data = tokenize_data(tokenizer, train_data, 512, 256)
    print("train data is tokenized")
    tokenized_val_data = tokenize_data(tokenizer, val_data, 512, 256)
    print("validation data is tokenized")

    train_args = TrainingArguments(
    config["path"]["ckpt"],
    evaluation_strategy=config["train"]["eval_strategy"],
    learning_rate=config["train"]["learning_rate"],
    per_device_train_batch_size=config["train"]["batch_size"],
    per_device_eval_batch_size=config["train"]["batch_size"],
    save_steps=config["train"]["save_step"],
    eval_steps=config["train"]["val_steps"],
    num_train_epochs=config["train"]["batch_size"],
    weight_decay=config["train"]["weight_decay"]) 

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_val_data,
        tokenizer=tokenizer)
    # start training
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, type=str, help="path to config")
    args = parser.parse_args()

    config = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
    main(args, config)