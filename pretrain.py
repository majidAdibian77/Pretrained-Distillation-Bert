
from transformers import AutoTokenizer, BertForMaskedLM, AutoConfig
from transformers import TrainingArguments, Trainer
from torch.utils.data import random_split
import torch
import argparse
import yaml
import os

from data import read_data, process_pretraining_data, tokenize_pretraining_data, load_processed_pretraining_data, save_processed_pretraining_data
from preprocess import NSP_preprocessing, MLM_preprocessing
from tokenizer import train_tokenizer, save_tokenizer



def main(args, config):
    os.environ["CUDA_VISIBLE_DEVICES"]= f"{config['train']['device'].split(':')[1]}"

    model_name = f"google/bert_uncased_L-{config['model']['num_layers']}_H-{config['model']['hidden_size']}_A-{config['model']['hidden_size']//64}"

    if args.load_local_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"]["save_paths"])
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_config = AutoConfig.from_pretrained(model_name)
    model = BertForMaskedLM(model_config).to(config["train"]["device"])

    if args.load_tokenized_data:
        print("load load_tokenized data ...")
        data = torch.load(config["data"]["tokenized_path"])
    else:
        print("read raw data ...")
        data = read_data(config["data"]["corpus_path"])
        print("preprocess data ...")
        data = process_pretraining_data(data, config)
        if args.train_tokenizer:
            print("train tokenizer ...")
            tokenizer = train_tokenizer(data, tokenizer, config, model_config)
            print("save tokenizer ...")
            save_tokenizer(tokenizer, config)
    
        print("tokenize data ...")
        data = tokenize_pretraining_data(data, tokenizer)

        print("preprocess data for next sentence prediction ...")
        data = NSP_preprocessing(data, tokenizer, config)

        if args.save_tokenized_data:
            print("save dataset object ...")
            torch.save(data, config["data"]["tokenized_path"])

    mlm_collator = MLM_preprocessing(tokenizer, config)

    print("split data to train and validation ...")
    val_size = config["train"]["val_size"]
    train_size = len(data)-val_size
    val_data, train_data = random_split(data, [val_size, train_size])
    print(f"number of train input data: {len(train_data)}")
    print(f"number of val input data: {len(val_data)}")

    train_args = TrainingArguments(
        config["train"]["ckpt_path"],
        optim="adamw_torch",
        learning_rate=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
        per_device_train_batch_size=config["train"]["batch_size"],
        per_device_eval_batch_size=config["train"]["batch_size"],
        evaluation_strategy="steps",
        save_steps=config["train"]["save_step"],
        eval_steps=config["train"]["val_steps"],
        num_train_epochs=config["train"]["num_train_epochs"],
        logging_dir=config["train"]["log_path"], 
        logging_steps=config["train"]["log_step"]) 

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=mlm_collator,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer)
    # start training
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, type=str, help="path to config")
    parser.add_argument("-save_tokenized_data", action="store_true", help="save processed data or not")
    parser.add_argument("-load_tokenized_data", action="store_true", help="load from processed data or load raw data and preprocess")
    parser.add_argument("-load_local_tokenizer", action="store_true", help="load local saved tokenizer or load tokenizer from hub")
    parser.add_argument("-train_tokenizer", action="store_true", help="retrain tokenizer on local data or not")
    args = parser.parse_args()

    config = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
    main(args, config["pretrain"])

    """
    first run command:
        python3 pretrain.py --config_path "config.yaml"  -save_tokenized_data  -train_tokenizer 

    next runs command:
        python3 pretrain.py --config_path "config.yaml"  -load_tokenized_data  -load_local_tokenizer
    """