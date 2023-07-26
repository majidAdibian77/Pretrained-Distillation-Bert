
from transformers import AutoTokenizer, BertForMaskedLM, AutoConfig
from transformers import TrainingArguments, Trainer
from torch.utils.data import random_split
import argparse
import yaml

from data import read_data, process_pretraining_data, tokenize_pretraining_data, load_processed_pretraining_data, save_processed_pretraining_data
from preprocess import NSP_preprocessing, MLM_preprocessing
from tokenizer import train_tokenizer, save_tokenizer


def main(args, config):
    
    model_name = f"google/bert_uncased_L-{config['model']['num_layers']}_H-{config['model']['hidden_size']}_A-{config['model']['hidden_size']//64}"

    if args.load_local_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"]["save_paths"])
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_config = AutoConfig.from_pretrained(model_name)
    model = BertForMaskedLM(model_config)

    if args.load_processed_data:
        print("load processed data ...")
        data = load_processed_pretraining_data(tokenizer, config)
    else:
        print("read raw data ...")
        data = read_data(config["data"]["corpus_path"])
        print("preprocess data ...")
        data = process_pretraining_data(data, config)
        if args.save_processed_data:
            print("save processed data ...")
            save_processed_pretraining_data(data, config)

    if args.train_tokenizer:
        print("train tokenizer ...")
        tokenizer = train_tokenizer(data, tokenizer, config, model_config)
        print("save tokenizer ...")
        save_tokenizer(tokenizer, config)
    
    print("tokenize data ...")
    data = tokenize_pretraining_data(data, tokenizer)

    print("preprocess data for next sentence prediction ...")
    data = NSP_preprocessing(data, tokenizer, config)

    print("preprocess data for masked language modeling ...")
    mlm_collator = MLM_preprocessing(data, tokenizer, config)
    
    print("split data to train and validation ...")
    val_size = config["train"]["val_size"]
    train_size = len(data)-val_size
    val_data, train_data = random_split(data, [val_size, train_size])
    print(f"number of train input data: {len(train_data)}")
    print(f"number of val input data: {len(val_data)}")

    train_args = TrainingArguments(
        config["train"]["ckpt_path"],
        learning_rate=config["train"]["learning_rate"],
        optim="adamw_torch",
        per_device_train_batch_size=config["train"]["batch_size"],
        per_device_eval_batch_size=config["train"]["batch_size"],
        save_steps=config["train"]["save_step"],
        eval_steps=config["train"]["val_steps"],
        num_train_epochs=config["train"]["batch_size"],
        weight_decay=config["train"]["weight_decay"],
        logging_dir=config["train"]["log_path"], 
        logging_steps=config["train"]["log_step"]) 

    trainer = Trainer(
        model=model.to(config["train"]["device"]),
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
    parser.add_argument("-save_processed_data", action="store_true", help="save processed data or not")
    parser.add_argument("-load_processed_data", action="store_true", help="load from processed data or load raw data and preprocess")
    parser.add_argument("-load_local_tokenizer", action="store_true", help="load local saved tokenizer or load tokenizer from hub")
    parser.add_argument("-train_tokenizer", action="store_true", help="retrain tokenizer on local data or not")
    args = parser.parse_args()

    config = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
    main(args, config["pretrain"])

    """
    first run command:
        python3 pretrain.py --config_path "config.yaml"  -save_processed_data  -train_tokenizer 
           or (if it's not needed to train tokenizer)
        python3 pretrain.py --config_path "config.yaml"  -save_processed_data

    next runs command:
        python3 pretrain.py --config_path "config.yaml"  -load_processed_data  -load_local_tokenizer
           or (if there is no trained local tokenizer)
        python3 pretrain.py --config_path "config.yaml"  -load_processed_data
    """