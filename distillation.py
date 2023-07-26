from transformers import AutoTokenizer, BertForMaskedLM
from transformers import TrainingArguments
from torch.utils.data import random_split

from loss import DistillationTrainer
from data import read_data, tokenize_ditillation_data
from preprocess import ditillation_data_processing

import os
import yaml

def train(args, config):
    model_checkpoint = os.path.join(config["model"]["student_model"], f'checkpoint-{args.restore_step}')
    student_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    student_model = BertForMaskedLM.from_pretrained(model_checkpoint)
    student_model = student_model.to(config["train"]["device"])

    teacher_tokenizer = AutoTokenizer.from_pretrained(config["model"]["teacher_model"])
    teacher_model = BertForMaskedLM.from_pretrained(config["model"]["teacher_model"])
    teacher_model = teacher_model.eval().to(config["train"]["device"])

    print("read raw data ...")
    data = read_data(config["data"]["corpus_path"])

    print("tokenize data ...")
    student_data, teacher_data = tokenize_ditillation_data(data, student_tokenizer, teacher_tokenizer)
    data = ditillation_data_processing(student_data, teacher_data, student_tokenizer, teacher_tokenizer, config)

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

    trainer = DistillationTrainer(
        model=student_model.to(config["train"]["device"]),
        teacher = teacher_model.to(config["train"]["device"]),
        config=config,
        args=train_args,
        train_dataset=train_data,
        eval_dataset=val_data)
    # start training
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, type=str, help="path to config")
    parser.add_argument("--restore_step", required=True, type=int, help="model restore step")
    args = parser.parse_args()

    config = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
    train(args, config["distillation"])

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