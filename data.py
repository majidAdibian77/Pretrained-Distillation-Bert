from hazm import sent_tokenize
import re
import string
from tqdm import tqdm

def read_data(config):
    with open(config["data"]["corpus_path"]) as f:
        data = f.readlines()
    return data

def save_processed_data(data, config):
    with open(config["data"]["processed_path"], "w") as f:
        for sents in tqdm(data):
            f.write("\n".join(sents))
            f.write("\n" + "="*50 + "\n")

def cleaner(text, types):
    if "english" in types:
        text = re.sub(r"[a-zA-Z]", "", text)
    if "number" in types:
        text = re.sub(r"[0-9]", "", text)   
    if "punctuation" in types:
        punctuations = string.punctuation + "؛،؟!"
        text = text.translate(str.maketrans('', '', punctuations))   
    return text

def preprocess_data(data, config):
    processed_data = []
    for doc in tqdm(data): 
        doc2 = cleaner(doc, ["english", "number", "punctuation"])
        if len(doc2.strip().split()) == 0: ## ignore lines that not contain Persian text
            continue
        if (re.search(r'^[0-9]+\s*(\.|\)|:)', doc) is not None) or (doc[0] == "-"): ## append list rows to previouse doc
            processed_data[-1].append(doc.strip()) 
        else:
            raw_sentences = sent_tokenize(doc)
            sentences = []
            for sent in raw_sentences:
                sent = cleaner(sent, ["english", "number", "punctuation"])
                if len(sent.strip().split()) > 0: ## ignore lines that not contain Persian text
                    sentences.append(sent)
            if len(sentences) != 0:
                if len(processed_data)>0 and len(processed_data[-1])==1: ## append one line docs to next doc
                    processed_data[-1].extend(sentences)
                else:
                    processed_data.append(sentences)
    return processed_data

def tokenize_data(data, tokenizer):
    data_tokens = []
    for doc in tqdm(data):
        tokens = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)) for sent in doc]
        data_tokens.append(tokens)
    return data_tokens

def load_processed_data(tokenizer, config):
    processed_data = []
    with open(config["data"]["processed_path"]) as f:
        line = f.readline()
        if line[0] != "=":
            processed_data.append(line)
    processed_data = [
        [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)) for sent in doc]
        for doc in processed_data
    ]
    return processed_data