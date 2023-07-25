from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import datasets
import random
from tqdm import tqdm


def MLM_preprocessing(data, tokenizer, config):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=config["data"]["mask_prob"]
    )
    return data_collator

def NSP_preprocessing(documents, tokenizer, config):
    max_seq_len = config["data"]["max_input_tokens"] - tokenizer.num_special_tokens_to_add(pair=True)
    input_sentences = []

    for doc_id, sentences in tqdm(enumerate(documents)):
        target_seq_length = max_seq_len
        if random.random() < config["data"]["short_seq_prob"]:
            target_seq_length = random.randint(2, max_seq_len)
        chunk_sents = []  # a buffer stored current working segments
        chunk_length = 0
        i = -1
        while i+1 < len(sentences):
            i+=1
            chunk_sents.append(sentences[i])
            chunk_length += len(sentences[i])
            if chunk_length > target_seq_length or i==len(sentences)-1:
                if chunk_length > max_seq_len:
                    chunk_sents = chunk_sents[:-1]
                if len(chunk_sents) == 0:
                    # chunk_sents, chunk_length = [], 0
                    continue
                sent_a, sent_b = [], []
                ### create first sentence: sent_a
                split_sent_a_index = 1
                if len(chunk_sents) > 1:
                    split_sent_a_index = random.randint(1, len(chunk_sents)-1)
                for sent in chunk_sents[:split_sent_a_index]:
                    sent_a.extend(sent)

                ### create second sentence: sent_b
                if random.random()<config["data"]["next_sentence_prob"] and len(chunk_sents)>1: ## create second sentence from remain 
                    is_next_sent = True                                                         ## sentences of currenct chunk
                    for sent in chunk_sents[split_sent_a_index:]:
                        sent_b.extend(sent)

                else: ## use next sentence from other docs
                    is_next_sent = False
                    target_sent_b_length = max_seq_len - len(sent_a)
                    i -= (len(chunk_sents) - split_sent_a_index) ## use next sentences in next data to avoid wasting them
                    ## randomly select a "different" doc
                    while True:
                        random_doc_id = random.randint(0, len(documents)-1)
                        if random_doc_id != doc_id:
                            break
                    ## select random chunk of senetences from random selected doc as next sentenc
                    random_doc = documents[random_doc_id]
                    random_start = random.randint(0, len(random_doc)-1)
                    for j in range(random_start, len(random_doc)):
                        if len(sent_b) + len(random_doc[j]) <= target_sent_b_length:
                            sent_b.extend(random_doc[j])
                        else:
                            break
                input_sentences.append((sent_a, sent_b, is_next_sent))
                chunk_sents = []
                chunk_length = 0

    from matplotlib import pyplot as plt
    a = [len(sent1) for sent1,_,_ in input_sentences]
    b = [len(sent2) for _,sent2,_ in input_sentences]
    next_num = sum([1 for row in input_sentences if len(row[0])==1])
    not_next_num = sum([1 for row in input_sentences if len(row[1])==1])

    data = {"input_ids":[], "token_type_ids":[], "attention_mask":[], "labels":[]}
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    for sent_a, sent_b, is_next_sent, in input_sentences:
        input_ids = tokenizer.build_inputs_with_special_tokens(sent_a, sent_b)
        # add token type ids, 0 for sentence a, 1 for sentence b
        token_type_ids = tokenizer.create_token_type_ids_from_sequences(sent_a, sent_b)
        padded = tokenizer.pad(
            {"input_ids": input_ids, "token_type_ids": token_type_ids},
            padding="max_length",
            max_length=config["model"]["max_seq_len"],
        )
        data["input_ids"].append(padded["input_ids"])
        data["token_type_ids"].append(padded["token_type_ids"])
        data["attention_mask"].append(padded["attention_mask"])
        data["labels"].append(0 if is_next_sent else 1)
    data = datasets.Dataset.from_dict(data)
    return data.shuffle()
                
        