

def train_tokenizer(data, tokenizer, config, model_config):
    batch_size = config["tokenizer"]["batch_size"]
    def batch_iterator():
        for i in range(0, len(data), batch_size):
            yield[" ".join(doc) for doc in data[i : i + batch_size]]

    tokenizer = tokenizer.train_new_from_iterator(
        batch_iterator(), vocab_size=model_config.vocab_size
    )
    return tokenizer

def save_tokenizer(tokenizer, config):
    tokenizer.save_pretrained(config["tokenizer"]["save_paths"])