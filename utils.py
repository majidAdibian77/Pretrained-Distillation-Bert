import random

def add_mask(text, num_mask, tokenizer):
    words = text.split()
    maske_indexes = random.sample(range(len(words)), k=num_mask)
    for maske_index in maske_indexes:
        words[maske_index] = tokenizer.mask_token
    return " ".join(words)