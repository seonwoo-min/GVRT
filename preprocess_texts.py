# Written by Seonwoo Min, LG AI Research (seonwoo.min0@gmail.com)

import os
import sys
import numpy as np
from tqdm import tqdm
from collections import Counter
from sentence_transformers import SentenceTransformer

from src.data import Vocabulary
from src.pycocotools.coco import COCO
from src.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


def main(path):
    texts, tokenized_texts, vocab = build_vocab(path)
    embed_texts(path, texts, tokenized_texts, vocab)


def embed_texts(path, texts, tokenized_texts, vocab):
    model = SentenceTransformer('clip-ViT-B-32')

    texts_s, texts_w, lengths = [], [], []
    for text in tqdm(texts):
        try:
            e = model.encode(text)
        except:
            e = model.encode(".".join(text.split(".")[:-2]))
        texts_s.append(e)

    MaxLength = 64 if ("CUB-DG" in path or "domain_net" in path) else 32
    for text in tokenized_texts:
        if len(text) < MaxLength - 2:
            text += ["<EOS>"] * (MaxLength - 2 - len(text))
        elif len(text) > MaxLength - 2:
            text = text[:MaxLength - 2]
        texts_w.append(np.array([vocab(vocab.start_token)] + [vocab(word) for word in text] + [vocab(vocab.end_token)]))
        lengths.append(len(text) + 1)
        
    np.save(os.path.join(path, "texts", "texts_s.npy"), np.stack(texts_s, 0))
    np.save(os.path.join(path, "texts", "texts_w.npy"), np.stack(texts_w, 0))
    np.save(os.path.join(path, "texts", "lengths.npy"), np.stack(lengths, 0))


def build_vocab(path, threshold=1):
    tokenizer = PTBTokenizer()

    texts, tokenized_texts = [], []
    if "CUB-DG" in path:
        coco = COCO(os.path.join(path, "descriptions.json"))
        texts_words = PTBTokenizer().tokenize(coco.imgToAnns)
        for img, anns in tqdm(sorted(coco.imgToAnns.items())):
            for i in range(len(anns)):
                texts.append(anns[i]["caption"])
                tokenized_texts.append(texts_words[img][i])
    else:
        tokenized_texts = []
        for filename in tqdm(sorted(os.listdir(os.path.join(path, "texts")))):
            if not filename.endswith(".txt"): continue
            with open(os.path.join(path, "texts", filename), "r") as FILE:
                line = FILE.readlines()[0].strip()
                texts.append(line)
                tokenized_texts.append(PTBTokenizer().tokenize({0: [{'caption': line}]})[0][0])

    counter = Counter()
    for text in tokenized_texts:
        counter.update(text)
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary()
    for word in words:
        vocab.add_word(word)
    Vocabulary.save(vocab, os.path.join(path, "texts/vocab.pkl"))

    return texts, tokenized_texts, vocab


if __name__ == "__main__":
    main(sys.argv[1])
