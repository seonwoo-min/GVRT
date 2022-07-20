# Written by Seonwoo Min, LG AI Research (seonwoo.min0@gmail.com)
# Some parts of the code were referenced from or inspired by below
# - DomainBed (github.com/facebookresearch/DomainBed)

import os
import json
import pickle
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.pycocotools.coco import COCO
from src.pycocoevalcap.eval import COCOEvalCap


class CUB_DG_Dataset(torch.utils.data.Dataset):
    def __init__(self, domain, gvrt_flag):
        self.data_path = "data/CUB-DG/"
        self.coco = COCO(os.path.join(self.data_path, "descriptions.json"))
        self.domain = domain
        self.num_classes = len(self.coco.cats)
        with open(os.path.join(self.data_path, "split.json"), 'r') as FILE:
            self.split = json.load(FILE)

        self.train_flag = False
        self.transform_train, self.transform_eval = get_transforms()

        self.gvrt_flag = gvrt_flag
        if self.gvrt_flag:
            self.vocab = Vocabulary.load(os.path.join(self.data_path, "texts/vocab.pkl"))
            self.texts_s = torch.from_numpy(np.load(os.path.join(self.data_path, "texts/texts_s.npy")))
            self.texts_w = torch.from_numpy(np.load(os.path.join(self.data_path, "texts/texts_w.npy")))
            self.lengths = torch.from_numpy(np.load(os.path.join(self.data_path, "texts/lengths.npy")))

    def __getitem__(self, img_id):
        image_dict = self.coco.loadImgs(img_id)[0]
        file_name, label = image_dict["file_name"], image_dict["label"] - 1
        image = Image.open(os.path.join(self.data_path, self.domain, file_name)).convert('RGB')
        image = self.transform_train(image) if self.train_flag else self.transform_eval(image)
        label = torch.tensor(label, dtype=torch.long)

        if self.gvrt_flag:
            img_id_t = img_id
            rand_idx = np.random.randint(len(self.coco.imgToAnns[img_id_t]))

            text_id = self.coco.imgToAnns[img_id_t][rand_idx]['id'] - 1
            text_s = self.texts_s[text_id]
            text_w = self.texts_w[text_id]
            length = self.lengths[text_id]

            return image, label, text_s, text_w, length, file_name

        else:
            return image, label

    def __len__(self):
        return len(self.coco.imgs)

    def set_train_flag(self, train_flag):
        self.train_flag = train_flag


def get_datasets_and_iterators(env_flag, gvrt_flag, eval_flag=False):
    """ load CUB-DG datasets """
    DOMAINS = ["Photo", "Cartoon", "Art", "Paint"]
    BATCH_SIZE = 32
    NUM_WORKERS = 8

    source_num = 0
    datasets, iterators_train, iterators_eval, names_eval = [], [], [], []
    for d, domain in enumerate(DOMAINS):
        dataset = CUB_DG_Dataset(domain, gvrt_flag)
        datasets.append(dataset)

        if not eval_flag and d != env_flag:
            iterators_train.append(InfiniteDataLoader(
                _SplitDataset(dataset, dataset.split["train%d" % source_num], train_flag=True), BATCH_SIZE, NUM_WORKERS))
            iterators_eval.append(torch.utils.data.DataLoader(
                _SplitDataset(dataset, dataset.split["valid%d" % source_num], train_flag=False),
                batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, pin_memory=True))
            names_eval.append("env%d_1" % d)
            source_num += 1
        if d == env_flag:
            iterators_eval.append(torch.utils.data.DataLoader(
                _SplitDataset(dataset, dataset.split["test"], train_flag=False),
                batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, pin_memory=True))
            names_eval.append("env%d_2" % d)

    if not eval_flag:
        iterators_train = zip(*iterators_train)

    return datasets, iterators_train, iterators_eval, names_eval


def get_transforms():
    """ get transforms for CUB datasets """
    resize, cropsize = 512, 448

    transform_train = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomCrop(cropsize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_eval = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(cropsize),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform_train, transform_eval


class _SplitDataset(torch.utils.data.Dataset):
    """ used by split_dataset """
    def __init__(self, underlying_dataset, keys, train_flag):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
        self.train_flag = train_flag

    def __getitem__(self, key):
        self.underlying_dataset.set_train_flag(self.train_flag)
        return self.underlying_dataset[self.keys[key]]

    def __len__(self):
        return len(self.keys)


class _InfiniteSampler(torch.utils.data.Sampler):
    """ wraps another Sampler to yield an infinite stream """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, num_workers):
        super().__init__()

        batch_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(dataset, replacement=True),
            batch_size=batch_size,
            drop_last=True)

        self._infinite_iterator = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            pin_memory=True,
            batch_sampler=_InfiniteSampler(batch_sampler)
        )

        self.iterator = iter(self._infinite_iterator)

    def __iter__(self):
        while True:
            yield next(self.iterator)

    def __len__(self):
        raise ValueError


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self, unknown_token='<UNK>', start_token='<SOS>', end_token='<EOS>'):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.unknown_token = unknown_token
        self.start_token = start_token
        self.end_token = end_token

        self.add_word(start_token)
        self.add_word(end_token)
        self.add_word(unknown_token)

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def get_word_from_idx(self, idx):
        if not idx in self.idx2word:
            return self.unknown_token
        return self.idx2word[idx]

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unknown_token]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            vocab = pickle.load(f)
        assert isinstance(vocab, cls)
        return vocab

    @classmethod
    def save(cls, vocab, path):
        assert isinstance(vocab, cls)
        with open(path, 'wb') as f:
            pickle.dump(vocab, f)


def evaluate_text(labels_path, outputs_path):
    """ evaluate texts using COCOeval package """
    coco = COCO(labels_path)
    cocoRes = coco.loadRes(outputs_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.evaluate()

    return cocoEval.eval
    

class ImageFolderwithTexts(ImageFolder):
    def __init__(self, root, transform, texts_s, texts_w, lengths):
        super(ImageFolderwithTexts, self).__init__(root, transform)
        self.texts_s = texts_s
        self.texts_w = texts_w
        self.lengths = lengths

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        target = torch.tensor(target, dtype=torch.long)

        text_s = self.texts_s[target]
        text_w = self.texts_w[target]
        length = self.lengths[target]

        return sample, target, text_s, text_w, length
