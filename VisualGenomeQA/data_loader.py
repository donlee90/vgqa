import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from vocab import Vocabulary
import json


class VGQADataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, img_dir, qas, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            img_dir: image directory.
            qas: VisualGenome QA annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.img_dir = img_dir
        self.annos = self.load_from_json(qas)
        self.ids = list(self.annos.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        vocab = self.vocab
        ann_id = self.ids[index]
        question = self.annos[ann_id]['question']
        answer = self.annos[ann_id]['answer']
        img_id = self.annos[ann_id]['image_id']
        path = "%d.jpg" % (img_id)

        image = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert question (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(question).lower())
        question = []
        question.append(vocab('<start>'))
        question.extend([vocab(token) for token in tokens])
        question.append(vocab('<end>'))
        question = torch.Tensor(question)

        # Convert answer (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(answer).lower())
        answer = []
        answer.append(vocab('<start>'))
        answer.extend([vocab(token) for token in tokens])
        answer.append(vocab('<end>'))
        answer = torch.Tensor(answer)

        return image, question, answer

    def __len__(self):
        return len(self.ids)

    def load_from_json(self, json_file):
        """ Load annotations from json file"""
        with open(json_file) as f:
            vg_qas = json.load(f)

        annos = {}
        for entry in vg_qas:
            for qa in entry["qas"]:
                anno = {}
                anno["question"] = str(qa["question"])
                anno["answer"] = str(qa["answer"])
                anno["image_id"] = entry["id"]
                annos[qa["qa_id"]] = anno

        return annos


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, questions, answers = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge questions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in questions]
    questions_tensor = torch.zeros(len(questions), max(lengths)).long()
    for i, cap in enumerate(questions):
        end = lengths[i]
        questions_tensor[i, :end] = cap[:end]        

    # Merge answers (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in answers]
    answers_tensor = torch.zeros(len(answers), max(lengths)).long()
    for i, cap in enumerate(answers):
        end = lengths[i]
        answers_tensor[i, :end] = cap[:end] 

    return images, questions_tensor, answers_tensor


def get_loader(img_dir, qas, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    vgqa = VGQADataset(img_dir=img_dir,
                       qas=qas,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=vgqa, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
