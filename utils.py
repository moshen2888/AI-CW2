import nltk


from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import re

import numpy as np

from vocabulary import Vocabulary
from config import *

from copy import copy

def read_lines(filepath):
    """ Open the ground truth captions into memory, line by line. 
    Args:
        filepath (str): the complete path to the tokens txt file
    """
    file = open(filepath, 'r')
    lines = []

    while True: 
        # Get next line from file 
        line = file.readline() 
        if not line: 
            break
        lines.append(line.strip())
    file.close()
    return lines


def parse_lines(lines):
    """
    Parses token file captions into image_ids and captions.
    Args:
        lines (str list): str lines from token file
    Return:
        image_ids (int list): list of image ids, with duplicates
        cleaned_captions (list of lists of str): lists of words
    """
    image_ids = []
    cleaned_captions = []


    # QUESTION 1.1
    for line in lines:
        tokens = line.split("\t", 1)
        image_id = line.split('.')[0]
        image_caption = tokens[1:]
        caption = list(map(lambda x: x.lower(), image_caption))
        image_ids.append(image_id)
        caption = ''.join(caption)
        remove_chars = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
        caption = re.sub(remove_chars, '', caption)
        cleaned_captions.append(caption)
        

    return image_ids, cleaned_captions


def build_vocab(cleaned_captions):
    """ 
    Parses training set token file captions and builds a Vocabulary object
    Args:
        cleaned_captions (str list): cleaned list of human captions to build vocab with

    Returns:
        vocab (Vocabulary): Vocabulary object
    """

    # QUESTION 1.1
    # TODO collect words
    words = dict()

    # Save words whose frequency meets the threshold
    for sentence in cleaned_captions:
        for word in sentence.split():
            words[word] = words.get(word, 0) + 1
    
    words_vocab = dict(filter(lambda item: item[1]>3, words.items()))
    # print(words_vocab)
    print(len(words_vocab))
    
    
    # create a vocab instance
    vocab = Vocabulary()

    # add the token words
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # TODO add the rest of the words from the cleaned captions here
    # vocab.add_word('word')
    for word in words_vocab:
        vocab.add_word(word)
    return vocab



def decode_caption(sampled_ids, vocab):
    """ 
    Args:
        sampled_ids (int list): list of word IDs from decoder
        vocab (Vocabulary): vocab for conversion
    Return:
        predicted_caption (str): predicted string sentence
    """
    predicted_caption = ""


    # QUESTION 2.1
    word_caption = []
    for word_id in sampled_ids:
        words = []
        for ids in word_id:
            ids = list(ids.cpu().numpy())
            for idx in ids:
                word = vocab.idx2word[idx]
                if word == '<end>':
                    break
                if word not in ("<start>","<unk>","<pad>"):
                    words.append(word)
        word_caption.append(" ".join(words))
    predicted_caption = word_caption[::1]
    # print(predicted_caption)
    return predicted_caption


"""
We need to overwrite the default PyTorch collate_fn() because our 
ground truth captions are sequential data of varying lengths. The default
collate_fn() does not support merging the captions with padding.

You can read more about it here:
https://pytorch.org/docs/stable/data.html#dataloader-collate-fn. 
"""
def caption_collate_fn(data):
    """ Creates mini-batch tensors from the list of tuples (image, caption).
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 224, 224).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 224, 224).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length from longest to shortest.
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # merge images (from tuple of 3D tensor to 4D tensor).
    # if using features, 2D tensor to 3D tensor. (batch_size, 256)
    images = torch.stack(images, 0) 

    # merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths


def getreference(ids):
    reference = [] 
    test_lines = read_lines(TOKEN_FILE_TEST)
    for line in test_lines:
        if ids in line:
            strs = ''.join([i for i in line.strip().strip(' .').split('#')[1] if not i.isdigit()])
            reference.append(strs.strip().lower())
    ref_words = []
    for sentence in reference:
        ref_words.append(sentence.split(' '))  
    return ref_words


def BleuScore(cleaned_captions, predicted_words, weights):


  # cleaned_captions: reference caption
  # predicted_caption: model predicted caption
  # return BLEU score

    BLEUscore = nltk.translate.bleu_score.sentence_bleu(cleaned_captions, predicted_words, smoothing_function=SmoothingFunction().method7, weights=weights)
        
    return BLEUscore


def Cosine_sim(predicted_caption, cleaned_captions, vocab, idx):
    embedding =  nn.Embedding(len(vocab), EMBED_SIZE)
    # embedding vector for predicted caption
    avg_vectors_pre = []
    vector_pre = []
    predicted_caption_batch = predicted_caption[idx]
    words = [x for x in predicted_caption_batch.split(' ')
            if ('<end>' not in x and '<start>' not in x and '<unk>' not in x)]
    for word in words:
        vector = vocab.__call__(word)
        tensors = torch.tensor([vector])
        word_embedding = embedding(tensors)
        vector_pre.append(word_embedding.detach().numpy())
    avg_vector = sum(vector_pre)/len(vector_pre)
    
    avg_vectors_pre.append(avg_vector)
    

    # embedding vector for reference caption
    five_ref_avg_vector = []
    for sentence in cleaned_captions:
        ref_vector = []
        # ref_words = sentence.split()
        for word in sentence:
            vector_ref = vocab.__call__(word)
            tensors_ref = torch.tensor([vector_ref])
            word_embedding_ref = embedding(tensors_ref)
            ref_vector.append(word_embedding_ref.detach().numpy())
            
        
        avg_ref_vector = sum(ref_vector) / len(ref_vector)
        five_ref_avg_vector.append(avg_ref_vector)
        
    # cosine similarity score
    cosine_scores = []
    for i in range(len(five_ref_avg_vector)):
        ref_avg_vector = five_ref_avg_vector[i]
        cosine_score = cosine_similarity(avg_vector, ref_avg_vector)
        cosine_scores.append(cosine_score)
    # print(cosine_scores)
    cosine_final_score = np.mean(cosine_scores)
    return cosine_final_score