import pdb
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json as js
import nltk
import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer,SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import *
from operator import itemgetter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import csv
import xlsxwriter
import random
import re

import tensorflow as tf
import tensorflow_hub as hub
import bert
from tensorflow.keras.models import Model



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

"""
generate masks based on the original BERT
"""

def get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))


"""
generate segments based on the original BERT
"""

def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


"""
generate embeddings based on the original BERT
"""
def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids


def generate_bert(corona_pos_all_text):
    FullTokenizer = bert.bert_tokenization.FullTokenizer

    """
    input token ids (tokenizer converts tokens using vocab file)
    input masks (1 for useful tokens, 0 for padding)
    segment ids (for 2 text training: 0 for the first one, 1 for the second one)
    """

    max_seq_length = 25000  # Your choice here.
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                           name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                        name="segment_ids")

    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                                trainable=True)

    """
    pooled_output of shape [batch_size,768] with representations for the entire input sequences
    sequence_output of shape [batch_size,max_seq_length,768] with representations for each input token (in context)
    """

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])


    """
    convert to numpy array
    convert to all lowercase
    tokenizer now has above two properties
    """
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = FullTokenizer(vocab_file, do_lower_case)

    embeddings_info=[]
    j=0
    for text in corona_pos_all_text:
        if j % 1000 == 0:
            print("Tokenizing (number %d)..." % j)

        stokens = tokenizer.tokenize(text)
        if len(stokens)>=25000:
            continue
        j+=1
        stokens = ["[CLS]"] + stokens + ["[SEP]"]

        input_ids = get_ids(stokens, tokenizer, max_seq_length)
        input_masks = get_masks(stokens, max_seq_length)
        input_segments = get_segments(stokens, max_seq_length)

        embeddings_info.append(input_ids)

    return embeddings_info


