#!/usr/bin/env python
# coding: utf-8

# Imports:
import pickle
import pandas as pd
import contractions
import string
punctuation = string.punctuation
# tokenizer:
from nltk.tokenize import TreebankWordTokenizer as TWT
# lemmatizer:
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk import pos_tag
from nltk.corpus import wordnet
import sqlite3

# Opening pickled dataframe:
with open('ce_data.pkl', 'rb') as f:
    data = pickle.load(f)
list_data = data[['sentences', 'cause', 'effect']].values.tolist()


clean_tokens = list()
bi_or_trigram = list()
i = 1

for line in list_data:
    # Getting sentence, cause and effect:
    sent = line[0] 
    cause = TWT().tokenize(line[1].lower())
    effect = TWT().tokenize(line[2].lower())
    # Expanding contractions:
    sentence = contractions.fix(sent) 
    # Removing punctuation and making words lowercase:
    clean_sentence = sentence.translate(str.maketrans('', '', punctuation)).lower() 
    
    tokens = TWT().tokenize(clean_sentence) # tokenizing
    labels = list()
    
    for word in tokens:
        if word in cause:
            # Adding label 'C' for one word causes:
            if len(cause) == 1: 
                if word == cause[0] and 'C' not in labels:
                    labels.append('C')
            # Adding label 'B-C'/'I-C' for multi word causes:
            elif len(cause) > 1:
                if word == cause[0]:
                    labels.append('B-C')
                elif labels.count('I-C') != len(cause) - 1:
                    labels.append('I-C')
                    
        elif word in effect:
            # Adding label 'E' for one word effects:   
            if len(effect) == 1 and 'E' not in labels:
                if word == effect[0]:
                    labels.append('E')
            # Adding label 'B-E'/'I-E' for multi word effects:
            elif len(effect) > 1:
                if word == effect[0]:
                    labels.append('B-E')
                elif labels.count('I-E') != len(effect) - 1:
                    labels.append('I-E')
        
        # Adding label 'O' for any other token: 
        else:
            labels.append('O')
    
    clean_tokens.append([i, tokens, labels])

    i += 1

bi_or_tri_grams = list()
for x in clean_tokens:
    if 'B-E' in x[2] or 'B-C' in x[2]:
        bi_or_tri_grams.append(x)
len(bi_or_tri_grams)


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


# Lemmitization:
reg_data_tokens, pos_data_tokens = [], []
for x in clean_tokens:
    reg_temp, pos_temp = [], []
    for word in x[1]:
        if word.isalpha():
            pos_tok = lemmatizer.lemmatize(word.lower(), get_wordnet_pos(word)) # lemmatization
            pos_temp.append(pos_tok)
            reg_temp.append(word) # simple
        elif word.isnumeric() or word.isalnum():
            reg_temp.append('[N]')
            pos_temp.append('[N]')
    reg_data_tokens.append([x[0], reg_temp, x[2]])
    pos_data_tokens.append([x[0], pos_temp, x[2]])


# Functions for writing to dataframe or sqlite file:

def to_df(tokens):
    ids, token_n, token, label = [], [], [], []
    for x in tokens:
        if len(x[1]) == len(x[2]):
            ids.extend([x[0]]*len(x[1]))
            token_n.extend([i+1 for i in range(len(x[1]))])
            token.extend(x[1])
            label.extend(x[2])
    ent_df = pd.DataFrame({'id': ids, 'token_n': token_n, 'token': token, 'label': label})
    
    return ent_df


def to_sqlite(tokens, file_name, ent_name):
    ent = to_df(tokens)
    conn = sqlite3.connect(file_name)
    ent.to_sql(ent_name, conn, if_exists = 'replace', index = False)
    conn.close()


# Calling function:
to_sqlite(reg_data_tokens, 'semeval_normal.sqlite', 'ent_simple')
to_sqlite(pos_data_tokens, 'semeval_normal.sqlite', 'ent_lem_pos')


reg_data_tokens_df = pd.DataFrame(reg_data_tokens, columns = ['id', 'tokens', 'labels'])
reg_data_tokens_df


pos_data_tokens_df = pd.DataFrame(pos_data_tokens, columns = ['id', 'tokens', 'labels'])
pos_data_tokens_df



