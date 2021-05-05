from typing import TypeVar, List, Tuple, Dict, Iterable, NamedTuple, Set
from collections import defaultdict, Counter
import re
import random
import math
import time # compute time of execution 
import glob
import mailbox
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

X = TypeVar('X')  # generic type to represent a data point
porter = PorterStemmer()

def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    """Split data into fractions [prob, 1 - prob]"""
    data = data[:]                    # Make a shallow copy
    random.shuffle(data)              # because shuffle modifies the list.
    cut = int(len(data) * prob)       # Use prob to find a cutoff
    return data[:cut], data[cut:]     # and split the shuffled list there.

def read_messages(path: str):
    """receive a system path and return the read information of the email files in the directory and return a pandas dataframe"""
    is_spam = []
    subject = []
    for filename in glob.glob(path):
        spam = "ham" not in filename
        mail = mailbox.mbox(filename)
        if len(mail.keys()) > 0:
            tags = mail[0].keys()
            if 'Subject' in tags:
                subject.append(mail[0]['subject'])
                is_spam.append(spam)

    dictionary = {"is_spam":is_spam, 
                  "subject": subject}
    df = pd.DataFrame(dictionary)

    return df
            
def split_df(df, ratio: float, seed = 0):
    """Split Dataframe based on the ratio provided"""
    if ratio > 1:
        raise ValueError("Invalid value, please select an amount in the range 0 < ratio < 1")
    else:
        train_df = df.sample(frac = ratio, random_state = seed) # split DataFrame into the train part
        test_df = df.drop(train_df.index)
        return train_df, test_df

def count_messages(df, label:str = 'is_spam'):
    """Responsible for count the amount of messages that are spam or not spam"""
    if not label in df.keys():
        raise ValueError('Entry label is not availabe in the Dataframe selected')
    else:
        not_spam, spam = df[label].value_counts(normalize = False)
        return spam, not_spam

def stemSentence(sentence): # function got from https://www.datacamp.com/community/tutorials/stemming-lemmatization-python
    token_words = word_tokenize(sentence)
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

def pre_process_message(df, label:str = 'subject'):
    """This function will be responsible for removing punctuation, spit the message and stemm it"""
    df[label] = df[label].astype(str).str.replace('[^\w\s]','') # remove puctuation
    df[label] = df[label].astype(str).str.lower() # change words to lowercase
    #print(df.head())
    #print(df[label])
    df[label] = df[label].apply(stemSentence) # stemming the text from the message
    df[label] = df[label].astype(str).str.split() # split string into a list of strings
    print(df.head())


df = read_messages('emails/*/*/*')
train_messages, test_messages = split_df(df, 0.75)
spam, not_spam = count_messages(train_messages)
#print(train_messages.head())
pre_process_message(train_messages)
