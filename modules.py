from typing import TypeVar, List, Tuple, Dict, Iterable, NamedTuple, Set
from collections import defaultdict, Counter
import re
import random
import math
import time # compute time of execution 
import glob
import mailbox
import pandas as pd

X = TypeVar('X')  # generic type to represent a data point

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


df = read_messages('emails/*/*/*')
print(df.keys())
train_messages, test_messages = split_df(df, 0.75)
spam, not_spam = count_messages(train_messages)

print(spam)
print(not_spam)
