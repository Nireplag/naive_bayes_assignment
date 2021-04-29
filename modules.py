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
            
