from typing import List, Tuple, Dict, Iterable, NamedTuple, Set
from collections import defaultdict, Counter
import re
import random
import math
import time # compute time of execution 
import glob
import mailbox
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


porter = PorterStemmer()


def read_messages(path: str):
    """receive a system path and return the read information of the email files in the directory and return a pandas dataframe"""
    is_spam = []
    subject = []
    for filename in glob.glob(path):
        spam = "ham" not in filename 
        mail = mailbox.mbox(filename)
        if len(mail.keys()) > 0: # check if file is not corrupted
            tags = mail[0].keys()
            if 'Subject' in tags:
                subject.append(mail[0]['subject']) # load list of message subjects
                is_spam.append(spam) # load list of messages type

    dictionary = {"is_spam":is_spam, 
                  "subject": subject} # convert into a dictionary
    df = pd.DataFrame(dictionary) # generate a DataFrame

    return df
            
def split_df(df, ratio: float, seed = 0):
    """Split Dataframe based on the ratio provided"""
    if ratio > 1:
        raise ValueError("Invalid value, please select an amount in the range 0 < ratio < 1") # validate if the split ratio is possible 
    else:
        train_df = df.sample(frac = ratio, random_state = seed) # split DataFrame into the train part
        test_df = df.drop(train_df.index)
        return train_df, test_df  # return split DataFrames

def count_messages(df, label:str = 'is_spam'):
    """Responsible for count the amount of messages that are spam or not spam"""
    if not label in df.keys():
        raise ValueError('Entry label is not availabe in the Dataframe selected') # verify if a column is part of the DataFrame
    else:
        not_spam, spam = df[label].value_counts(normalize = False) # generate the count for the different values in the column 
        return spam, not_spam

def stemSentence(sentence):
    """function got from https://www.datacamp.com/community/tutorials/stemming-lemmatization-python responsible by applying the Stemm into the string """
    token_words = word_tokenize(sentence)
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

def tokenize(text: str):
    text = text.lower()                         # Convert to lowercase,
    text = text.replace('[^\w\s]','')
    text = stemSentence(text)
    all_words = re.findall("[a-z0-9']+", text)  # extract the words, and
    return set(all_words)                       # remove duplicates.

def pre_process_message(df, label:str = 'subject'):
    """This function will be responsible for removing punctuation, spit the message, stemm it and remove duplicates"""
    df[label] = df[label].astype(str).str.replace('[^\w\s]','') # remove puctuation
    df[label] = df[label].astype(str).str.lower() # change words to lowercase
    df[label] = df[label].apply(stemSentence) # stemming the text from the message
    df[label] = df[label].astype(str).str.split() # split string into a list of strings
    df[label] = df[label].apply(set) #remove duplicates
    #print(df[label].head())

def vocabulary(df, label:str = 'subject') -> set:
    """This will create a set from all the tokens of the dataframe column indicated"""
    token_list = [] 
    for subject in df[label].to_numpy().flatten():
        for word in subject:
            token_list.append(word)
    return set(token_list)

def message_type_dict(df, is_spam:bool, is_spam_label:str = 'is_spam', dict_label:str = 'subject') -> dict:
    """Create dictionary of word count for the words inside a type of message"""

    df = df[df[is_spam_label] == is_spam]
    message_dict = dict()
    for message in df[dict_label].to_numpy().flatten():
        for word in message:
            if word in message_dict.keys():
                message_dict[word] = message_dict[word] + 1 # increment if key is already present
            else:
                message_dict[word] = 1 # create key if it do not exist
    return message_dict

class NaiveBayesClassifier:
    def __init__(self, k: float = 0.5) -> None:
        self.k = k  # smoothing factor

        self.tokens: Set[str] = set()
        self.token_spam_counts: Dict[str, int] = defaultdict(int)
        self.token_ham_counts: Dict[str, int] = defaultdict(int)
        self.spam_messages = self.ham_messages = 0

    def train(self, df) -> None:
        pre_process_message(df)
        self.spam_messages, self.ham_messages = count_messages(df)
        self.tokens = vocabulary(df)
        self.token_spam_counts = message_type_dict(df, True)
        self.token_ham_counts = message_type_dict(df, False)


    def probabilities(self, token: str) -> Tuple[float, float]:
        """returns P(token | spam) and P(token | not spam)"""
        #Code below needed to avoid situation where word only appears in SPAM or HAM, causing ErrorKey
        if token in self.token_spam_counts.keys():
            spam = self.token_spam_counts[token]
        else:
            spam = 0
        if token in self.token_ham_counts.keys():
            ham = self.token_ham_counts[token]
        else:
            ham = 0

        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
        p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)

        return p_token_spam, p_token_ham

    def predict(self, text: str) -> float:
        text_tokens = tokenize(str(text)) # force type string for messages with special characters
        log_prob_if_spam = log_prob_if_ham = 0.0

        # Iterate through each word in our vocabulary.
        for token in self.tokens:
            prob_if_spam, prob_if_ham = self.probabilities(token)

            # If *token* appears in the message,
            # add the log probability of seeing it;
            if token in text_tokens:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_ham += math.log(prob_if_ham)

            # otherwise add the log probability of _not_ seeing it
            # which is log(1 - probability of seeing it)
            else:
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_ham += math.log(1.0 - prob_if_ham)

        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham = math.exp(log_prob_if_ham)
        return prob_if_spam / (prob_if_spam + prob_if_ham)


###############################################################################
# Exemplo com mensagens verdadeiras
# 



import glob
from sklearn.metrics import confusion_matrix

# modify the path to wherever you've put the files
path = 'emails/*/*/*'

messages_df = read_messages(path)

train_messages, test_messages = split_df(messages_df, 0.75)

model = NaiveBayesClassifier()
model.train(train_messages)

predictions = []

for message in test_messages['subject']:
    predictions.append(model.predict(message))
    
predictions = np.array(predictions)


# Assume that spam_probability > 0.5 corresponds to spam prediction
# and count the combinations of (actual is_spam, predicted is_spam)
conf_matrix = confusion_matrix(test_messages['is_spam'], predictions > 0.5 )

print(conf_matrix)

def p_spam_given_token(token: str, model: NaiveBayesClassifier) -> float:
    prob_if_spam, prob_if_ham = model.probabilities(token)

    return prob_if_spam / (prob_if_spam + prob_if_ham)

words = sorted(model.tokens, key=lambda t: p_spam_given_token(t, model))

print("spammiest_words", words[-10:])
print("hammiest_words", words[:10])









#df = read_messages('emails/*/*/*')
#train_messages, test_messages = split_df(df, 0.75)
#spam, not_spam = count_messages(train_messages)
#print(train_messages.head())
#pre_process_message(train_messages)
#words = vocabulary(train_messages)
#spam_dict = message_type_dict(train_messages, True)
#print(spam_dict)

