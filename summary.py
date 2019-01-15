import pandas as pd
import os, sys, email
from talon.signature.bruteforce import extract_signature
from nltk.tokenize import sent_tokenize
import nltk
import numpy as np
from nltk.corpus import stopwords 
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from skipthoughts import *
nltk.download('punkt')
df = pd.read_csv('/Users/Srini/email_summarization/emails.csv',nrows=50)
print (df.shape)
print (df.head(5))

#Sample Email from the dataset
print(df['message'][3])
emails = df['message'].tolist()


def preprocess(emails):
    """
    Performs preprocessing operations such as:
        1. Removing signature lines (only English emails are supported)
        2. Removing new line characters.
    """
    n_emails = len(emails)
    for i in range(n_emails):
        email = emails[i]
        email, _ = extract_signature(email)
        lines = email.split('\n')
        for j in reversed(range(len(lines))):
            lines[j] = lines[j].strip()
            if lines[j] == '':
                lines.pop(j)
        emails[i] = ' '.join(lines)    
        
def split_sentences(emails):
    """
    Splits the emails into individual sentences
    """
    n_emails = len(emails)
    for i in range(n_emails):
        email = emails[i]
        sentences = sent_tokenize(email)
        #sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
        for j in reversed(range(len(sentences))):
            sent = sentences[j]
            sentences[j] = sent.strip()
            if sent == '':
                sentences.pop(j)
        emails[i] = sentences        

def skipthought_encode(emails):
    """
    Obtains sentence embeddings for each sentence in the emails
    """
    enc_emails = [None]*len(emails)
    cum_sum_sentences = [0]
    sent_count = 0
    for email in emails:
        sent_count += len(email)
        cum_sum_sentences.append(sent_count)

    all_sentences = [sent for email in emails for sent in email]
    print('Loading pre-trained models...')
    model = load_model()
    encoder = Encoder(model)
    print('Encoding sentences...')
    enc_sentences = encoder.encode(all_sentences, verbose=False)

    for i in range(len(emails)):
        begin = cum_sum_sentences[i]
        end = cum_sum_sentences[i+1]
        enc_emails[i] = enc_sentences[begin:end]
    return enc_emails

def summarize(emails):
    """
    Performs summarization of emails
    """
    n_emails = len(emails)
    summary = [None]*n_emails
    print('Preprocesing...')
    #preprocess(emails)
    print('Splitting into sentences...')
    split_sentences(emails)
    print('Starting to encode...')
    enc_emails = skipthought_encode(emails)
    print('Encoding Finished')
    for i in range(n_emails):
        enc_email = enc_emails[i]
        n_clusters = int(np.ceil(len(enc_email)**0.5))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans = kmeans.fit(enc_email)
        avg = []
        closest = []
        for j in range(n_clusters):
            idx = np.where(kmeans.labels_ == j)[0]
            avg.append(np.mean(idx))
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_,\
                                                   enc_email)
        ordering = sorted(range(n_clusters), key=lambda k: avg[k])
        summary[i] = ' '.join([emails[i][closest[idx]] for idx in ordering])
    print('Clustering Finished')
    return summary

df1 = summarize(emails)
print (df1[3])
      