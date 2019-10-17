import pandas as pd
import nltk
import re
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from sentiment_module import sentiment
import numpy as np
import sklearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

stop_words = nltk.corpus.stopwords.words('english')

trump = pd.read_csv('trump.csv')
print(trump)
print(trump['text'])
# sentences = sent_tokenize(trump['text']) #tokenize into sentences with nltk


def tokenize_words(doc):
    #tokenize into words
    punc = re.compile('[%s]' % re.escape(string.punctuation))
    term_vec = []
    for d in doc:
        d=d.lower()
        d= punc.sub('', d)
        term_vec.append(word_tokenize(d))
    return term_vec

def remove_stop_words(term_vec, sw):
    for i in range(0, len(term_vec)):
        term_list = []
        for term in term_vec[i]:
            if term not in sw:
                term_list.append(term)
        term_vec[i] = term_list
    return(term_vec)

def porter_stem(term_vec):
    porter = nltk.stem.PorterStemmer()
    for i in range(0, len(term_vec)):
        for j in range(0, len(term_vec[i])):
            term_vec[i][j] = porter.stem(term_vec[i][j])
    return term_vec

def clean(term_vec):
    #remove unicode codes from text (u0092, etc)
    for i in range(0, len(term_vec)):
        term_list = []
        for term in term_vec[i]:
            index = term.find('u00')
            if (index > -1):
                term = term[:index]
                print(term)
                term_list.append(term)
            else:
                term_list.append(term)
        term_vec[i] = term_list
    return term_vec

def return_sentiment(term_vec):
    sent_v = []
    for term in term_vec:
        s = sentiment.sentiment(term)
        sent_v.append(s) #append each dictionary returned by the sentiment function into a list
    return sent_v

def rebuild(term_vec):
    for i in range(0, len(term_vec)):
        doc = ""
        doc = ' '.join(term_vec[i])
        term_vec[i] = doc
    return term_vec

term_vec = tokenize_words(trump['text'])
term_vec= clean(term_vec)
term_vec = remove_stop_words(term_vec, stop_words)
term_vec = porter_stem(term_vec)
sentiment = return_sentiment(term_vec)
s = pd.DataFrame(sentiment) #make a DataFrame out of the dictionary returned from the sentiment function

def show_graphs(df):
    sns.set()
    # Plot scatterplot of arousal vs valence
    ax = sns.scatterplot(x="arousal", y="valence", data=df)
    plt.show()

    ax = sns.distplot(df['arousal']) #distribution of arousal
    plt.show()

    ax = sns.distplot(df['valence']) #distribution of valence
    plt.show()

show_graphs(s)

docs = rebuild(term_vec)

def ldirichlet(docs):
    cv = CountVectorizer(min_df=0.,max_df=1.)
    cv_matrix=cv.fit_transform(docs)
    cv_matrix
    # print(cv_matrix)
    lda=LatentDirichletAllocation(n_components=6,max_iter=10000,random_state=0)
    dt_matrix = lda.fit_transform(cv_matrix)
    features=pd.DataFrame(dt_matrix,columns=['T1','T2','T3', 'T4', 'T5', 'T6'])

    vocab=cv.get_feature_names()

    tt_matrix=lda.components_
    return tt_matrix

def kmeans(matrix, corpus_df):
    sim = cosine_similarity(tt_matrix)
    kmeans = KMeans(n_clusters=6, max_iter=10000, random_state=0).fit(sim)
    corpus_df['kmeans_cluster'] = pd.Series(kmeans.labels_)
    speech_clusters =(corpus_df[['kmeans_cluster','Document']].sort_values(
            by=['kmeans_cluster'],ascending=False).groupby('kmeans_cluster').head(10))

    speech_clusters=speech_clusters.copy(deep=True)
    for cluster_num in range(6):
        speech = speech_clusters[speech_clusters['kmeans_cluster']== cluster_num]['Document'].values.tolist()
        print('CLUSTER #'+ str(cluster_num+1))
        print('Top Document: ',speech)
        print('-'*20)

tt = ldirichlet(docs)
kmeans(tt, corpus_df)
