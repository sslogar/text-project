import pandas as pd
import nltk
import re
import string
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import sklearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

stop_words = nltk.corpus.stopwords.words('english')

presidents = pd.read_csv('inaug_speeches.csv', encoding='ISO-8859-1') #take care of Unicode characters
# presidents = pd.read_csv('inaug_speeches.csv', encoding='UTF-8')

bush = presidents[presidents['Name'] == 'George W. Bush'] #filter for just Bush
# print(bush)

bush2001 = bush.iloc[0, :] #data frame with first Bush speech
bush2005 = bush.iloc[1, :] #data frame with 2nd Bush speech

sent2001 = sent_tokenize(bush2001['text']) #tokenize into sentences with nltk
sent2005 = sent_tokenize(bush2005['text'])

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

def rebuild(term_vec):
    for i in range(0, len(term_vec)):
        doc = ""
        doc = ' '.join(term_vec[i])
        term_vec[i] = doc
    return term_vec

term_vec2001 = tokenize_words(sent2001)
term_vec2001 = clean(term_vec2001)
term_vec2001 = remove_stop_words(term_vec2001, stop_words)
term_vec2001 = porter_stem(term_vec2001)
docs2001 = rebuild(term_vec2001)

corpus_df=pd.DataFrame({'Document':docs2001})

# for topic_weights in tt_matrix:
#     topic = [(token,weight) for token, weight in zip(vocab, topic_weights)]
#     topic = sorted(topic, key=lambda x: -x[1])
#     topic = [item for item in topic if item[1] >2]
#     print(topic)
#     print()

#spectral clustering
#msd

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
    matrix = normalize(matrix)
    sim = cosine_similarity(matrix)
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

tt = ldirichlet(docs2001)
kmeans(tt, corpus_df)

term_vec2005 = tokenize_words(sent2005)
term_vec2005 = clean(term_vec2005)
term_vec2005 = remove_stop_words(term_vec2005, stop_words)
term_vec2005 = porter_stem(term_vec2005)
docs2005 = rebuild(term_vec2005)

corpus_df=pd.DataFrame({'Document':docs2005})

tt = ldirichlet(docs2005)
kmeans(tt, corpus_df)
