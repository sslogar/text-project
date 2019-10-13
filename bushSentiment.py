import pandas as pd
import nltk
import re
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from sentiment_module import sentiment
import seaborn as sns
import matplotlib.pyplot as plt

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

# def clean(term_vec):
#     for i in range(0, len(term_vec)):
#         term_list = []
#         for term in term_vec[i]:
#             term = term.encode('ascii', 'ignore')
#             term = term.decode('UTF-8')
#             term_list.append(term)
#         term_vec[i] = term_list
#     return(term_vec)

# def clean(term_vec):
#     arr = []
#     for term in term_vec:
#         for word in term:
#             index = word.find('u00')
#             if (index > -1):
#                 word = word.slice(index: )

def return_sentiment(term_vec):
    sent_v = []
    for term in term_vec:
        s = sentiment.sentiment(term)
        sent_v.append(s) #append each dictionary returned by the sentiment function into a list
    return sent_v

term_vec2001 = tokenize_words(sent2001)
term_vec2001 = remove_stop_words(term_vec2001, stop_words)
term_vec2001 = porter_stem(term_vec2001)
# term_vec2001 = clean(term_vec2001)
# for vec in term_vec2001:
#     print(vec)

sentiment2001 = return_sentiment(term_vec2001)
s_2001 = pd.DataFrame(sentiment2001) #make a DataFrame out of the dictionary returned from the sentiment function

def show_graphs(df):
    sns.set()
    # Plot scatterplot of arousal vs valence
    ax = sns.scatterplot(x="arousal", y="valence", data=df)
    plt.show()

    ax = sns.distplot(df['arousal']) #distribution of arousal
    plt.show()

    ax = sns.distplot(df['valence']) #distribution of valence
    plt.show()

term_vec2005 = tokenize_words(sent2005)
term_vec2005 = remove_stop_words(term_vec2005, stop_words)
term_vec2005 = porter_stem(term_vec2005)

sentiment2005 = return_sentiment(term_vec2005)
s_2005 = pd.DataFrame(sentiment2005)

show_graphs(s_2005)
# there appears to be a difference in sentiment between 2001 and 2005
