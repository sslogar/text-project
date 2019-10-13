import pandas as pd
import nltk
import re
import string
from nltk.tokenize import sent_tokenize, word_tokenize

stop_words = nltk.corpus.stopwords.words('english')

presidents = pd.read_csv('inaug_speeches.csv', encoding='ISO-8859-1') #take care of Unicode characters

# print(presidents)
bush = presidents[presidents['Name'] == 'George W. Bush'] #filter for just Bush
# print(bush)

bush2001 = bush.iloc[0, :] #data frame with first Bush speech
bush2005 = bush.iloc[1, :] #data frame with 2nd Bush speech

sent2001 = sent_tokenize(bush2001['text']) #tokenize into sentences with nltk
sent2005 = sent_tokenize(bush2005['text'])


punc = re.compile('[%s]' % re.escape(string.punctuation))

def word_t(doc, punc):
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

term_vec2001 = word_t(sent2001, punc)
term_vec2001 = remove_stop_words(term_vec2001, stop_words)
term_vec2001 = porter_stem(term_vec2001)
# term_vec2001 = clean(term_vec2001)
for vec in term_vec2001:
    print(vec)
