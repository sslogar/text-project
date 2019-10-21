import pandas as pd
import nltk
import re
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentiment_module import sentiment
import seaborn as sns
import matplotlib.pyplot as plt

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

def vader(sentence, threshold=0.1, verbose=False):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(sentence)
    agg_score = scores['compound']
    final_sentiment = 'positive' if agg_score >= threshold else 'negative'
    s_d = {'final_sentiment': final_sentiment, 'Polarity': round(agg_score, 2)}
    if verbose:
        positive = str(round(scores['pos'], 2) * 100) + '%'
        final = round(agg_score, 2)
        negative = str(round(scores['neg'], 2) * 100) + '%'
        neutral = str(round(scores['neu'], 2) * 100) + '%'
        sentiment_frame = pd.DataFrame([[final_sentiment, final, positive, negative, neutral]],
                                        columns=pd.MultiIndex(levels=[['Sentiment Stats:'],
                                        ['Predicted Sentiment', 'Polarity Score', 'Postive', 'negative', 'Neutral']],
                                        labels=[[0,0,0,0,0], [0,1,2,3,4]]
                                        ))
        print(sentiment_frame)
    return s_d

def return_vader_sentiment(stemmed):
    overall_sentiment=[]
    for sentence in stemmed:
        pred = vader(sentence, threshold=0.4, verbose=False)
        overall_sentiment.append(pred)
    overall_sentiment = pd.DataFrame(overall_sentiment)
    ax = sns.lineplot(x=range(1, len(overall_sentiment.index)+1), y='Polarity', data=overall_sentiment)
    plt.show()
    return overall_sentiment

def show_graphs(df):
    sns.set()
    # Plot scatterplot of arousal vs valence
    ax = sns.scatterplot(x="arousal", y="valence", data=df)
    plt.show()

    ax = sns.distplot(df['arousal']) #distribution of arousal
    plt.show()

    ax = sns.distplot(df['valence']) #distribution of valence
    plt.show()

#####Read in the President Data#####
stop_words = nltk.corpus.stopwords.words('english')

presidents = pd.read_csv('inaug_speeches.csv', encoding='ISO-8859-1') #take care of Unicode characters
# presidents = pd.read_csv('inaug_speeches.csv', encoding='UTF-8')

bush = presidents[presidents['Name'] == 'George W. Bush'] #filter for just Bush
# print(bush)

#####Sentiment for George Bush in 2001#####
bush2001 = bush.iloc[0, :] #data frame with first Bush speech
sent2001 = sent_tokenize(bush2001['text']) #tokenize into sentences with nltk
term_vec2001 = tokenize_words(sent2001)
term_vec2001 = clean(term_vec2001)
term_vec2001 = remove_stop_words(term_vec2001, stop_words)
term_vec2001 = porter_stem(term_vec2001)
sentiment2001 = return_sentiment(term_vec2001)
s_2001 = pd.DataFrame(sentiment2001) #make a DataFrame out of the dictionary returned from the sentiment function
stemmed_sentences_2001 = rebuild(term_vec2001)
overall_sentiment2001 = return_vader_sentiment(stemmed_sentences_2001)

#####Sentiment for George Bush in 2005#####
bush2005 = bush.iloc[1, :] #data frame with 2nd Bush speech
sent2005 = sent_tokenize(bush2005['text'])
term_vec2005 = tokenize_words(sent2005)
term_vec2005 = clean(term_vec2005)
term_vec2005 = remove_stop_words(term_vec2005, stop_words)
term_vec2005 = porter_stem(term_vec2005)
sentiment2005 = return_sentiment(term_vec2005)
s_2005 = pd.DataFrame(sentiment2005)
stemmed_sentences_2005 = rebuild(term_vec2005)
overall_sentiment2005 = return_vader_sentiment(stemmed_sentences_2005)


# show_graphs(s_2005)
# there appears to be a difference in sentiment between 2001 and 2005


#####Plot valence and arousal on same graph#####
s_2005['year'] = '2005'
s_2001['year'] = '2001'
overall_sentiment = pd.concat([s_2001, s_2005])


ax = sns.scatterplot(x="arousal", y="valence", style='year', data=overall_sentiment, s=75)
plt.show()
