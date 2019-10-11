import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

presidents = pd.read_csv('inaug_speeches.csv', encoding='ISO-8859-1') #take care of Unicode characters

# print(presidents)
bush = presidents[presidents['Name'] == 'George W. Bush'] #filter for just Bush
# print(bush)

bush2001 = bush.iloc[0, :] #data frame with first Bush speech
bush2005 = bush.iloc[1, :] #data frame with 2nd Bush speech

sent2001 = sent_tokenize(bush2001['text']) #tokenize into sentences with nltk
sent2005 = sent_tokenize(bush2005['text'])

print(len(sent2001)) #96 sentences
