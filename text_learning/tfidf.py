import pickle

word_data = pickle.load(open('your_word_data.pkl'))
print word_data[152]

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words = 'english')
vectorizer.fit_transform(word_data)
print len(vectorizer.get_feature_names())
print vectorizer.get_feature_names()[34597]

#chars = set()
#for word in vectorizer.get_feature_names():
#    for char in word:
#        chars.add(char)
#print sorted(list(chars))
