import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

nltk.download("stopwords")

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def tokenize(sentence):
    
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-zA-Z ]", "", sentence)
    tokens = sentence.split()

    clean = []
    for word in tokens:
        if word not in stop_words:
            clean.append(stemmer.stem(word))

    return clean


def load_data(data):
    text = []
    numbers = []

    for filename in os.listdir(data):
        if filename.endswith(".txt"):
            path = os.path.join(data, filename)
            with open(path, "r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    sentence, number = line.rsplit("\t", 1)
                    tokens = tokenize(sentence)
                    text.append(" ".join(tokens))
                    numbers.append(int(number))
    
    return text, numbers

def build_matrix(texts, max_features=2048):
    vectorizer = CountVectorizer(max_features=max_features)
    D = vectorizer.fit_transform(texts)
    words = vectorizer.get_feature_names_out()
    return D, words
