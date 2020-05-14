import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk.tokenize import word_tokenize

def nltk_ngrams(data):
    sentences = data["Sentence"].tolist()
    trigrams = []
    sentence_trigrams = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        current_trigrams = nltk.trigrams(words)
        current_trigrams = list(current_trigrams)
        sentence_trigrams.append(current_trigrams)
        trigrams.extend(current_trigrams)

    freq_dist = nltk.FreqDist(trigrams)
    kneser_ney = nltk.KneserNeyProbDist(freq_dist)
    features = []
    for current_trigrams in sentence_trigrams:
        prob = []
        for trigram in current_trigrams:
            prob.append(kneser_ney.prob(trigram))
        features.append(prob)
    features = np.array(features)




def scikit_ngrams(data):
    vectorizer = CountVectorizer()
    sentences = data["Sentence"]
    sentence_features = vectorizer.fit_transform(sentences)
    # print(vectorizer.get_feature_names())
    # print(sentence_features.toarray()[0])
    bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
    probs = bigram_vectorizer.fit_transform(sentences)
    # print(probs.toarray()[0])
    # print(bigram_vectorizer.get_feature_names())
    trigram_vectorizer = CountVectorizer(ngram_range=(3, 3))
    probs = trigram_vectorizer.fit_transform(sentences)
    # print(probs.toarray()[0])


def cross_validation(data):
    folds = data["Fold"].unique()

    for fold in folds:
        fold_train = data[data["Fold"] != fold]
        fold_test = data[data["Fold"] == fold]
        nltk_ngrams(fold_train)
        scikit_ngrams(fold_train)
        break


if __name__ == "__main__":
    data = pd.read_csv("final_data.csv")


    # print(data.head())
    # x = vectorizer.fit_transform(sentences)
    # print(vectorizer.get_feature_names())
    # tokens = nltk.word_tokenize(sentences[0])
    # print(tokens)

    cross_validation(data)
