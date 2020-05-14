from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def create_split(df):
    X_train, X_test, y_train, y_test = train_test_split(df['abstract'], df['label'], random_state=1)
    return X_train, X_test, y_train, y_test

def create_word_vector(X_train, X_test, ngram):
    cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english', ngram_range=ngram)

    X_train_cv = cv.fit_transform(X_train)
    X_test_cv = cv.transform(X_test)

    return X_train_cv, X_test_cv

def naive_bayes(X_train_cv, y_train, X_test_cv):
    naive_bayes = MultinomialNB()
    naive_bayes.fit(X_train_cv, y_train)
    predictions = naive_bayes.predict(X_test_cv)
    return predictions

def main():
    ngram_types = [["Unigram" , (1,1)], ["Bigram", (2,2)], ["Trigram", (3,3)]]
    df = pd.read_csv("final_data.csv")
    X_train, X_test, y_train, y_test = create_split(df)
    for ngram in ngram_types:
        print(ngram[0])
        X_train_cv, X_test_cv = create_word_vector(X_train, X_test, ngram[1])
        predictions = naive_bayes(X_train_cv, y_train, X_test_cv)

        print('Accuracy score: ', accuracy_score(y_test, predictions))
        print('Precision score: ', precision_score(y_test, predictions))
        print('Recall score: ', recall_score(y_test, predictions))
        print('************')

if __name__ == "__main__":
    main()