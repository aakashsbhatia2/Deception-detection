from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

def create_word_vector(X_train, X_test, pipeline):
    # cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english', ngram_range=ngram)

    X_train_cv = pipeline.fit_transform(X_train)
    X_test_cv = pipeline.transform(X_test)

    return X_train_cv, X_test_cv

def naive_bayes(X_train_cv, y_train, X_test_cv):
    naive_bayes = MultinomialNB()
    naive_bayes.fit(X_train_cv, y_train)
    predictions = naive_bayes.predict(X_test_cv)
    return predictions

def update_metrics(score, metrics = {}, report = None):
    if len(metrics) == 0 or report == None:
        return {
            'precision_class_positive': 0,
            'recall_class_positive': 0,
            'f1_score_class_positive': 0,
            'precision_class_negative': 0,
            'recall_class_negative': 0,
            'f1_score_class_negative': 0,
            'accuracy': 0
        }
    metrics['accuracy'] += score
    metrics['precision_class_positive'] += report['1.0']['precision']
    metrics['recall_class_positive'] += report['1.0']['recall']
    metrics['f1_score_class_positive'] += report['1.0']['f1-score']
    metrics['precision_class_negative'] += report['0.0']['precision']
    metrics['recall_class_negative'] += report['0.0']['recall']
    metrics['f1_score_class_negative'] += report['0.0']['f1-score']
    return metrics

def cross_validation(data, pipeline):
    data["abstract"].apply(lambda x: x.lower())
    folds = data["fold"].unique()
    metrics = update_metrics(0)
    for fold in folds:
        fold_train = data[data["fold"] != fold]
        fold_test = data[data["fold"] == fold]

        train_features, test_features = create_word_vector(fold_train['abstract'], fold_test['abstract'], pipeline)
        train_labels = fold_train['label']
        test_labels = fold_test['label']

        naive_bayes = MultinomialNB()
        naive_bayes.fit(train_features, train_labels)

        y_pred = naive_bayes.predict(test_features)
        score = naive_bayes.score(test_features, test_labels)
        report = classification_report(test_labels, y_pred, output_dict=True)
        metrics = update_metrics(score, metrics, report)

    print(f"Accuracy: {metrics['accuracy'] / len(folds)}")
    print(f"P(Truthful): {metrics['precision_class_positive'] / len(folds)}, R(Truthful): {metrics['recall_class_positive'] / len(folds)}, F1(Truthful): {metrics['f1_score_class_positive'] / len(folds)}")
    print(f"P(Deceptive): {metrics['precision_class_negative'] / len(folds)}, R(Deceptive): {metrics['recall_class_negative'] / len(folds)}, F1(Deceptive): {metrics['f1_score_class_negative'] / len(folds)}")


def main():
    # ngram_types = [["Unigram" , (1,1)], ["Bigram", (1,2)], ["Trigram", (2,3)]]
    df = pd.read_csv("final_data.csv")
    pipelines = {
        "Unigram": Pipeline([('vect', CountVectorizer(ngram_range=(1, 1)))]),
        "Bigram": Pipeline([('vect', CountVectorizer(ngram_range=(1, 2)))]),
        "Trigram": Pipeline([('vect', CountVectorizer(ngram_range=(1, 3)))]),
        "Unigram-without-stopwords": Pipeline([('vect', CountVectorizer(stop_words='english', ngram_range=(1, 1)))]),
        "Bigram-without-stopwords": Pipeline([('vect', CountVectorizer(stop_words='english', ngram_range=(1, 2)))]),
        "Trigram-without-stopwords": Pipeline([('vect', CountVectorizer(stop_words='english', ngram_range=(1, 3)))]),
        "Unigram-Tfidf": Pipeline([
            ('vect', CountVectorizer(ngram_range=(1, 1))),
            ('tfidf', TfidfTransformer())
        ]),
        "Bigram-Tfidf": Pipeline([
            ('vect', CountVectorizer(ngram_range=(1, 2))),
            ('tfidf', TfidfTransformer())
        ]),
        "Trigram-Tfidf": Pipeline([
            ('vect', CountVectorizer(ngram_range=(1, 3))),
            ('tfidf', TfidfTransformer())
        ]),
    }
    for key in pipelines:
        print(f"---------{key}----------")
        cross_validation(df, pipelines[key])

if __name__ == "__main__":
    main()