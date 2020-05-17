import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

def ngram_representation(train, test, pipeline):
    train_sentences = train["abstract"]
    test_sentences = test["abstract"]
    train_features = pipeline.fit_transform(train_sentences)
    train_labels = train["label"]
    test_features = pipeline.transform(test_sentences)
    test_labels = test["label"]
    return train_features, train_labels, test_features, test_labels

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
    folds = data["fold"].unique()
    metrics = update_metrics(0)
    for fold in folds:
        fold_train = data[data["fold"] != fold]
        fold_test = data[data["fold"] == fold]
        train_features, train_labels, test_features, test_labels = ngram_representation(fold_train, fold_test, pipeline)
        lin_clf = svm.LinearSVC()
        lin_clf.fit(train_features, train_labels)
        y_pred = lin_clf.predict(test_features)
        score = lin_clf.score(test_features, test_labels)
        report = classification_report(test_labels, y_pred, output_dict=True)
        metrics = update_metrics(score, metrics, report)

    print(f"Accuracy: {metrics['accuracy'] / len(folds)}")
    print(f"P(Truthful): {metrics['precision_class_positive'] / len(folds)}, R(Truthful): {metrics['recall_class_positive'] / len(folds)}, F1(Truthful): {metrics['f1_score_class_positive'] / len(folds)}")
    print(f"P(Deceptive): {metrics['precision_class_negative'] / len(folds)}, R(Deceptive): {metrics['recall_class_negative'] / len(folds)}, F1(Deceptive): {metrics['f1_score_class_negative'] / len(folds)}")


if __name__ == "__main__":
    data = pd.read_csv("final_data.csv")
    pipelines = {
        "Unigram": Pipeline([('vect', CountVectorizer(ngram_range = (1, 1)))]),
        "Bigram": Pipeline([('vect', CountVectorizer(ngram_range = (1, 2)))]),
        "Trigram": Pipeline([('vect', CountVectorizer(ngram_range=(1, 3)))]),
        "Unigram-without-stopwords": Pipeline([('vect', CountVectorizer(stop_words='english', ngram_range = (1, 1)))]),
        "Bigram-without-stopwords": Pipeline([('vect', CountVectorizer(stop_words='english', ngram_range = (1, 2)))]),
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
        cross_validation(data, pipelines[key])


