from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
import nltk
from sklearn import svm
import pandas as pd

def pos_representation(train, test):
    train_sentences = train["abstract"]
    test_sentences = test["abstract"]
    sentence_tokens = []
    overall_tokens = set()
    for sentence in train_sentences:
        tokens = nltk.word_tokenize(sentence)
        word_tags = nltk.pos_tag(tokens)
        tags = [word_tag[1] for word_tag in word_tags]
        sentence_tokens.append(tags)
    vectorizer = CountVectorizer(preprocessor=lambda x: x, tokenizer= lambda x: x)
    train_features = vectorizer.fit_transform(sentence_tokens)
    sentence_tokens = []
    for sentence in test_sentences:
        tokens = nltk.word_tokenize(sentence)
        word_tags = nltk.pos_tag(tokens)
        tags = [word_tag[1] for word_tag in word_tags]
        sentence_tokens.append(tags)
    test_features = vectorizer.transform(sentence_tokens)
    train_labels = train["label"]
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

def cross_validation(data):
    representation_map = {
        "unigram": (1, 1),
        "bigram": (1, 2),
        "trigram": (1, 3)
    }
    data["abstract"].apply(lambda x: x.lower())
    folds = data["fold"].unique()
    metrics = update_metrics(0)
    for fold in folds:
        fold_train = data[data["fold"] != fold]
        fold_test = data[data["fold"] == fold]
        train_features, train_labels, test_features, test_labels = pos_representation(fold_train, fold_test)
        lin_clf = svm.LinearSVC(max_iter=10000)
        lin_clf.fit(train_features, train_labels)
        y_pred = lin_clf.predict(test_features)
        score = lin_clf.score(test_features, test_labels)
        report = classification_report(test_labels, y_pred, output_dict=True)
        metrics = update_metrics(score, metrics, report)

    for metric in metrics:
        metrics[metric] = metrics[metric] / len(folds)
        print(f"{metric} : {metrics[metric]}")


if __name__ == "__main__":
    data = pd.read_csv("final_data.csv")

    print("------------Unigram Representation-----------")
    cross_validation(data)