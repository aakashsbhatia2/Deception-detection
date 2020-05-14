import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn import svm

def ngram_representation(train, test, vectorizer):
    train_sentences = train["abstract"]
    test_sentences = test["abstract"]
    train_features = vectorizer.fit_transform(train_sentences)
    train_labels = train["label"]
    test_features = vectorizer.transform(test_sentences)
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

def cross_validation(data, representation = "unigram"):
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
        vectorizer = CountVectorizer(ngram_range = representation_map[representation])
        train_features, train_labels, test_features, test_labels = ngram_representation(fold_train, fold_test, vectorizer)
        lin_clf = svm.LinearSVC()
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
    print("")

    print("------------Bigram Representation------------")
    cross_validation(data, "bigram")

    print("")
    print("------------Trigram Representation-----------")
    cross_validation(data, "trigram")

