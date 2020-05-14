from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

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

def cross_validation(data, ngram):
    data["abstract"].apply(lambda x: x.lower())
    folds = data["fold"].unique()
    metrics = update_metrics(0)
    for fold in folds:
        fold_train = data[data["fold"] != fold]
        fold_test = data[data["fold"] == fold]

        train_features, test_features = create_word_vector(fold_train['abstract'], fold_test['abstract'], ngram)
        train_labels = fold_train['label']
        test_labels = fold_test['label']

        naive_bayes = MultinomialNB()
        naive_bayes.fit(train_features, train_labels)

        y_pred = naive_bayes.predict(test_features)
        score = naive_bayes.score(test_features, test_labels)
        report = classification_report(test_labels, y_pred, output_dict=True)
        metrics = update_metrics(score, metrics, report)

    for metric in metrics:
        metrics[metric] = metrics[metric] / len(folds)
        print(f"{metric} : {metrics[metric]}")


def main():
    ngram_types = [["Unigram" , (1,1)], ["Bigram", (1,2)], ["Trigram", (2,3)]]
    df = pd.read_csv("final_data.csv")

    for ngram in ngram_types:
        print(ngram[0])
        cross_validation(df, ngram[1])

        # print('Accuracy score: ', accuracy_score(y_test, predictions))
        # print('Precision score: ', precision_score(y_test, predictions))
        # print('Recall score: ', recall_score(y_test, predictions))
        # print('Recall score: ', f1_score(y_test, predictions))
        # print('************')

if __name__ == "__main__":
    main()