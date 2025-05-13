from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


class Model:


    def __init__(self):
        ...

    @staticmethod
    def get_naive_bayes_classifier_model():
        return Pipeline(
            [
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', MultinomialNB()),
            ]
        )

    @staticmethod
    def get_linear_support_vector_model():
        return Pipeline(
            [
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(
                    loss='hinge', penalty='l2', alpha=1e-3,
                    random_state=42, max_iter=10, tol=None)),
            ]
        )

    @staticmethod
    def get_logistic_regression_model():
        return Pipeline(
            [
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
            ]
        )
