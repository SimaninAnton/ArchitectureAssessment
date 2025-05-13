import pandas
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import memory_profiler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from base_data import BaseData
from base_dataframe import BaseDataFrame
from model import Model


numbers = [i for i in range(0, 60)]

@memory_profiler.profile
def education_models():
    """Получение датафрейма"""
    base_data_frame = BaseDataFrame()
    df = base_data_frame.df
    df.info()
    print(df)

    """Обработка текста"""
    base_data = BaseData()

    df['description'] = df.apply(lambda x: base_data.clean_text(x['description']), axis=1)
    df['description'].dropna(inplace=True)

    base_data.tokenize_text_list(df['description'])
    base_data.delete_tokenize_stop_words()
    base_data.lemmatize_tokenize_test()

    """Обучение модели"""
    df['text_stem'] = base_data.tokenize_tests_list_lemmtize
    X = df['text_stem']
    y = df['class_cri']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 42)

    models = Model()

    dfr = {"number": numbers,"text_stem": X_test, "class_cri": y_test}
    dfr = pandas.DataFrame(data=dfr)

    """Классификатор Naive Bayes Classifier"""
    nbs = models.get_naive_bayes_classifier_model()
    nbs.fit(X_train, y_train)

    sns.relplot(x="number", y='class_cri', data=dfr)

    y_pred = nbs.predict(X_test)

    plt.yticks(np.arange(min(y), max(y)+1, 1.0))
    plt.show()

    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred))


    """Модель Linear Support Vector Machine"""
    sgd = models.get_linear_support_vector_model()
    sgd.fit(X_train, y_train)

    y_pred = sgd.predict(X_test)

    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred))


    """Модель Logistic Regression"""

    lr = models.get_logistic_regression_model()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)

    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred))

education_models()