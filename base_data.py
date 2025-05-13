import re
import string

import nltk
from nltk import wordpunct_tokenize
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm.auto import tqdm

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')


stop_words = set(stopwords.words('english'))

class BaseData:

    def __init__(self):
        self.tokenize_texts_list: list = []
        self.tokenize_texts_list_without_stop_words: list = []
        self.tokenize_tests_list_lemmtize: list = []

    @staticmethod
    def clean_text(text):
        text = "".join([ch if ch not in string.punctuation else ' ' for ch in text])
        text = re.sub(r'\s+', ' ', text, flags=re.I)
        return text

    def tokenize_text_list(self, texts):
        for text in tqdm(texts):
            tokens = word_tokenize(text) # токкенизация вариант 1
            #tokens = sent_tokenize(text) # токкенизация предложениями
            #tokens = wordpunct_tokenize(text) # токкенизация вариант 2
            self.tokenize_texts_list.append(tokens)

    def delete_tokenize_stop_words(self):
        for text in self.tokenize_texts_list:
            filtered_tokens = [word for word in text if word not in stop_words]
            self.tokenize_texts_list_without_stop_words.append(filtered_tokens)

    def lemmatize_tokenize_test(self):
        lemmatizer = WordNetLemmatizer()
        for text in self.tokenize_texts_list_without_stop_words:
            lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
            lemmatized_words = " ".join(lemmatized_words)
            self.tokenize_tests_list_lemmtize.append(lemmatized_words)
