import json


paths_json: list[str] = [
    'datasets/dataset_json/certbot.json',
    'datasets/dataset_json/compose.json',
    'datasets/dataset_json/django_rest_framework.json',
    'datasets/dataset_json/flask.json',
    'datasets/dataset_json/keras.json',
    'datasets/dataset_json/mitmproxy.json',
    'datasets/dataset_json/pipenv.json',
    'datasets/dataset_json/requests.json',
    'datasets/dataset_json/scikit-learn.json',
    'datasets/dataset_json/scrapy.json',
    'datasets/dataset_json/spaCy.json',
    'datasets/dataset_json/tornado.json'
]

paths_txt: list[str] = [
    'datasets/dataset_txt/certbot/certbot',
    'datasets/dataset_txt/compose/compose',
    'datasets/dataset_txt/django_rest_framework/django_rest_framework',
    'datasets/dataset_txt/flask/flask',
    'datasets/dataset_txt/keras/keras',
    'datasets/dataset_txt/mitmproxy/mitmproxy',
    'datasets/dataset_txt/pipenv/pipenv',
    'datasets/dataset_txt/requests/requests',
    'datasets/dataset_txt/scikit_learn/scikit-learn',
    'datasets/dataset_txt/scrapy/scrapy',
    'datasets/dataset_txt/spaCy/spaCy',
    'datasets/dataset_txt/tornado/tornado'
]


class GeneratorDatasetTXT:


    def __init__(self, paths_json: list[str], paths_txt: list[str]):
        self.paths_json = paths_json
        self.paths_txt = paths_txt

    def convert_dataset_json_to_txt(self):

        for number_path in range(len(self.paths_json)):
            with open(paths_json[number_path], 'r') as file_read:
                data_bugs_report = json.load(file_read)

            for type_bugs in data_bugs_report:
                for number_bugs, data_bugs in data_bugs_report[type_bugs].items():
                    with open(f'{paths_txt[number_path]}_{number_bugs}.txt', 'w',
                              encoding='utf-8') as file_write:
                        file_write.write(data_bugs['issue_description'])
                    with open(f'{paths_txt[number_path]}_{number_bugs}_summary.txt', 'w',
                              encoding='utf-8') as file_write:
                        file_write.write(data_bugs['issue_summary'])


GDT = GeneratorDatasetTXT(paths_json=paths_json, paths_txt=paths_txt)
GDT.convert_dataset_json_to_txt()
