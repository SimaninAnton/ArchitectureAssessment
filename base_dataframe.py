import os
import pandas

from tqdm.auto import tqdm

from config import Config


class BaseDataFrame:

    """Базовый класс для работы с датафреймом"""

    def __init__(self):
        self._df = None
        self.numbers = [i for i in range(0, 299)]

    @property
    def df(self):
        if self._df is not None:
            return self._df
        else:
            self.generate_dataframe()
            return self._df

    def generate_dataframe(self):
        services: list = []
        microservices: list = []
        others: list = []
        final_classes_list: list = []

        for file_name in tqdm(os.listdir(Config.path_services)):
            with open(f'{Config.path_services}/{file_name}', 'r', encoding='utf-8') as file:
                data = file.read()
                services.append(data)

        for file_name in tqdm(os.listdir(Config.path_microservices)):
            with open(f'{Config.path_microservices}/{file_name}', 'r', encoding='utf-8') as file:
                data = file.read()
                microservices.append(data)

        for file_name in tqdm(os.listdir(Config.path_others)):
            with open(f'{Config.path_others}/{file_name}', 'r', encoding='utf-8') as file:
                data = file.read()
                others.append(data)

        for i in range(len(services)):
            final_classes_list.append(1)

        for i in range(len(microservices)):
            final_classes_list.append(2)

        for i in range(len(others)):
            final_classes_list.append(3)

        final_description_list = services + microservices + others
        final_data: dict = {"description": final_description_list, "class_cri": final_classes_list}

        self._df = pandas.DataFrame(data=final_data)
