import pandas as pd
from dataset import FileWork

class Preparation_df:
    @staticmethod
    def display_dataset_info(dataset: pd.DataFrame):
        print("Информация о датасете до изменения:")
        print(dataset.info())
        print("\n\tПервые 5 строк датасета:")
        print(dataset.head())
        print("\n\tСтатистическая сводка числовых данных:")
        print(dataset.describe())
        print("\n\tКоличество пропущенных значений в каждом столбце:")
        print(dataset.isnull().sum())
    @staticmethod
    def filter_columns(dataset: pd.DataFrame) -> pd.DataFrame:
        required_columns = ['cve_id', 'vendor_project', 'product', 'cvss', 'severity', 'required_action']
        filtered_dataset = dataset[required_columns]
        return filtered_dataset
    @staticmethod
    def display_filtered_info(dataset: pd.DataFrame):
        print("Информация об измененном датасете:")
        print(dataset.info())
        print("\n\tПервые 5 строк измененного датасета:")
        print(dataset.head())
        print("\n\tКоличество пропущенных значений в каждом столбце:")
        print(dataset.isnull().sum())
    @staticmethod
    def save_dataset(dataset: pd.DataFrame, save_path: str):
        dataset.to_csv(save_path, index=False)
    @staticmethod
    def main():
        dataset = pd.read_csv(FileWork.path_to_file())
        Preparation_df.display_dataset_info(dataset)
        filtered_dataset = Preparation_df.filter_columns(dataset)
        Preparation_df.display_filtered_info(filtered_dataset)
        save_path = "D:\\university\\7_semestr\\kursach\\dataset\\filtered_dataset.csv"
        Preparation_df.save_dataset(filtered_dataset, save_path)
        return filtered_dataset

