import pandas as pd
import os

class FileWork:
    @staticmethod
    def excel_to_csv(excel_file_path, csv_file_path, sheet_name=0):
        df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
        df.to_csv(csv_file_path, index=False)
        print(f'Файл {excel_file_path} успешно преобразован в {csv_file_path}')
    @staticmethod
    def create_new_file_path(old_file_path, suffix='_new', new_extension='.csv'):
        directory, filename = os.path.split(old_file_path)
        file_base, file_extension = os.path.splitext(filename)
        new_filename = f"{file_base}{suffix}{new_extension}"
        new_file_path = os.path.join(directory, new_filename)
        return new_file_path
    @staticmethod
    def file_exists(file_path):
        return os.path.exists(file_path)
    @staticmethod
    def path_to_file():
        path = "D:\\university\\7_semestr\\kursach\\dataset\\2022-06-08-enriched.csv"
        if FileWork.file_exists(path):
            print("Файл существует.")
        else:
            print("Файл не найден.")
        return path
    @staticmethod
    def work_with_file():
        print('\t\tРабота с файлом')
        file_path = FileWork.path_to_file()
        path_to_save = FileWork.create_new_file_path(file_path)
        if file_path.endswith('.xls') or file_path.endswith('.xlsx'):
            FileWork.excel_to_csv(file_path, path_to_save)
        elif file_path.endswith('.csv'):
            print("У файла уже расширение .csv\n", 250 * str('='))
            return file_path

