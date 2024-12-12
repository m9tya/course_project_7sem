from file_infrormation import Preparation_df
import numpy as np
import os
import gc
import pandas as pd
from apyori import apriori
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

# class Apriori:
#     @staticmethod
#     def apriori_algorithm():
#
#         df = pd.read_csv("D:\\university\\7_semestr\\kursach\\dataset\\filtered_dataset.csv")
#
#         transactions = []
#         for _, row in df.iterrows():
#             transaction = [str(item) for item in row if pd.notna(item)]
#             transactions.append(transaction)
#
#         results = list(apriori(transactions, min_support=0.2, min_confidence=0.3, min_lift=1))
#
#         output_file_path = "D:\\university\\7_semestr\\kursach\\dataset\\apriori_results.txt"
#
#         with open(output_file_path, mode='w', encoding='utf-8') as file:
#             arr = []
#
#             for relation_record in results:
#                 itemset = relation_record.items
#                 support = relation_record.support
#                 if len(itemset) == 1:
#                     if '' not in itemset:
#                         result = {"itemset": itemset, "support": support}
#                         arr.append(result)
#                         file.write(f"Itemset: {itemset} (Support: {support:.2f})\n")
#                 else:
#                     ordered_statistics = relation_record.ordered_statistics
#                     for rule in ordered_statistics:
#                         antecedent = rule.items_base
#                         consequent = rule.items_add
#                         confidence = rule.confidence
#                         lift = rule.lift
#                         if '' not in antecedent and '' not in consequent:
#                             file.write(f"Правило: {antecedent} -> {consequent}\n")
#                             file.write(f"  Поддержка: {support:.2f}\n")
#                             file.write(f"  Уверенность: {confidence:.2f}\n")
#                             file.write(f"  Лифт: {lift:.2f}\n")
#                             file.write("\n")
#
#                             arr.append({"antecedent": antecedent, "consequent": consequent,
#                                         "support": support, "confidence": confidence, "lift": lift})
import time
import psutil

class Apriori:
    @staticmethod
    def apriori_algorithm():
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / (1024 * 1024)
        df = pd.read_csv("D:\\university\\7_semestr\\kursach\\dataset\\filtered_dataset.csv")
        transactions = []
        for _, row in df.iterrows():
            transaction = [str(item) for item in row if pd.notna(item)]
            transactions.append(transaction)
        results = list(apriori(transactions, min_support=0.2, min_confidence=0.3, min_lift=1))
        memory_after = psutil.Process().memory_info().rss / (1024 * 1024)
        end_time = time.time()
        execution_time = end_time - start_time
        memory_usage = memory_after - memory_before
        output_file_path = "D:\\university\\7_semestr\\kursach\\dataset\\apriori_results.txt"

        with open(output_file_path, mode='w', encoding='utf-8') as file:
            arr = []
            for relation_record in results:
                itemset = relation_record.items
                support = relation_record.support
                if len(itemset) == 1:
                    if '' not in itemset:
                        result = {"itemset": itemset, "support": support}
                        arr.append(result)
                        file.write(f"Itemset: {itemset} (Support: {support:.2f})\n")
                else:
                    ordered_statistics = relation_record.ordered_statistics
                    for rule in ordered_statistics:
                        antecedent = rule.items_base
                        consequent = rule.items_add
                        confidence = rule.confidence
                        lift = rule.lift
                        if '' not in antecedent and '' not in consequent:
                            file.write(f"Правило: {antecedent} -> {consequent}\n")
                            file.write(f"  Поддержка: {support:.2f}\n")
                            file.write(f"  Уверенность: {confidence:.2f}\n")
                            file.write(f"  Лифт: {lift:.2f}\n")
                            file.write("\n")
                            arr.append({"antecedent": antecedent, "consequent": consequent,
                                        "support": support, "confidence": confidence, "lift": lift})
        num_rules = len([rule for rule in arr if "confidence" in rule])
        avg_confidence = (
            sum(rule["confidence"] for rule in arr if "confidence" in rule) / num_rules
            if num_rules > 0 else 0
        )
        avg_lift = (
            sum(rule["lift"] for rule in arr if "lift" in rule) / num_rules
            if num_rules > 0 else 0
        )
        print("Apriori")
        print(f"Время выполнения: {execution_time:.2f} секунд")
        print(f"Использование памяти: {memory_usage:.2f} MB")
        print(f"Количество правил: {num_rules}")
        print(f"Средняя уверенность: {avg_confidence:.2f}")
        print(f"Средний лифт: {avg_lift:.2f}")

