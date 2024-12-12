import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
import time
import psutil

class FP_Growth:
    @staticmethod
    def generate_rules(input_csv: str, output_path: str, min_support: float = 0.2, min_confidence: float = 0.3, min_lift: float = 1.0):
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / (1024 * 1024)  # В MB

        df = pd.read_csv(input_csv)
        df_encoded = pd.get_dummies(df[['vendor_project', 'product', 'severity', 'required_action']],
                                    prefix=['vendor', 'prod', 'sev', 'action'])
        binary_directory = "D:\\university\\7_semestr\\kursach\\dataset"
        binary_file_path = os.path.join(binary_directory, 'binary.csv')
        df_encoded.to_csv(binary_file_path, index=False)

        frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)

        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift, num_itemsets=1)
        rules = rules[rules['confidence'] >= min_confidence]

        memory_after = psutil.Process().memory_info().rss / (1024 * 1024)  # В MB
        end_time = time.time()
        execution_time = end_time - start_time
        memory_usage = memory_after - memory_before

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            for index, row in rules.iterrows():
                f.write(f"Правило: {', '.join(list(row['antecedents']))} -> {', '.join(list(row['consequents']))}\n")
                f.write(f"Поддержка: {row['support']}\n")
                f.write(f"Уверенность: {row['confidence']}\n")
                f.write(f"Лифт: {row['lift']}\n\n")
        num_rules = len(rules)
        avg_confidence = rules['confidence'].mean() if num_rules > 0 else 0
        avg_lift = rules['lift'].mean() if num_rules > 0 else 0

        print('FP-growth')
        print(f"Время выполнения: {execution_time:.2f} секунд")
        print(f"Использование памяти: {memory_usage:.2f} MB")
        print(f"Количество правил: {num_rules}")
        print(f"Средняя уверенность: {avg_confidence:.2f}")
        print(f"Средний лифт: {avg_lift:.2f}")

        FP_Growth.support_graphic(rules)
        FP_Growth.confidence_graphic(rules)
        FP_Growth.lift_graphic(rules)
        FP_Growth.rule_matrix(rules)
        FP_Growth.circle_graphic(rules)


    @staticmethod
    def support_graphic(Rules):
        supports = Rules['support']
        unique_supports = len(set(supports))
        mean_support = np.mean(supports)
        median_support = np.median(supports)
        plt.figure(figsize=(12, 8))
        counts, bins, _ = plt.hist(supports, bins=unique_supports, edgecolor='black', color='skyblue', alpha=0.7, label='Поддержка')
        for count, x in zip(counts, bins[:-1]):
            plt.text(x + (bins[1] - bins[0]) / 2, count + 0.1, str(int(count)), ha='center', fontsize=10)
        plt.axvline(mean_support, color='red', linestyle='--', linewidth=2, label=f'Среднее: {mean_support:.2f}')
        plt.text(mean_support, -0.08 * max(counts), f'{mean_support:.2f}', color='red', fontsize=12, ha='center')
        plt.xticks(sorted(set(supports)))
        plt.xlabel("Поддержка", fontsize=14)
        plt.ylabel("Частота", fontsize=14)
        plt.title("Распределение значений поддержки", fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def confidence_graphic(Rules):
        confidences = Rules['confidence']
        mean_confidence = np.mean(confidences)
        median_confidence = np.median(confidences)
        plt.figure(figsize=(12, 8))
        counts, bins, _ = plt.hist(confidences, bins='auto', edgecolor='black', color='skyblue', alpha=0.7, label='Уверенность')
        for count, x in zip(counts, bins[:-1]):
            plt.text(x + (bins[1] - bins[0]) / 2, count + 0.1, str(int(count)), ha='center', fontsize=10)
        plt.axvline(mean_confidence, color='red', linestyle='--', linewidth=2, label=f'Среднее: {mean_confidence:.2f}')
        plt.text(mean_confidence, -0.08 * max(counts), f'{mean_confidence:.2f}', color='red', fontsize=12, ha='center')
        plt.xlabel("Уверенность", fontsize=14)
        plt.ylabel("Частота", fontsize=14)
        plt.title("Распределение значений уверенности", fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def lift_graphic(Rules):
        lifts = Rules['lift']
        mean_lift = np.mean(lifts)
        median_lift = np.median(lifts)
        plt.figure(figsize=(12, 8))
        counts, bins, _ = plt.hist(lifts, bins='auto', edgecolor='black', color='skyblue', alpha=0.7, label='Лифт')
        for count, x in zip(counts, bins[:-1]):
            plt.text(x + (bins[1] - bins[0]) / 2, count + 0.1, str(int(count)), ha='center', fontsize=10)
        plt.axvline(mean_lift, color='red', linestyle='--', linewidth=2, label=f'Среднее: {mean_lift:.2f}')
        plt.text(mean_lift, -0.08 * max(counts), f'{mean_lift:.2f}', color='red', fontsize=12, ha='center')
        plt.xlabel("Лифт", fontsize=14)
        plt.ylabel("Частота", fontsize=14)
        plt.title("Распределение значений лифта", fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def rule_matrix(Rules, top_n=20):
        random_rules = random.sample(Rules.to_dict('records'), min(top_n, len(Rules)))
        antecedents = [', '.join(list(rule['antecedents'])) for rule in random_rules]
        consequents = [', '.join(list(rule['consequents'])) for rule in random_rules]
        supports = [rule['support'] for rule in random_rules]
        data = {"antecedent": antecedents, "consequent": consequents, "support": supports}
        df = pd.DataFrame(data)
        pivot_table = df.pivot_table(index="antecedent", columns="consequent", values="support", aggfunc="sum")
        plt.figure(figsize=(12, 10))
        sns.set(font_scale=0.8)
        ax = sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".2f", xticklabels=True, yticklabels=True,
                         cbar_kws={"shrink": 0.8})
        plt.title(f"Матрица правил для случайных {min(top_n, len(Rules))} правил", fontsize=16, fontweight='bold')
        plt.xlabel("Консеквент")
        plt.ylabel("Антецедент")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def circle_graphic(Rules, top_n=10):
        random_rules = random.sample(Rules.to_dict('records'), min(top_n, len(Rules)))
        supports = [rule['support'] for rule in random_rules]
        rules_labels = [
            f"{', '.join(list(map(str, rule['antecedents'])))} -> {', '.join(list(map(str, rule['consequents'])))}" for
            rule in random_rules]
        colors = plt.cm.Paired(range(len(random_rules)))
        short_labels = [f"{label[:20]}..." if len(label) > 20 else label for label in rules_labels]
        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(supports, labels=short_labels, autopct='%1.1f%%', startangle=90,
                                          colors=colors, wedgeprops={'edgecolor': 'black'})
        for text in texts:
            text.set_fontsize(10)
            text.set_fontweight('bold')
        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_fontweight('bold')
        plt.title(f'Случайно выбранные {len(random_rules)} правил по поддержке')
        plt.axis('equal')
        plt.show()

    @staticmethod
    def main():
        path_csv = "D:\\university\\7_semestr\\kursach\\dataset\\filtered_dataset.csv"
        output_txt = "D:\\university\\7_semestr\\kursach\\dataset\\fp_growth_results.txt"
        FP_Growth.generate_rules(input_csv=path_csv, output_path=output_txt)
