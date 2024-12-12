# import pandas as pd
# import numpy as np
# import random
# import itertools
# import matplotlib.pyplot as plt
# import seaborn as sns
# import networkx as nx
#
# np.random.seed(1)
#
# class EclatAlgorithm:
#     @staticmethod
#     def Read_Data(filename):
#         df = pd.read_csv(filename, header=None)
#         data = {}
#         trans = 0
#
#         for _, row in df.iterrows():
#             trans += 1
#             for item in row.dropna():
#                 if item not in data:
#                     data[item] = set()
#                 data[item].add(trans)
#         return data
#
#     @staticmethod
#     def eclat(prefix, items, minsup, dict_id, FreqItems):
#         while items:
#             i, itids = items.pop()
#             isupp = len(itids)
#             if isupp >= minsup:
#                 FreqItems[frozenset(prefix + [i])] = isupp
#                 suffix = []
#                 for j, ojtids in items:
#                     jtids = itids & ojtids
#                     if len(jtids) >= minsup:
#                         suffix.append((j, jtids))
#                 dict_id += 1
#                 EclatAlgorithm.eclat(prefix + [i], sorted(suffix, key=lambda item: len(item[1]), reverse=True), minsup,
#                                      dict_id, FreqItems)
#
#     @staticmethod
#     def rules(FreqItems, confidence, min_lift=1.5):
#         Rules = []
#         cnt = 0
#
#         for items, support in FreqItems.items():
#             if len(items) > 3:
#                 all_perms = list(itertools.permutations(items, len(items)))
#                 for lst in all_perms:
#                     antecedent = lst[:len(lst) - 1]
#                     consequent = lst[-1:]
#
#                     conf = float(FreqItems[frozenset(items)] / FreqItems[frozenset(antecedent)] * 100)
#                     if conf >= confidence:
#                         cnt += 1
#                         lift = float(conf / FreqItems[frozenset(consequent)])
#                         if lift >= min_lift:
#                             Rules.append((antecedent, consequent, support, conf, lift))
#         return Rules
#
#     @staticmethod
#     def save_Rules_to_file(Rules, output_file_path):
#         with open(output_file_path, mode='w', encoding='utf-8') as file:
#             for a, b, supp, conf, lift in sorted(Rules):
#                 file.write(f"Правило: {a} -> {b}\n")
#                 file.write(f"  Поддержка: {round(supp, 2)}\n")
#                 file.write(f"  Достоверность: {round(conf, 2)}\n")
#                 file.write(f"  Лифт: {round(lift, 2)}\n")
#                 file.write("\n")
#     @staticmethod
#     def support_graphic(Rules):
#         supports = [rule[2] for rule in Rules]
#         plt.figure(figsize=(12, 8))
#         plt.hist(supports, bins=30, edgecolor='black', color='skyblue', alpha=0.7)
#         plt.xlabel("Поддержка", fontsize=14)
#         plt.ylabel("Частота", fontsize=14)
#         plt.title("Распределение значений поддержки", fontsize=16, fontweight='bold')
#         plt.grid(axis='y', linestyle='--', alpha=0.7)
#         plt.tight_layout()
#         plt.show()
#     @staticmethod
#     def confidence_graphic(Rules):
#         confidences = [rule[3] for rule in Rules]
#         plt.figure(figsize=(12, 8))
#         plt.hist(confidences, bins=15, edgecolor='black', color='lightgreen', alpha=0.7)
#         plt.xlabel("Уверенность", fontsize=14)
#         plt.ylabel("Частота", fontsize=14)
#         plt.title("Распределение значений уверенности", fontsize=16, fontweight='bold')
#         plt.grid(axis='y', linestyle='--', alpha=0.7)
#         plt.tight_layout()
#         plt.show()
#     @staticmethod
#     def lift_graphic(Rules):
#         lifts = [rule[4] for rule in Rules]
#         plt.figure(figsize=(12, 8))
#         plt.hist(lifts, bins=15, edgecolor='black', color='skyblue', alpha=0.7)
#         plt.xlabel("Лифт", fontsize=14)
#         plt.ylabel("Частота", fontsize=14)
#         plt.title("Распределение значений лифта", fontsize=16, fontweight='bold')
#         plt.grid(axis='y', linestyle='--', alpha=0.7)
#         plt.tight_layout()
#         plt.show()
#     @staticmethod
#     def rule_matrix(Rules, top_n=20):
#         random_rules = random.sample(Rules, min(top_n, len(Rules)))
#         antecedents = [', '.join(list(rule[0])) for rule in random_rules]
#         consequents = [', '.join(list(rule[1])) for rule in random_rules]
#         supports = [rule[2] for rule in random_rules]
#         data = {"antecedent": antecedents, "consequent": consequents, "support": supports}
#         df = pd.DataFrame(data)
#         pivot_table = df.pivot_table(index="antecedent", columns="consequent", values="support", aggfunc="sum")
#         plt.figure(figsize=(12, 10))
#         sns.set(font_scale=0.8)
#         ax = sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".2f", xticklabels=True, yticklabels=True,
#                          cbar_kws={"shrink": 0.8})
#         plt.title(f"Матрица правил для случайных {min(top_n, len(Rules))} правил", fontsize=16, fontweight='bold')
#         plt.xlabel("Консеквент")
#         plt.ylabel("Антецедент")
#         ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
#         ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
#         plt.tight_layout()
#         plt.show()
#     @staticmethod
#     def circle_graphic(Rules, top_n=10):
#         random_rules = random.sample(Rules,min(top_n, len(Rules)))
#         supports = [rule[2] for rule in random_rules]
#         rules_labels = [f"{', '.join(list(rule[0]))} -> {', '.join(list(rule[1]))}" for rule in random_rules]
#         plt.pie(supports, labels=rules_labels, autopct='%1.1f%%', startangle=90)
#         plt.title(f'Случайно выбранные {len(random_rules)} правил по поддержке')
#         plt.axis('equal')
#         plt.tight_layout()
#         plt.show()
#
#     @staticmethod
#     def main():
#         minsup = 5
#         confidence = 80
#         min_lift = 1.5
#         data = EclatAlgorithm.Read_Data("D:\\university\\7_semestr\\kursach\\dataset\\filtered_dataset.csv")
#         FreqItems = {}
#         EclatAlgorithm.eclat([], sorted(data.items(), key=lambda item: len(item[1]), reverse=True), minsup, 0,
#                              FreqItems)
#         Rules = EclatAlgorithm.rules(FreqItems, confidence, min_lift)  # передаем min_lift
#         output_file_path = "D:\\university\\7_semestr\\kursach\\dataset\\eclat_results.txt"
#         EclatAlgorithm.save_Rules_to_file(Rules, output_file_path)
#
#         EclatAlgorithm.support_graphic(Rules)
#         EclatAlgorithm.confidence_graphic(Rules)
#         EclatAlgorithm.lift_graphic(Rules)
#         EclatAlgorithm.rule_matrix(Rules)
#         EclatAlgorithm.circle_graphic(Rules)
import pandas as pd
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil

np.random.seed(1)

class EclatAlgorithm:
    @staticmethod
    def Read_Data(filename):
        df = pd.read_csv(filename, header=None)
        data = {}
        trans = 0

        for _, row in df.iterrows():
            trans += 1
            for item in row.dropna():
                if item not in data:
                    data[item] = set()
                data[item].add(trans)
        return data

    @staticmethod
    def eclat(prefix, items, minsup, dict_id, FreqItems):
        while items:
            i, itids = items.pop()
            isupp = len(itids)
            if isupp >= minsup:
                FreqItems[frozenset(prefix + [i])] = isupp
                suffix = []
                for j, ojtids in items:
                    jtids = itids & ojtids
                    if len(jtids) >= minsup:
                        suffix.append((j, jtids))
                dict_id += 1
                EclatAlgorithm.eclat(prefix + [i], sorted(suffix, key=lambda item: len(item[1]), reverse=True), minsup,
                                     dict_id, FreqItems)

    @staticmethod
    def calculate_metrics(FreqItems, total_transactions, min_confidence, min_lift):
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / (1024 * 1024)  # В MB
        Rules = []
        for items, support_count in FreqItems.items():
            support = support_count / total_transactions
            if len(items) > 1:
                for antecedent_len in range(1, len(items)):
                    antecedent_candidates = itertools.combinations(items, antecedent_len)
                    for antecedent in antecedent_candidates:
                        antecedent = frozenset(antecedent)
                        consequent = items - antecedent
                        if antecedent in FreqItems:
                            confidence = support_count / FreqItems[antecedent]
                            lift = confidence / (FreqItems[consequent] / total_transactions)
                            if confidence >= min_confidence and lift >= min_lift:
                                Rules.append({
                                    "antecedent": antecedent,
                                    "consequent": consequent,
                                    "support": support,
                                    "confidence": confidence,
                                    "lift": lift
                                })
        memory_after = psutil.Process().memory_info().rss / (1024 * 1024)
        end_time = time.time()
        execution_time = end_time - start_time
        memory_usage = memory_after - memory_before
        num_rules = len(Rules)
        avg_confidence = sum(rule['confidence'] for rule in Rules) / num_rules if num_rules > 0 else 0
        avg_lift = sum(rule['lift'] for rule in Rules) / num_rules if num_rules > 0 else 0
        print('Eclat')
        print(f"Время выполнения: {execution_time:.2f} секунд")
        print(f"Использование памяти: {memory_usage:.2f} MB")
        print(f"Количество правил: {num_rules}")
        print(f"Средняя уверенность: {avg_confidence:.2f}")
        print(f"Средний лифт: {avg_lift:.2f}")
        return Rules

    @staticmethod
    def save_Rules_to_file(Rules, output_file_path):
        with open(output_file_path, mode='w', encoding='utf-8') as file:
            for rule in Rules:
                antecedent = ', '.join(rule['antecedent'])
                consequent = ', '.join(rule['consequent'])
                support = rule['support']
                confidence = rule['confidence']
                lift = rule['lift']

                file.write(f"Правило: {antecedent} -> {consequent}\n")
                file.write(f"  Поддержка: {support:.2f}\n")
                file.write(f"  Уверенность: {confidence:.2f}\n")
                file.write(f"  Лифт: {lift:.2f}\n")
                file.write("\n")

    @staticmethod
    def support_graphic(Rules):
        supports = [rule['support'] for rule in Rules]
        unique_supports = len(set(supports))
        mean_support = np.mean(supports)
        median_support = np.median(supports)
        plt.figure(figsize=(12, 8))
        counts, bins, _ = plt.hist(supports, bins=unique_supports, edgecolor='black', color='skyblue', alpha=0.7, label='Поддержка')
        for count, x in zip(counts, bins[:-1]):
            plt.text(x + (bins[1] - bins[0]) / 2, count + 0.1, str(int(count)), ha='center', fontsize=10)
        plt.axvline(mean_support, color='red', linestyle='--', linewidth=2, label=f'Среднее: {mean_support:.2f}')
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
        confidences = [rule['confidence'] for rule in Rules]
        mean_confidence = np.mean(confidences)
        median_confidence = np.median(confidences)
        plt.figure(figsize=(12, 8))
        counts, bins, _ = plt.hist(confidences, bins='auto', edgecolor='black', color='skyblue', alpha=0.7, label='Уверенность')
        for count, x in zip(counts, bins[:-1]):
            plt.text(x + (bins[1] - bins[0]) / 2, count + 0.1, str(int(count)), ha='center', fontsize=10)
        plt.axvline(mean_confidence, color='red', linestyle='--', linewidth=2, label=f'Среднее: {mean_confidence:.2f}')
        plt.xlabel("Уверенность", fontsize=14)
        plt.ylabel("Частота", fontsize=14)
        plt.title("Распределение значений уверенности", fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def lift_graphic(Rules):
        lifts = [rule['lift'] for rule in Rules]
        mean_lift = np.mean(lifts)
        median_lift = np.median(lifts)
        plt.figure(figsize=(12, 8))
        counts, bins, _ = plt.hist(lifts, bins='auto', edgecolor='black', color='skyblue', alpha=0.7, label='Лифт')
        for count, x in zip(counts, bins[:-1]):
            plt.text(x + (bins[1] - bins[0]) / 2, count + 0.1, str(int(count)), ha='center', fontsize=10)
        plt.axvline(mean_lift, color='red', linestyle='--', linewidth=2, label=f'Среднее: {mean_lift:.2f}')
        plt.xlabel("Лифт", fontsize=14)
        plt.ylabel("Частота", fontsize=14)
        plt.title("Распределение значений лифта", fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    @staticmethod
    def rule_matrix(Rules, top_n=20):
        random_rules = random.sample(Rules, min(top_n, len(Rules)))
        antecedents = [', '.join(list(rule['antecedent'])) for rule in random_rules]
        consequents = [', '.join(list(rule['consequent'])) for rule in random_rules]
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
        random_rules = random.sample(Rules, min(top_n, len(Rules)))
        supports = [rule['support'] for rule in random_rules]
        rules_labels = [f"{', '.join(list(rule['antecedent']))} -> {', '.join(list(rule['consequent']))}" for rule in
                        random_rules]
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
        plt.tight_layout()
        plt.show()

    @staticmethod
    def main():
        min_support = 0.2
        min_confidence = 0.3
        min_lift = 1.0
        data = EclatAlgorithm.Read_Data("D:\\university\\7_semestr\\kursach\\dataset\\filtered_dataset.csv")
        total_transactions = len(set(itertools.chain.from_iterable(data.values())))
        min_support_count = int(min_support * total_transactions)
        FreqItems = {}
        EclatAlgorithm.eclat([], sorted(data.items(), key=lambda item: len(item[1]), reverse=True), min_support_count, 0,
                             FreqItems)
        Rules = EclatAlgorithm.calculate_metrics(FreqItems, total_transactions, min_confidence, min_lift)
        output_file_path = "D:\\university\\7_semestr\\kursach\\dataset\\eclat_results.txt"
        EclatAlgorithm.save_Rules_to_file(Rules, output_file_path)
        EclatAlgorithm.support_graphic(Rules)
        EclatAlgorithm.confidence_graphic(Rules)
        EclatAlgorithm.lift_graphic(Rules)
        EclatAlgorithm.rule_matrix(Rules)
        EclatAlgorithm.circle_graphic(Rules)