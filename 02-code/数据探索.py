import pandas as pd
from collections import Counter
from config import Config


def load_data(path):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            texts.append(parts[0])
            labels.append(int(parts[1]))
    return texts, labels


config = Config()

train_texts, train_labels = load_data(config.train_path)
dev_texts,   dev_labels   = load_data(config.dev_path)     # ← 用 config
test_texts,  test_labels  = load_data(config.test_path)    # ← 用 config

print("train:", len(train_texts))
print("dev:",   len(dev_texts))
print("test:",  len(test_texts))

print("\n训练集标签分布:")
print(Counter(train_labels))

print("\n各类别样本数:")
for i, name in enumerate(config.class_list):
    print(f"{i} {name}: {train_labels.count(i)}")