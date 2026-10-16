"""
基于预训练 BERT 模型进行下游文本分类微调
（适配客服工单分类项目，8 类）
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # 关闭 TensorFlow 警告

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, BertConfig
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

from config import Config

# 创建配置对象
config = Config()

# 加载分词器与模型配置
BERT_TOKENIZER = BertTokenizer.from_pretrained(config.bert_path)
BERT_CONFIG = BertConfig.from_pretrained(config.bert_path)


# ==================== 1. 数据准备 ====================
def load_raw_data(file_path):
    """读取数据文件，返回 [(文本, 标签索引), ...]"""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text, label = line.split('\t')
            results.append((text, int(label)))
    return results


class MyDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        text, label = self.data_list[idx]
        return text, label


def collate_fn(batch):
    """批处理函数：分词、padding、转张量"""
    texts, labels = zip(*batch)

    output = BERT_TOKENIZER(
        list(texts),
        padding='max_length',
        truncation=True,
        max_length=config.max_len,
        return_tensors='pt'
    )

    input_ids = output['input_ids']
    attention_mask = output['attention_mask']
    labels = torch.tensor(labels, dtype=torch.int64)

    return input_ids, attention_mask, labels


def build_dataloader(train_dataset, dev_dataset, test_dataset):
    """构建三个数据加载器"""
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
    )

    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False   # 验证集不丢弃最后一批，保证指标完整
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False
    )

    return train_dataloader, dev_dataloader, test_dataloader


# ==================== 2. 模型定义 ====================
class MyBertClassifier(nn.Module):
    """BERT + 线性分类层"""
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.fc = nn.Linear(BERT_CONFIG.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = self.fc(output.pooler_output)
        return logits


# ==================== 3. 训练与评估 ====================
def train_one_epoch(model, train_dataloader, loss_fn, optimizer):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    total_preds = []
    total_label = []

    for batch in tqdm(train_dataloader, desc='Training'):
        input_ids, attention_mask, labels = [item.to(config.device) for item in batch]

        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        y_pred = logits.argmax(dim=-1).detach().cpu().numpy().tolist()
        total_preds.extend(y_pred)
        total_label.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(train_dataloader)
    acc = accuracy_score(total_label, total_preds)
    f1s = f1_score(total_label, total_preds, average='macro')

    return avg_loss, acc, f1s


def evaluate(model, dev_dataloader, loss_fn):
    """验证集评估"""
    model.eval()
    total_loss = 0.0
    total_preds = []
    total_label = []

    with torch.no_grad():
        for batch in tqdm(dev_dataloader, desc='Evaluating'):
            input_ids, attention_mask, labels = [item.to(config.device) for item in batch]

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            total_loss += loss.item()

            y_pred = logits.argmax(dim=-1).detach().cpu().numpy().tolist()
            total_preds.extend(y_pred)
            total_label.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(dev_dataloader)
    acc = accuracy_score(total_label, total_preds)
    f1s = f1_score(total_label, total_preds, average='macro')

    return avg_loss, acc, f1s


def train(train_dataloader, dev_dataloader, model):
    """完整训练流程"""
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_f1 = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []

    for epoch in range(config.epochs):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_dataloader, loss_fn, optimizer
        )
        val_loss, val_acc, val_f1 = evaluate(
            model, dev_dataloader, loss_fn
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), config.bert_model_path)

        print(f'epoch: {epoch}  trainloss={train_loss:.4f}  trainAcc={train_acc:.4f}  '
              f'trainF1={train_f1:.4f}  valloss={val_loss:.4f}  valAcc={val_acc:.4f}  valF1={val_f1:.4f}')

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_f1s.append(train_f1)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

    results = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'train_f1s': train_f1s,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'val_f1s': val_f1s
    }
    return results


# ==================== 4. 绘图 ====================
def plot_history(history):
    epochs = range(1, len(history['train_losses']) + 1)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_losses'], label='train_loss')
    plt.plot(epochs, history['val_losses'], label='val_loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_accs'], label='train_acc')
    plt.plot(epochs, history['val_accs'], label='val_acc')
    plt.title('Acc Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['train_f1s'], label='train_f1')
    plt.plot(epochs, history['val_f1s'], label='val_f1')
    plt.title('F1 Curve')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


# ==================== 5. 主程序 ====================
if __name__ == '__main__':
    train_data = load_raw_data(config.train_path)
    dev_data = load_raw_data(config.dev_path)
    test_data = load_raw_data(config.test_path)

    print('3份样本数:', len(train_data), len(dev_data), len(test_data))

    train_dataset = MyDataset(train_data)
    dev_dataset = MyDataset(dev_data)
    test_dataset = MyDataset(test_data)

    train_dataloader, dev_dataloader, test_dataloader = build_dataloader(
        train_dataset, dev_dataset, test_dataset
    )

    model = MyBertClassifier().to(config.device)

    results = train(train_dataloader, dev_dataloader, model)

    plot_history(results)