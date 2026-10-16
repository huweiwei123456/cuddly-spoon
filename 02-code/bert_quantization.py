'''
BERT 模型的动态量化（Dynamic Quantization, DQ）
说明：
    动态量化只支持 CPU，所以本脚本自包含一套 CPU 版的评估函数，
    完全不修改 bert_train 内部的 config.device，也不影响其他人训练。
'''

import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from config import Config

# ★ 只 import 需要的类/函数，不用 bert_train.evaluate
from bert_train import (
    MyBertClassifier,
    build_dataloader,
    load_raw_data,
    MyDataset,
)

# 量化脚本自己的 config，强制 CPU
config = Config()
config.device = torch.device("cpu")


# ==================== 自包含的 CPU 版评估函数 ====================
@torch.no_grad()
def evaluate_cpu(model, dataloader, loss_fn):
    """纯 CPU 评估，不依赖 bert_train 内部的 config"""
    model.eval()
    total_loss = 0.0
    total_preds, total_label = [], []

    for batch in tqdm(dataloader, desc="Evaluating"):
        # 全部强制放 CPU
        input_ids, attention_mask, labels = [x.to("cpu") for x in batch]

        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        y_pred = logits.argmax(dim=-1).cpu().numpy().tolist()
        total_preds.extend(y_pred)
        total_label.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(total_label, total_preds)
    f1 = f1_score(total_label, total_preds, average="macro")
    return avg_loss, acc, f1


# ==================== 工具函数 ====================
def get_model_size_mb(model, save_path):
    torch.save(model.state_dict(), save_path)
    return os.path.getsize(save_path) / (1024 * 1024)


def measure_latency(model, dataloader, repeat=1):
    model.eval()
    total_time = 0.0
    with torch.no_grad():
        for _ in range(repeat):
            start = time.time()
            for batch in dataloader:
                input_ids, attention_mask, _ = [x.to("cpu") for x in batch]
                _ = model(input_ids, attention_mask)
            total_time += time.time() - start
    return total_time / repeat


# ==================== 主流程 ====================
if __name__ == "__main__":
    # 1. 数据
    train_data = load_raw_data(config.train_path)
    dev_data   = load_raw_data(config.dev_path)
    test_data  = load_raw_data(config.test_path)
    print("3份样本数:", len(train_data), len(dev_data), len(test_data))

    train_dataset = MyDataset(train_data)
    dev_dataset   = MyDataset(dev_data)
    test_dataset  = MyDataset(test_data)

    train_dataloader, dev_dataloader, test_dataloader = build_dataloader(
        train_dataset, dev_dataset, test_dataset
    )

    # 2. 加载训练好的 float32 模型
    model = MyBertClassifier()
    model.load_state_dict(
        torch.load(config.bert_model_path, map_location="cpu")
    )
    model.to("cpu")
    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    # 3. 评估原始模型
    print("\n" + "=" * 60)
    print("评估原始模型（float32）")
    print("=" * 60)
    test_loss, test_acc, test_f1 = evaluate_cpu(model, test_dataloader, loss_fn)
    print(f"原始 BERT -> loss: {test_loss:.4f}  acc: {test_acc:.4f}  f1: {test_f1:.4f}")

    tmp_path = os.path.join(config.save_model_dir, "_tmp_float.pth")
    float_size = get_model_size_mb(model, tmp_path)
    float_latency = measure_latency(model, test_dataloader)
    print(f"原始模型大小: {float_size:.2f} MB")
    print(f"原始模型推理耗时: {float_latency:.3f} 秒")

    # 4. 动态量化
    print("\n" + "=" * 60)
    print("开始动态量化（Linear -> int8）")
    print("=" * 60)
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )

    # 5. 评估量化后的模型
    print("\n" + "=" * 60)
    print("评估量化后的模型（int8）")
    print("=" * 60)
    test_loss2, test_acc2, test_f12 = evaluate_cpu(quantized_model, test_dataloader, loss_fn)
    print(f"量化 BERT -> loss: {test_loss2:.4f}  acc: {test_acc2:.4f}  f1: {test_f12:.4f}")

    quant_size = get_model_size_mb(quantized_model, config.bert_quantization_model_path)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    quant_latency = measure_latency(quantized_model, test_dataloader)
    print(f"量化模型大小: {quant_size:.2f} MB")
    print(f"量化模型推理耗时: {quant_latency:.3f} 秒")

    # 6. 汇总
    print("\n" + "=" * 60)
    print("量化前后对比")
    print("=" * 60)
    print(f"{'指标':<12}{'原始模型':<15}{'量化模型':<15}{'变化':<15}")
    print(f"{'Loss':<12}{test_loss:<15.4f}{test_loss2:<15.4f}{test_loss2 - test_loss:<+15.4f}")
    print(f"{'Accuracy':<12}{test_acc:<15.4f}{test_acc2:<15.4f}{test_acc2 - test_acc:<+15.4f}")
    print(f"{'F1':<12}{test_f1:<15.4f}{test_f12:<15.4f}{test_f12 - test_f1:<+15.4f}")
    print(f"{'模型大小/MB':<12}{float_size:<15.2f}{quant_size:<15.2f}"
          f"{(quant_size - float_size) / float_size * 100:+.1f}%")
    print(f"{'推理耗时/s':<12}{float_latency:<15.3f}{quant_latency:<15.3f}"
          f"{(quant_latency - float_latency) / float_latency * 100:+.1f}%")

    # 7. 保存
    torch.save(quantized_model.state_dict(), config.bert_quantization_model_path)
    print(f"\n量化模型已保存到: {config.bert_quantization_model_path}")