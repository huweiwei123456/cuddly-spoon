"""
    配置文件：客服工单分类项目（BERT 微调）
    统一管理全局配置信息，包括数据路径、类别信息、BERT 模型路径等。
    数据格式：每行 文本\t标签，标签 0~7 对应 class.txt 中的 8 个类别。
"""
import os
from pathlib import Path


# ============ 1. 配置类 ============
class Config:
    def __init__(self):
        # ---------- 1.1 根路径 ----------
        # 当前文件所在目录的上一级下的 01-data 目录
        self.root_path = os.path.join(Path(__file__).parent.parent, "01-data")

        # ---------- 1.2 原始数据（去重前） ----------
        self.train_raw_path = os.path.join(self.root_path, "train_raw.txt")
        self.dev_raw_path   = os.path.join(self.root_path, "dev_raw.txt")
        self.test_raw_path  = os.path.join(self.root_path, "test_raw.txt")

        # ---------- 1.3 去重后数据（训练实际使用） ----------
        self.train_path = os.path.join(self.root_path, "train_raw.txt")
        self.dev_path   = os.path.join(self.root_path, "dev_raw.txt")
        self.test_path  = os.path.join(self.root_path, "test_raw.txt")

        # ---------- 1.4 类别文件 ----------
        self.class_path = os.path.join(self.root_path, "class.txt")

        # ---------- 1.5 停用词文件（可选，BERT 一般不用） ----------
        self.stopwords_path = os.path.join(self.root_path, "stopwords.txt")

        # ---------- 1.6 模型保存目录 ----------
        self.save_model_dir = os.path.join(self.root_path, "save_model")
        os.makedirs(self.save_model_dir, exist_ok=True)

        # ---------- 1.7 类别映射 ----------
        with open(self.class_path, "r", encoding="utf-8") as f:
            self.class_list = [line.strip() for line in f if line.strip()]

        self.num_labels = len(self.class_list)                        # 类别数量（本数据为 8）
        self.id2class   = {i: name for i, name in enumerate(self.class_list)}
        self.class2id   = {name: i for i, name in enumerate(self.class_list)}

        # ---------- 1.8 API 服务（部署时用） ----------
        self.api_host = "127.0.0.1"
        self.api_port = 8888

        # ---------- 1.9 BERT 相关 ----------
        # 中文 BERT 模型本地路径（提前下载好放在 01-data/bert-base-chinese）
        # 也可以换成 hfl/chinese-roberta-wwm-ext
        self.bert_path = os.path.join(self.root_path, "bert-base-chinese")

        # 微调后的模型保存路径
        self.bert_model_path = os.path.join(self.save_model_dir, "bert_model.pth")

        # 动态量化后的模型保存路径（bert_quantization.py 使用）
        self.bert_quantization_model_path = os.path.join(
            self.save_model_dir, "bert_quantization.pth"
        )

        # ---------- 1.10 硬件选择：优先 GPU ----------
        import torch
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():   # 苹果 M 系列芯片
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # ---------- 1.11 BERT 微调超参数 ----------
        self.epochs       = 5        # 训练轮数
        self.batch_size   = 16       # 批次大小（显存够可以调到 32/64）
        self.lr           = 2e-5     # BERT 微调学习率（建议 2e-5 ~ 5e-5）
        self.max_len      = 64       # 最大序列长度（客服工单短句，64 足够）
        self.weight_decay = 1e-2     # AdamW 权重衰减系数

        # ---------- 1.12 其他训练参数 ----------
        self.seed        = 42        # 随机种子
        self.num_workers = 0         # DataLoader 线程数（Windows 建议 0）


# ============ 2. 去重函数 ============
def remove_duplicates(input_file, output_file):
    """
    读取文件按整行去重并保存到新文件。
    数据格式：每行 文本\t标签
    """
    seen = set()          # 已出现的行
    total_lines = 0       # 原样本总数
    unique_lines = 0      # 去重后样本数

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:

        for line in infile:
            total_lines += 1
            # 统一去掉行尾 \r\n 再比较，避免因换行符不同导致漏去重
            line_key = line.rstrip("\r\n")
            if not line_key:
                continue
            if line_key not in seen:
                outfile.write(line_key + "\n")
                seen.add(line_key)
                unique_lines += 1

    print(f"[{os.path.basename(input_file)}]")
    print(f"  去重前总行数: {total_lines}")
    print(f"  去重后保留行数: {unique_lines}")
    print(f"  重复行数: {total_lines - unique_lines}")
    print("-" * 30)


# ============ 3. 简单校验 ============
def check_data_file(path, num_labels=8):
    """检查数据文件格式：每行两列，标签在 [0, num_labels-1]"""
    if not os.path.exists(path):
        print(f"[{os.path.basename(path)}] 文件不存在，已跳过")
        return
    bad_lines = 0
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            parts = line.split("\t")
            if len(parts) != 2:
                bad_lines += 1
                continue
            try:
                label = int(parts[1])
                if label < 0 or label >= num_labels:
                    bad_lines += 1
            except ValueError:
                bad_lines += 1
    print(f"[{os.path.basename(path)}] 总行数: {total}, 格式异常行数: {bad_lines}")


# ============ 4. 主入口测试 ============
if __name__ == "__main__":
    config = Config()

    print("=" * 45)
    print("根路径          :", config.root_path)
    print("训练数据路径    :", config.train_path)
    print("验证数据路径    :", config.dev_path)
    print("测试数据路径    :", config.test_path)
    print("类别文件路径    :", config.class_path)
    print("模型保存目录    :", config.save_model_dir)
    print("BERT 模型路径   :", config.bert_path)
    print("BERT 保存路径   :", config.bert_model_path)
    print("量化模型路径    :", config.bert_quantization_model_path)
    print("使用设备        :", config.device)
    print("类别数量        :", config.num_labels)
    print("id -> 类别      :", config.id2class)
    print("=" * 45)

    # ---------- 校验去重后的正式数据（训练真正使用的） ----------
    check_data_file(config.train_path, config.num_labels)
    check_data_file(config.dev_path,   config.num_labels)
    check_data_file(config.test_path,  config.num_labels)

    # ---------- 如果你有 raw 文件，再取消下面 6 行的注释 ----------
    # check_data_file(config.train_raw_path, config.num_labels)
    # check_data_file(config.dev_raw_path,   config.num_labels)
    # check_data_file(config.test_raw_path,  config.num_labels)
    # remove_duplicates(config.train_raw_path, config.train_path)
    # remove_duplicates(config.dev_raw_path,   config.dev_path)
    # remove_duplicates(config.test_raw_path,  config.test_path)