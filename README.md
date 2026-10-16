# 客服工单智能分类系统

基于 **BERT 中文预训练模型微调 + 动态量化 + DeepSeek 大模型** 的工单智能处理系统，覆盖「**数据 → 训练 → 评估 → 量化 → 部署 → 前端**」全链路闭环。

支持三种预测模式：本地量化 BERT 快速分类、DeepSeek 大模型分类、本地分类 + DeepSeek 生成客服回复。

---

## 项目结构

data/                                   ← 项目根目录
│
├── 01-data/                            ← 数据与预训练模型
│   ├── bert-base-chinese/              ← 预训练中文 BERT（底座）
│   │   ├── config.json
│   │   ├── configuration.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── vocab.txt
│   │   └── .gitattributes
│   ├── save_model/                     ← 训练产物
│   │   ├── bert_model.pth              ← 微调后的 BERT（教师/主模型）
│   │   └── bert_quantization.pth       ← 动态量化后的模型
│   ├── class.txt                       ← 8 个类别名称，一行一个
│   ├── train.txt                       ← 训练集
│   ├── dev.txt                         ← 验证集
│   ├── test.txt                        ← 测试集
│   ├── train_raw.txt                   ← （可选）去重前原始数据
│   ├── dev_raw.txt
│   └── test_raw.txt
│
└── 02-code/                            ← 所有代码
    ├── .env                            ← 存 DEEPSEEK_API_KEY（不要提交 Git）
    ├── .gitignore                      ← 排除 .env、__pycache__ 等
    ├── requirements.txt                ← 依赖清单
    ├── config.py                       ← 全局配置：路径、超参、device、API 端口
    ├── 数据探索.py                      ← 查看数据分布
    ├── bert_train.py                   ← 微调 BERT（产出 bert_model.pth）
    ├── bert_quantization.py            ← 动态量化（产出 bert_quantization.pth）
    ├── bert_predict_fun.py             ← 加载量化模型做本地预测
    ├── api_flask_server.py             ← Flask 后端（3 个路由）
    └── streamlit_app.py                ← Streamlit 前端界面
---

## 快速开始

### 1. 环境准备

```bash
# 建议 Python 3.10
conda create -n nlp python=3.10
conda activate nlp
pip install -r requirements.txt
```

`requirements.txt` 关键依赖：

torch
transformers
scikit-learn
pandas
flask
streamlit
requests
tqdm
matplotlib
altair
python-dotenv

### 2. 下载预训练模型

将 `bert-base-chinese` 完整目录（含 `config.json`、`vocab.txt`、`model.safetensors` 等）放到：

01-data/bert-base-chinese/

### 3. 配置 API Key（使用 DeepSeek 时必需）

在 `02-code/` 下创建 `.env`：

ini

```
DEEPSEEK_API_KEY=sk-你的真实key
```

### 4. 训练模型

bash

```
cd 02-code
python bert_train.py
```



- 自动读取 `01-data/train.txt`、`dev.txt`、`test.txt`
- 逐轮打印 loss、acc、F1
- 按验证集 F1 保存最优权重到 `01-data/save_model/bert_model.pth`
- 训练结束后弹出 Loss / Acc / F1 三张曲线图

### 5. 动态量化

bash

```
python bert_quantization.py
```



- 强制在 CPU 上执行（动态量化不支持 GPU）
- 对比量化前后的模型体积、推理耗时、精度
- 保存量化模型到 `01-data/save_model/bert_quantization.pth`

### 6. 启动后端

bash

```
python api_flask_server.py
```



服务监听 `http://127.0.0.1:8888`，提供三个接口：

| 接口                  | 方法 | 说明                             |
| :-------------------- | :--- | :------------------------------- |
| `/predict`            | POST | 本地量化 BERT 分类               |
| `/predict_llm`        | POST | DeepSeek 大模型分类              |
| `/predict_with_reply` | POST | 本地分类 + DeepSeek 生成客服回复 |

### 7. 启动前端

另开一个终端（后端保持运行）：

bash

```
streamlit run streamlit_app.py
```



浏览器自动打开 `http://localhost:8501`，界面提供三种预测模式。

## 技术架构

                          ┌─────────────────────┐
                          │   Streamlit 前端      │
                          │  三种预测模式可选       │
                          └──────────┬──────────┘
                                     │ HTTP
                                     ▼
                          ┌─────────────────────┐
                          │   Flask 后端          │
                          │  /predict            │
                          │  /predict_llm        │
                          │  /predict_with_reply │
                          └──────┬──────┬───────┘
                                 │      │
              ┌──────────────────┘      └──────────────────┐
              ▼                                            ▼
    ┌────────────────────┐                     ┌────────────────────┐
    │  量化 BERT（int8）    │                     │  DeepSeek 大模型      │
    │  本地 CPU 推理        │                     │  API 调用            │
    │  快速分类             │                     │  生成客服回复          │
    └────────────────────┘                     └────────────────────┘

## 三层能力

| 层次   | 任务             | 实现                              |
| :----- | :--------------- | :-------------------------------- |
| 第一层 | 工单分类（8 类） | 微调 BERT + 线性分类层            |
| 第二层 | 模型压缩与加速   | PyTorch 动态量化（Linear → int8） |
| 第三层 | 智能客服回复     | DeepSeek 基于分类结果生成话术     |

## 数据集

- **数据来源**：模拟真实客服工单生成
- **规模**：训练 14457 / 验证 3098 / 测试 3098
- **类别**：8 类均衡分布（见 `class.txt`）
- **格式**：每行 `文本\t标签索引`

| 索引 | 类别     |
| :--- | :------- |
| 0    | 订单查询 |
| 1    | 退款退货 |
| 2    | 商品咨询 |
| 3    | 账户问题 |
| 4    | 投诉建议 |
| 5    | 支付问题 |
| 6    | 发票问题 |
| 7    | 售后维修 |

------

## 核心指标

| 模型                  | 模型大小 | CPU 单次耗时 | Accuracy  | Macro-F1  |
| :-------------------- | :------- | :----------- | :-------- | :-------- |
| BERT 微调（float32）  | ~412 MB  | ~30 ms       | 91%+      | 90%+      |
| BERT 动态量化（int8） | ~100 MB  | ~30 ms       | 略降 1~3% | 略降 1~3% |

> 具体数字以你本机运行 `bert_quantization.py` 输出的对比表为准。

------

## 项目亮点

1. **完整 NLP 落地闭环**：数据清洗 → 模型微调 → 指标评估 → 动态量化 → API 部署 → 可视化前端
2. **模型压缩**：动态量化将模型体积压缩至原始的 1/3 ~ 1/4，保留精度 97%+
3. **小模型 + 大模型协作**：本地 BERT 负责低成本分类，DeepSeek 负责高质量话术生成，兼顾速度、成本、可读性
4. **工程规范**：`.env` 隔离敏感信息、`.gitignore` 防泄露、`config.py` 统一管理路径与超参
5. **可视化前端**：类别徽章、概率分布图、示例工单一键填入、三种模式实时切换

------

## 常见问题

**Q1：`from config import Config` 报 ModuleNotFoundError**
把 `02-code` 目录在 IDE 里标记为 **Sources Root**（PyCharm 右键 → Mark Directory as → Sources Root）。

**Q2：训练时显存不足**
把 `config.py` 里的 `batch_size` 从 32 调到 16 或 8。

**Q3：量化时提示 `RuntimeError: Expected all tensors to be on the same device`**
量化只能在 CPU 上做。`bert_quantization.py` 里已强制 `config.device = cpu`，若仍报错，检查是否混用了 `bert_train` 内部的 `config`。

**Q4：DeepSeek 调用 401**
检查 `.env` 中 `DEEPSEEK_API_KEY` 是否填对，以及 `api_flask_server.py` 顶部是否加了 `load_dotenv()`。

**Q5：`bert-base-chinese` 加载报 `UNEXPECTED`**
这是 BERT 预训练自带的 MLM/NSP 头被忽略，属于正常现象，不影响分类。

## 启动顺序一览

bash

```
# 1. 训练（如已有 bert_model.pth 可跳过）
cd 02-code && python bert_train.py

# 2. 量化（如已有 bert_quantization.pth 可跳过）
python bert_quantization.py

# 3. 后端（保持终端不关）
python api_flask_server.py

# 4. 前端（新开终端）
streamlit run streamlit_app.py
```
