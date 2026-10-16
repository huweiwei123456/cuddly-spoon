import time
import torch
import torch.nn as nn
from config import Config
from bert_train import MyBertClassifier
from transformers import BertTokenizer

config = Config()

# ---------- 加载量化模型（关键） ----------
base_model = MyBertClassifier()                       # 普通结构
quantized_model = torch.quantization.quantize_dynamic(
    base_model, {nn.Linear}, dtype=torch.qint8
)                                                     # 量化结构骨架
quantized_model.load_state_dict(
    torch.load(config.bert_quantization_model_path, map_location="cpu")
)
quantized_model.to("cpu")
quantized_model.eval()

BERT_TOKENIZER = BertTokenizer.from_pretrained(config.bert_path)


@torch.no_grad()
def predict_fun(data):
    text = data['text']
    output = BERT_TOKENIZER(
        [text],
        padding='max_length',
        truncation=True,
        max_length=config.max_len,
        return_tensors='pt'
    )

    # 量化模型强制 CPU，所以这里 .to("cpu")
    logits = quantized_model(
        input_ids=output['input_ids'].to("cpu"),
        attention_mask=output['attention_mask'].to("cpu"),
    )

    y_pred = torch.argmax(logits, dim=-1).item()
    y_pred_class = config.id2class[y_pred]

    probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
    data['pred_class'] = y_pred_class
    data['probs'] = probs
    data['confidence'] = probs[y_pred]
    return data


if __name__ == '__main__':
    data = {'text': '我的快递好几天没更新了，什么时候能送到？'}
    result = predict_fun(data)
    print(result['pred_class'], result['confidence'])

    start = time.time()
    for _ in range(100):
        predict_fun(data)
    end = time.time()
    print('单次推理时间：', (end - start) / 100 * 1000, 'ms')