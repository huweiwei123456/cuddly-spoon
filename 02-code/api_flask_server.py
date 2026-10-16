import os
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from bert_predict_fun import predict_fun
from config import Config

load_dotenv()
config = Config()
app = Flask(__name__)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

LABELS = list(config.class2id.keys())


# ---------- 路由 1：本地 BERT ----------
@app.route('/predict', methods=['POST'])
def predict():
    return jsonify(predict_fun(request.get_json()))


# ---------- 路由 2：DeepSeek 直接分类 ----------
@app.route('/predict_llm', methods=['POST'])
def predict_llm():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({"error": "text 不能为空"}), 400

    system_prompt = (
        "你是一个客服工单分类助手。"
        f"请从以下类别中选择最合适的一个：{ '、'.join(LABELS) }。"
        "只输出类别名称，不要解释，不要标点，不要多余文字。"
    )

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        "temperature": 0.0,
        "max_tokens": 16,
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()

        pred_class = content.replace("。", "").replace("\n", "").strip()
        if pred_class not in config.class2id:
            for label in LABELS:
                if label in pred_class:
                    pred_class = label
                    break
            else:
                pred_class = "未识别"

        return jsonify({"text": text, "pred_class": pred_class})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------- 路由 3：本地分类 + DeepSeek 生成客服回复 ----------
@app.route('/predict_with_reply', methods=['POST'])
def predict_with_reply():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({"error": "text 不能为空"}), 400

    bert_result = predict_fun({'text': text})
    pred_class = bert_result['pred_class']
    confidence = bert_result.get('confidence', 0.0)

    system_prompt = (
        "你是一位电商客服人员。"
        f"用户提交了一条工单，系统已将其归类为「{pred_class}」。"
        "请根据用户原文，生成一段 50 字以内、态度友好、专业简洁的客服回复。"
        "不要重复类别名称，不要加任何前缀，只输出回复内容本身。"
    )

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        "temperature": 0.7,
        "max_tokens": 120,
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        reply = ""
        print("DeepSeek 生成失败：", e)

    return jsonify({
        "text": text,
        "pred_class": pred_class,
        "confidence": confidence,
        "reply": reply,
    })


# ---------- ★ 主入口：启动 Flask（之前缺的就是这段）----------
if __name__ == '__main__':
    app.run(host=config.api_host, port=config.api_port, debug=True)