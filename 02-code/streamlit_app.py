import time
import requests
import streamlit as st
import pandas as pd
import altair as alt

from config import Config

config = Config()

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="投满分 · 工单智能分类系统",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== 全局样式 ====================
st.markdown("""
<style>
/* 全局字体 */
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: "Microsoft YaHei", "PingFang SC", "Segoe UI", sans-serif;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #f5f8fc 0%, #eef2f9 100%);
}

/* 顶栏 */
.app-header {
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 60%, #3b82f6 100%);
    border-radius: 14px;
    padding: 28px 36px;
    color: #fff;
    box-shadow: 0 6px 24px rgba(30, 58, 138, 0.25);
    margin-bottom: 22px;
}
.app-header h1 {
    margin: 0;
    font-size: 28px;
    font-weight: 700;
    letter-spacing: 1px;
}
.app-header p {
    margin: 8px 0 0;
    font-size: 14px;
    color: #c7d9f5;
}
.app-header .badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.18);
    border: 1px solid rgba(255, 255, 255, 0.35);
    border-radius: 20px;
    padding: 3px 12px;
    margin-left: 10px;
    font-size: 12px;
    vertical-align: middle;
}

/* 卡片 */
.card {
    background: #fff;
    border-radius: 12px;
    padding: 22px 26px;
    box-shadow: 0 2px 12px rgba(15, 40, 70, 0.06);
    border: 1px solid #e8eef6;
    margin-bottom: 18px;
}
.card-title {
    font-size: 15px;
    font-weight: 600;
    color: #1e3a8a;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #eef2f7;
}

/* 结果徽章 */
.result-badge {
    display: inline-block;
    padding: 10px 26px;
    border-radius: 26px;
    color: #fff;
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 1px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
}
.metric-row { margin-top: 16px; }
.metric-item {
    display: inline-block;
    background: #f0f4fa;
    border-radius: 8px;
    padding: 8px 18px;
    margin-right: 12px;
    font-size: 13px;
    color: #334155;
}
.metric-item b { color: #1e3a8a; font-size: 16px; }

/* 客服回复框 */
.reply-box {
    background: linear-gradient(135deg, #eef6ff 0%, #e0ecff 100%);
    border-left: 4px solid #2563eb;
    border-radius: 10px;
    padding: 18px 22px;
    font-size: 15px;
    color: #1e293b;
    line-height: 1.9;
}

/* 侧边栏标题 */
.side-title {
    font-size: 13px;
    font-weight: 700;
    color: #1e3a8a;
    margin: 12px 0 6px;
    letter-spacing: 0.5px;
}

/* 页脚 */
.app-footer {
    text-align: center;
    color: #94a3b8;
    font-size: 12px;
    margin-top: 36px;
    padding-top: 16px;
    border-top: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# ==================== 顶栏 ====================
st.markdown("""
<div class="app-header">
    <h1>投满分 · 工单智能分类系统 <span class="badge">BERT + DeepSeek</span></h1>
    <p>本地量化 BERT 快速分类 · DeepSeek 大模型智能回复 · 支持单条工单实时预测</p>
</div>
""", unsafe_allow_html=True)

# ==================== 接口地址 ====================
bert_url  = f"http://{config.api_host}:{config.api_port}/predict"
llm_url   = f"http://{config.api_host}:{config.api_port}/predict_llm"
reply_url = f"http://{config.api_host}:{config.api_port}/predict_with_reply"


# ==================== 工具函数 ====================
def call_bert(text):
    start = time.time()
    r = requests.post(bert_url, json={"text": text}, timeout=10)
    r.raise_for_status()
    rj = r.json()
    rj["_elapsed_ms"] = (time.time() - start) * 1000
    return rj


def call_llm(text):
    start = time.time()
    r = requests.post(llm_url, json={"text": text}, timeout=30)
    r.raise_for_status()
    rj = r.json()
    rj["_elapsed_ms"] = (time.time() - start) * 1000
    return rj


def call_reply(text):
    start = time.time()
    r = requests.post(reply_url, json={"text": text}, timeout=30)
    r.raise_for_status()
    rj = r.json()
    rj["_elapsed_ms"] = (time.time() - start) * 1000
    return rj


# 类别配色（与 class.txt 顺序一致）
CLASS_COLORS = [
    "#2563eb", "#0e9f6e", "#d97706", "#dc2626",
    "#7c3aed", "#0e7490", "#db2777", "#475569",
]
CLASS_NAMES = list(config.class2id.keys())


def get_class_color(name):
    if name in CLASS_NAMES:
        return CLASS_COLORS[CLASS_NAMES.index(name)]
    return "#64748b"


# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown("## ⚙️ 系统控制台")
    st.markdown('<div class="side-title">系统信息</div>', unsafe_allow_html=True)
    st.info(
        f"**本地模型**：BERT 量化（int8）\n\n"
        f"**生成模型**：DeepSeek-Chat\n\n"
        f"**类别数**：{config.num_labels} 类\n\n"
        f"**后端地址**：`{bert_url}`"
    )

    st.markdown('<div class="side-title">工单类别</div>', unsafe_allow_html=True)
    with st.expander("查看 8 类说明", expanded=False):
        for i, name in enumerate(CLASS_NAMES):
            color = CLASS_COLORS[i % len(CLASS_COLORS)]
            st.markdown(
                f'<span style="display:inline-block;width:10px;height:10px;'
                f'background:{color};border-radius:50%;margin-right:8px;"></span>'
                f"**{i} · {name}**",
                unsafe_allow_html=True,
            )

    st.markdown('<div class="side-title">示例工单</div>', unsafe_allow_html=True)
    EXAMPLES = {
        "商品咨询": "请问这款手机支持无线充电吗？",
        "订单查询": "订单20261015003一直显示待发货，能帮我查一下吗？",
        "物流问题": "我的快递好几天没更新了，什么时候能送到？",
        "退款退货": "收到的商品有质量问题，我要申请换货，运费谁出？",
        "发票问题": "这个订单可以开增值税发票吗？抬头怎么写？",
        "账户问题": "密码忘了怎么找回？修改手机号在哪里操作？",
        "支付问题": "微信支付一直失败，扣款了但订单没生成怎么办？",
        "投诉建议": "客服处理速度太慢了，等了三天没人回复，强烈投诉！",
    }
    for name, txt in EXAMPLES.items():
        if st.button(f"{name}：{txt[:12]}…", key=f"ex_{name}", use_container_width=True):
            st.session_state["input_text"] = txt


# ==================== 主区：输入 ====================
st.markdown('<div class="card"><div class="card-title">📝 工单内容</div></div>',
            unsafe_allow_html=True)

text = st.text_area(
    "请输入工单文本：",
    key="input_text",
    height=110,
    placeholder="示例：我的快递好几天没更新了，什么时候能送到？",
)

model_choice = st.radio(
    "选择预测模式：",
    [
        "本地 BERT（量化）",
        "DeepSeek 大模型",
        "★ 本地分类 + DeepSeek 生成客服回复",
    ],
    horizontal=True,
)

col_btn, _ = st.columns([1, 3])
with col_btn:
    predict_clicked = st.button("🚀 开始预测", type="primary", use_container_width=True)


# ==================== 结果渲染 ====================
def render_bert_result(rj):
    pred = rj["pred_class"]
    conf = rj.get("confidence", 0.0)
    color = get_class_color(pred)

    st.markdown(
        f'<div class="card">'
        f'<div class="card-title">预测结果</div>'
        f'<span class="result-badge" style="background:{color};">{pred}</span>'
        f'<div class="metric-row">'
        f'<span class="metric-item">置信度 <b>{conf:.1%}</b></span>'
        f'<span class="metric-item">推理耗时 <b>{rj["_elapsed_ms"]:.1f} ms</b></span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # 概率分布图
    if "probs" in rj and rj["probs"]:
        df = pd.DataFrame({
            "类别": CLASS_NAMES,
            "概率": rj["probs"],
        }).sort_values("概率", ascending=False)

        chart = (
            alt.Chart(df)
            .mark_bar(cornerRadiusEnd=3, size=20)
            .encode(
                x=alt.X("概率:Q", scale=alt.Scale(domain=[0, 1]), title="概率"),
                y=alt.Y("类别:N", sort="-x", title=None),
                color=alt.condition(
                    alt.datum.类别 == pred,
                    alt.value(color),
                    alt.value("#dbe4f0"),
                ),
                tooltip=["类别", alt.Tooltip("概率:Q", format=".2%")],
            )
            .properties(height=280, title="各类别概率分布")
        )
        st.altair_chart(chart, use_container_width=True)


def render_llm_result(rj):
    pred = rj["pred_class"]
    color = get_class_color(pred)

    st.markdown(
        f'<div class="card">'
        f'<div class="card-title">DeepSeek 分类结果</div>'
        f'<span class="result-badge" style="background:{color};">{pred}</span>'
        f'<div class="metric-row">'
        f'<span class="metric-item">推理耗时 <b>{rj["_elapsed_ms"]:.1f} ms</b></span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def render_reply_result(rj):
    pred = rj["pred_class"]
    conf = rj.get("confidence", 0.0)
    color = get_class_color(pred)

    # 左右两栏
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(
            f'<div class="card">'
            f'<div class="card-title">本地 BERT 分类</div>'
            f'<span class="result-badge" style="background:{color};">{pred}</span>'
            f'<div class="metric-row">'
            f'<span class="metric-item">置信度 <b>{conf:.1%}</b></span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f'<div class="card">'
            f'<div class="card-title">端到端耗时</div>'
            f'<div style="font-size:34px;font-weight:700;color:#1e3a8a;padding:10px 0;">'
            f'{rj["_elapsed_ms"]:.0f} <span style="font-size:16px;color:#64748b;">ms</span>'
            f'</div>'
            f'<div style="color:#64748b;font-size:13px;">'
            f'含本地分类 + DeepSeek 生成回复</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # 回复
    st.markdown('<div class="card-title" style="margin-top:8px;">💬 DeepSeek 生成的客服回复</div>',
                unsafe_allow_html=True)
    if rj.get("reply"):
        st.markdown(f'<div class="reply-box">{rj["reply"]}</div>',
                    unsafe_allow_html=True)
    else:
        st.warning("DeepSeek 没返回内容，请检查 API Key 或网络")


# ==================== 主流程 ====================
if predict_clicked:
    if not text.strip():
        st.warning("请先输入工单文本")
    else:
        try:
            if model_choice == "本地 BERT（量化）":
                with st.spinner("本地 BERT 正在分类..."):
                    rj = call_bert(text)
                render_bert_result(rj)

            elif model_choice == "DeepSeek 大模型":
                with st.spinner("DeepSeek 正在思考..."):
                    rj = call_llm(text)
                render_llm_result(rj)

            elif model_choice == "★ 本地分类 + DeepSeek 生成客服回复":
                with st.spinner("本地模型分类中，DeepSeek 撰写回复中..."):
                    rj = call_reply(text)
                render_reply_result(rj)

        except Exception as e:
            st.error("调用失败，请确认后端 `python api_flask_server.py` 已启动")
            print("error:", e)


# ==================== 页脚 ====================
st.markdown(
    '<div class="app-footer">投满分 · 工单智能分类系统 · '
    '本地量化 BERT + DeepSeek 生成 · 仅供学习演示</div>',
    unsafe_allow_html=True,
)