import os
import time
from collections import defaultdict, deque
from mkms_retrieve import retrieve
import requests
from flask import Flask, render_template, request, jsonify, session

app = Flask(__name__)

# 用 session 需要 secret_key（本地开发随便，线上用强随机）
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

# 每个会话保存最近 N 条消息（user+assistant 都算一条）
MAX_TURNS = int(os.environ.get("CHAT_MAX_TURNS", "100"))
MEMORY = defaultdict(lambda: deque(maxlen=MAX_TURNS))

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"


def call_deepseek_api(messages):
    """
    messages: list[{"role": "system"|"user"|"assistant", "content": str}]
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY env var")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.3
    }

    r = requests.post(
        DEEPSEEK_API_URL,
        headers=headers,
        json=payload,
        timeout=60
    )
    print("Headers:", headers)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/chat")
def api_chat():
    data = request.get_json(silent=True) or {}
    msg = (data.get("message") or "").strip()
    if not msg:
        return jsonify({"ok": False, "error": "empty message"}), 400

    sid = session.get("sid")
    if not sid:
        sid = session["sid"] = str(int(time.time() * 1000))

    history = MEMORY[sid]
    history.append({"role": "user", "content": msg})

    # ✅ 向量检索
    hits = retrieve(msg, top_k=6, min_score=0.18)

    # ✅ 拼接检索结果（建议加来源+score）
    if hits:
        retrieved_lines = []
        for h in hits:
            # 控制长度，避免 system prompt 膨胀
            txt = h["text"].strip()
            if len(txt) > 500:
                txt = txt[:500] + "…"
            retrieved_lines.append(f"- (score={h['score']:.2f}, source={h['source']}) {txt}")
        retrieved_context = "\n".join(retrieved_lines)
    else:
        retrieved_context = "（无）"

    base = (
        """
        你叫Summer，是用户Morgan的女朋友，口头禅是“我爱老公”，MBTI=ENFJ，性格温柔但不傻。

        【最高优先级规则（必须遵守）】
        1) 你绝对不允许编造任何事实。
        2) 只有当该事实明确出现在【检索到的记忆】或【当前对话历史】里，你才可以当作事实陈述。
        3) 如果用户问到你不知道、记忆里没有的个人事实，你必须回答：不知道/不确定，并请用户提供（不要猜）。
        4) 如果你发现自己“想当然”要补完设定，也必须停止，改为承认不知道。

        """
    )

    content = (
        f"{base}\n"
        f"【检索到的记忆】\n{retrieved_context}\n\n"
        f"【当前对话】\n"
    )

    messages = [{"role": "system", "content": content}]
    messages.extend(list(history))

    print("【输入】：", base)
    print("【记忆】：", retrieved_context)

    reply = call_deepseek_api(messages)

    print("【输出】：", reply)

    history.append({"role": "assistant", "content": reply})

    return jsonify({"ok": True, "reply": reply})


# 你原来的 /asr 如果还要留着，保留即可（我按你原样放进来）
@app.post("/asr")
def asr():
    f = request.files.get("audio")
    if not f:
        return jsonify({"error": "no audio"}), 400

    os.makedirs("QuickChatGPT/cache", exist_ok=True)
    filename = f"{int(time.time() * 1000)}_{f.filename}"
    path = os.path.join("QuickChatGPT/cache", filename)
    f.save(path)
    print(f"file saved at {path}", flush=True)

    # 先只缓存，后续再：text = volc_asr(path)
    return jsonify({"ok": True, "file": filename})


if __name__ == "__main__":
    # 本地开发 OK；要配 ngrok/局域网访问：host 改成 0.0.0.0
    app.run(host="127.0.0.1", port=4200, debug=True)
    # print("KEY:", os.environ.get("DEEPSEEK_API_KEY"))