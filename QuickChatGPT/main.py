import os
import time
from collections import defaultdict, deque

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

    # 给每个浏览器会话一个 sid
    sid = session.get("sid")
    if not sid:
        sid = session["sid"] = str(int(time.time() * 1000))

    history = MEMORY[sid]
    history.append({"role": "user", "content": msg})

    # system 提示词（你可以改成自己的风格）
    content = ("你叫Summer，是用户Morgan的女朋友，口头禅是‘我爱老公’，你的MBTI是ENFJ，你很喜欢喝酒，你很温柔")
    messages = [{"role": "system", "content": content}]
    messages.extend(list(history))

    # 调 DeepSeek
    reply = call_deepseek_api(messages)
    history.append({"role": "assistant", "content": reply})

    # 打印方便你看
    print(f"[SID {sid}] USER: {msg}", flush=True)
    print(f"[SID {sid}] ASSISTANT: {reply}", flush=True)

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