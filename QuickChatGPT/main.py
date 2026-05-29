import os
import time
# 使用统一检索入口
from mail_indexer.unified_retrieve import unified_search, UnifiedRetriever
# 保留旧的导入以兼容
from mkms_retrieve import retrieve, retrieve_from_emails
import requests
from flask import Flask, render_template, request, jsonify, Response
import json
from tavily import TavilyClient
from QuickChatGPT.chat_history import ChatHistory


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

tavily_client = TavilyClient(api_key="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")

MAX_TURNS = int(os.environ.get("CHAT_MAX_TURNS", "100"))
# MEMORY = defaultdict(lambda: deque(maxlen=MAX_TURNS))

chat_db = ChatHistory('chat_history.db')


DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"


def search_with_tavily(query):
    """
    用 Tavily 搜索（AI原生搜索，直接返回整理好的内容）
    """
    # 基础搜索（返回整理好的摘要）
    response = tavily_client.search(
        query=query,
        search_depth="basic",  # "basic" 或 "advanced"
        max_results=3,
        include_answer=False,  # 是否返回简短答案
        include_raw_content=False,  # 是否返回原始内容
        include_images=False
    )

    if not response or 'results' not in response:
        return ""

    search_text = "\n\n【Tavily搜索结果】\n"

    # 如果有直接答案（像股票价格这种）
    if 'answer' in response and response['answer']:
        search_text += f"\n📊 答案：{response['answer']}\n"

    # 搜索结果
    for i, r in enumerate(response['results'][:3]):
        title = r.get('title', '')[:100]
        content = r.get('content', '')[:200]  # Tavily 直接返回内容摘要
        url = r.get('url', '')
        search_text += f"\n{i + 1}. {title}\n   内容：{content}\n   来源：{url}\n"

    return search_text


def search_with_tavily_advanced(query):
    """
    Tavily 高级版：带直接答案和原始内容
    """
    response = tavily_client.search(
        query=query,
        search_depth="advanced",  # 深度搜索
        max_results=5,
        include_answer=True,  # 返回简短答案
        include_raw_content=False,
        include_images=False
    )

    if not response:
        return ""

    search_text = "\n\n【Tavily深度搜索结果】\n"
    print(search_text)

    # 直接答案（最适合股价、天气这种）
    if 'answer' in response and response['answer']:
        search_text += f"\n📊 {response['answer']}\n"

    # 相关结果
    for i, r in enumerate(response['results'][:3]):
        search_text += f"\n{i + 1}. {r.get('title', '')}\n   {r.get('content', '')[:200]}\n   来源：{r.get('url', '')}\n"

    # 如果有相关问题
    if 'related_questions' in response:
        search_text += "\n❓ 相关问题：\n"
        for q in response['related_questions'][:2]:
            search_text += f"   • {q}\n"

    return search_text


def call_deepseek_api(prompt):
    deepseek_headers = {
        "Authorization": f"Bearer {os.environ.get('DEEPSEEK_API_KEY')}",
        "Content-Type": "application/json"
    }
    deepseek_payload = {
        "model": "deepseek-chat",
        "messages": [{
            "role": "user",
            "content": prompt
        }],
        "temperature": 0.1
    }
    deepseek_response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers=deepseek_headers,
        json=deepseek_payload
    )
    if deepseek_response.status_code == 200:
        return deepseek_response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"API调用失败: {deepseek_response.status_code}, {deepseek_response.text}")


# 在文件开头加个简单缓存
_search_cache = {}


def enable_search(user_msg):
    """让DeepSeek自己判断要不要搜索（带缓存）"""
    # 如果最近1分钟内判断过同样的问题，直接返回缓存结果
    cache_key = user_msg[:50]  # 用前50字符做key
    if cache_key in _search_cache:
        cached_time, cached_result = _search_cache[cache_key]
        if time.time() - cached_time < 60:  # 1分钟内有效
            print(f"【缓存】搜索判断命中: {cached_result}")
            return cached_result

    prompt = f"""
判断以下用户问题是否需要联网搜索实时信息。

需要搜索的情况：
- 问天气、股价、汇率、新闻等实时数据
- 问最近发生的事、最新情况
- 问具体数字、价格、数据
- 问“今天”、“现在”等时间相关的
- 问技术问题、编程问题、需要查找资料的问题
- 用户明确要求搜索的问题

不需要搜索的情况：
- 问个人信息（应该去记忆库找）
- 闲聊、打招呼
- 简单问候
- **只有你100%确定不需要搜索时才返回NO**

只返回 YES 或 NO，不要其他内容。

用户问题：{user_msg}
"""
    result = call_deepseek_api(prompt).strip().upper() == "YES"

    # 存入缓存
    _search_cache[cache_key] = (time.time(), result)
    return result

def call_deepseek_api_stream(messages):
    """流式调用 DeepSeek API"""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY env var")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.3,
        "stream": True
    }

    r = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, stream=True, timeout=60)
    r.raise_for_status()
    return r


def format_retrieved_context(hits):
    """格式化检索结果"""
    if not hits:
        return "（无）"

    lines = []
    for h in hits:
        txt = h["text"].strip()
        if len(txt) > 500:
            txt = txt[:500] + "…"
        lines.append(f"- (score={h['score']:.2f}, source={h['source']}) {txt}")
    return "\n".join(lines)


def build_messages(retrieved_context, history, user_msg, search_results):
    """构建 messages 列表"""

    base = """
你是Morgan的个人AI助手，专业、中性的知识管理助手，是125工程（MKMS工程）的产物。

【核心功能】
- 帮助Morgan整理和回忆个人信息
- 基于【检索到的记忆】回答问题
- 如果需要实时信息（天气、股价、新闻等），会使用Tavily搜索引擎获取最新数据
- 不知道就说不知道，不编造

【行为准则】
1) 绝对不编造任何事实
2) 只有出现在【检索到的记忆】或【实时搜索结果】里的信息，才能当作事实回答
3) 不知道就说不知道，让用户提供
4) 保持专业、简洁、中性的语气

【代码格式要求】
当输出代码时，必须使用标准Markdown格式，例如：
```python
print("Hello World")
"""
    content = f"{base}\n【检索到的记忆】\n{retrieved_context}\n\n【实时搜索结果】\n{search_results}\n\n【当前对话】\n"
    messages = [{"role": "system", "content": content}]
    messages.extend(list(history))
    messages.append({"role": "user", "content": user_msg})
    return messages


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/chat")
def api_chat():
    data = request.get_json(silent=True) or {}
    msg = (data.get("message") or "").strip()

    # 固定用 MORGAN 作为会话ID
    sid = "MORGAN"

    # 👇 从数据库读历史
    history = chat_db.get_recent_messages(sid, MAX_TURNS)

    # 👇 保存用户消息
    chat_db.add_message(sid, "user", msg)

    # 使用统一检索（文档 + 邮件）
    all_hits = unified_search(msg, top_k=10, min_score=0.18)
    retrieved_context = format_retrieved_context(all_hits)
    print(retrieved_context)

    search_result = ""
    if enable_search(msg):
        search_result = search_with_tavily_advanced(msg)

    messages = build_messages(retrieved_context, history, msg, search_result)

    print("【输入】：", msg)
    print("【记忆】：", retrieved_context[:100], '...')
    print("【搜索结果】：", search_result[:100], '...')

    response = call_deepseek_api_stream(messages)
    _reply_written = False

    def generate():
        full_reply = ""
        buffer = ""

        for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
            if not chunk:
                continue

            buffer += chunk

            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip()

                if not line:
                    continue

                if line.startswith('data: '):
                    data = line[6:]

                    if data == '[DONE]':
                        break

                    try:
                        chunk_data = json.loads(data)
                        content = chunk_data.get('choices', [{}])[0].get('delta', {}).get('content', '')

                        if content:
                            full_reply += content
                            yield f"data: {json.dumps({'content': content})}\n\n"
                    except:
                        continue

        yield "data: [DONE]\n\n"

        # 👇 保存助手回复
        chat_db.add_message(sid, "assistant", full_reply)
        _reply_written = True
        print(f"✅ 回复完成，长度: {len(full_reply)}")

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/api/history")
def get_history():
    """获取当前会话的历史记录"""
    # 固定用 MORGAN 作为会话ID
    sid = "MORGAN"

    history = chat_db.get_recent_messages(sid, MAX_TURNS)
    return jsonify({
        "ok": True,
        "messages": history
    })



@app.post("/asr")
def asr():
    f = request.files.get("audio")
    if not f:
        return jsonify({"error": "no audio"}), 400

    os.makedirs("QuickChatGPT/cache", exist_ok=True)
    filename = f"{int(time.time() * 1000)}_{f.filename}"
    path = os.path.join("QuickChatGPT/cache", filename)
    f.save(path)

    print(f"file saved at {path}")
    return jsonify({"ok": True, "file": filename})


@app.route("/meal_decide", methods=["GET", "POST"])
def meal_decide():
    """
    餐饮决策助手 - 接收前端定位，返回附近餐厅推荐
    GET: 返回一个简单的HTML页面
    POST: 接收定位数据，调用高德API返回餐厅列表
    """
    if request.method == "GET":
        # 返回一个简单的HTML页面，包含获取定位的JS
        return render_template("meal_decide.html")

    elif request.method == "POST":
        # 处理POST请求：接收前端定位，返回附近餐厅
        try:
            data = request.get_json() or {}
            lat = data.get('lat')
            lng = data.get('lng')
            radius = data.get('radius', 5000)

            if not lat or not lng:
                return jsonify({"ok": False, "error": "缺少位置参数"}), 400

            # 调用高德API搜索附近餐厅
            restaurants = search_nearby_restaurants(lng, lat, radius)

            return jsonify({
                "ok": True,
                "total": len(restaurants),
                "restaurants": restaurants[:20]  # 只返回前20条
            })

        except Exception as e:
            print(f"❌ meal_decide POST 错误: {e}")
            return jsonify({"ok": False, "error": str(e)}), 500


def search_nearby_restaurants(lng, lat, radius=5000):
    """
    调用高德API搜索附近餐厅
    """
    amap_key = 'fa89d2e6151c3ccaac40189c8228fb90'  # 记得配置环境变量

    url = "https://restapi.amap.com/v3/place/around"
    params = {
        "key": amap_key,
        "location": f"{lng},{lat}",
        "keywords": "餐饮|餐厅|美食|饭店",
        "types": "050000",  # 餐饮服务
        "radius": radius,
        "offset": 20,
        "page": 1,
        "extensions": "all",
        "output": "JSON"
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        data = response.json()

        if data.get("status") != "1":
            print(f"高德API错误: {data.get('info')}")
            return []

        results = []
        for poi in data.get("pois", [])[:20]:
            # 解析评分
            rating = poi.get("biz_ext", {}).get("rating", "暂无")

            # 解析人均
            cost = poi.get("biz_ext", {}).get("cost", "暂无")

            # 解析营业时间
            business_hours = poi.get("opentime", "")

            # 在 search_nearby_restaurants 函数里，构造结果时加上这两个字段
            results.append({
                "name": poi.get("name", "未知"),
                "address": poi.get("address", "未知"),
                "distance": f"{poi.get('distance', '0')}米",
                "phone": poi.get("tel", "暂无电话"),
                "rating": rating,
                "cost": cost,
                "photo": poi.get("photos", [{}])[0].get("url", "") if poi.get("photos") else "",
                # 👇 关键：加上跳转链接

            })

        return results

    except Exception as e:
        print(f"高德API调用失败: {e}")
        return []


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=4200, debug=False)
    # 2026-05-29