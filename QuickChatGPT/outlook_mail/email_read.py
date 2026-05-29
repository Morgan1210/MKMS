import requests
import time
import json
from pathlib import Path

# ==================== 配置 ====================
CLIENT_ID = "4d7ff71c-b75b-4e45-bfeb-1b536a99c11e"
TENANT = 'consumers'  # 个人账号专用租户

SCOPES = [
    "Mail.Read",
    "Mail.ReadWrite",
    "Mail.Send",
    "offline_access"
]

TOKEN_FILE = "ms_graph_token.json"


# ==================== Token 管理 ====================
def get_token_device_code():
    """设备代码流获取 token"""
    device_code_url = f"https://login.microsoftonline.com/{TENANT}/oauth2/v2.0/devicecode"
    device_params = {
        "client_id": CLIENT_ID,
        "scope": " ".join(SCOPES)
    }

    print("🔄 请求设备代码...")
    device_resp = requests.post(device_code_url, data=device_params)
    device_data = device_resp.json()

    if "error" in device_data:
        print(f"❌ 请求失败: {device_data.get('error_description')}")
        return None

    print("\n" + "=" * 50)
    print("🔑 请完成授权：")
    print(f"1. 在浏览器中打开: {device_data['verification_uri']}")
    print(f"2. 输入验证码: {device_data['user_code']}")
    print("=" * 50 + "\n")

    token_url = f"https://login.microsoftonline.com/{TENANT}/oauth2/v2.0/token"
    token_params = {
        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        "client_id": CLIENT_ID,
        "device_code": device_data["device_code"]
    }

    print("⏳ 等待授权...")
    expires_in = device_data.get("expires_in", 600)
    interval = device_data.get("interval", 5)
    start_time = time.time()

    while time.time() - start_time < expires_in:
        token_resp = requests.post(token_url, data=token_params)
        token_data = token_resp.json()

        if "access_token" in token_data:
            print("✅ 授权成功！")
            return token_data
        elif token_data.get("error") == "authorization_pending":
            time.sleep(interval)
        elif token_data.get("error") == "slow_down":
            interval += 5
            time.sleep(interval)
        else:
            print(f"❌ 错误: {token_data.get('error_description')}")
            return None

    print("❌ 授权超时")
    return None


def save_token(token_data):
    """保存 token 到文件"""
    token_data["acquired_at"] = time.time()
    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        json.dump(token_data, f, ensure_ascii=False, indent=2)
    print(f"💾 Token 已保存到 {TOKEN_FILE}")


def load_token():
    """从文件加载 token，自动刷新"""
    if not Path(TOKEN_FILE).exists():
        return None

    with open(TOKEN_FILE, "r", encoding="utf-8") as f:
        token_data = json.load(f)

    expires_in = token_data.get("expires_in", 3600)
    acquired_at = token_data.get("acquired_at", 0)

    # 如果 token 还有效（预留5分钟缓冲）
    if time.time() - acquired_at < expires_in - 300:
        return token_data["access_token"]

    print("🔄 Token 已过期，尝试刷新...")
    refresh_token = token_data.get("refresh_token")
    if not refresh_token:
        return None

    refresh_url = f"https://login.microsoftonline.com/{TENANT}/oauth2/v2.0/token"
    refresh_params = {
        "grant_type": "refresh_token",
        "client_id": CLIENT_ID,
        "refresh_token": refresh_token,
        "scope": " ".join(SCOPES)
    }

    refresh_resp = requests.post(refresh_url, data=refresh_params)
    if refresh_resp.status_code == 200:
        new_token = refresh_resp.json()
        new_token["acquired_at"] = time.time()
        save_token(new_token)
        return new_token["access_token"]

    print(f"❌ 刷新失败: {refresh_resp.status_code}")
    return None


# ==================== 邮件读取 ====================
def get_inbox_stats(access_token):
    """获取收件箱统计信息"""
    url = "https://graph.microsoft.com/v1.0/me/mailFolders/inbox"
    headers = {"Authorization": f"Bearer {access_token}"}

    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        data = resp.json()
        total = data.get("totalItemCount", 0)
        unread = data.get("unreadItemCount", 0)
        print(f"\n📊 收件箱统计: 总共 {total} 封，未读 {unread} 封")
        return total, unread
    else:
        print(f"❌ 获取统计失败: {resp.status_code}")
        return None, None


def get_all_emails(access_token, batch_size=50, max_pages=None):
    """
    分页获取所有邮件

    Args:
        access_token: 访问令牌
        batch_size: 每页数量 (最大1000)
        max_pages: 最大页数，None 表示获取全部
    """
    url = "https://graph.microsoft.com/v1.0/me/messages"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {
        "$top": batch_size,
        "$orderby": "receivedDateTime DESC",
        "$select": "subject,from,receivedDateTime,isRead,body,id,parentFolderId",
        "$count": "true",
    }

    all_messages = []
    page_count = 1

    print(f"\n📦 开始获取邮件...")

    while url and (max_pages is None or page_count <= max_pages):
        print(f"   第 {page_count} 页...", end="")

        resp = requests.get(url, headers=headers, params=params if page_count == 1 else {})

        if resp.status_code != 200:
            print(f" ❌ 失败: {resp.status_code}")
            break

        data = resp.json()
        messages = data.get("value", [])
        all_messages.extend(messages)

        print(f" 本页 {len(messages)} 封，累计 {len(all_messages)} 封")

        # 获取下一页链接
        url = data.get("@odata.nextLink")
        params = None  # 第一页之后不需要 params
        page_count += 1

    print(f"\n✅ 完成！共获取 {len(all_messages)} 封邮件")

    with open('email_content.json', "w", encoding="utf-8") as f:
        json.dump(all_messages, f, ensure_ascii=False, indent=2)

    return all_messages


def display_emails(messages, show_body=False, max_preview=50):
    """
    显示邮件列表

    Args:
        messages: 邮件列表
        show_body: 是否显示正文预览
        max_preview: 正文预览最大长度
    """
    if not messages:
        print("📭 没有邮件")
        return

    for i, msg in enumerate(messages, 1):
        # 基本信息
        subject = msg.get('subject', '无主题')
        read_status = "✅" if msg.get('isRead') else "📌"
        from_name = msg.get('from', {}).get('emailAddress', {}).get('address', '未知')
        received_time = msg.get('receivedDateTime', '未知')

        print(f"\n{i}. {read_status} [{from_name}]")
        print(f"   主题: {subject}")
        print(f"   时间: {received_time[:10]} {received_time[11:16] if len(received_time) > 16 else ''}")

        # 正文预览
        if show_body:
            preview = msg.get('bodyPreview', '')[:max_preview]
            if preview:
                print(f"   预览: {preview}{'...' if len(preview) == max_preview else ''}")

        # 分隔线
        if i % 5 == 0 and i < len(messages):
            print("-" * 40)


def get_unread_emails(access_token, top=50):
    """获取未读邮件"""
    url = "https://graph.microsoft.com/v1.0/me/messages"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {
        "$top": top,
        "$filter": "isRead eq false",
        "$orderby": "receivedDateTime DESC",
        "$select": "subject,from,receivedDateTime,bodyPreview"
    }

    resp = requests.get(url, headers=headers, params=params)
    if resp.status_code == 200:
        messages = resp.json().get("value", [])
        print(f"\n📌 未读邮件 ({len(messages)} 封):")
        for msg in messages:
            print(f"   [{msg.get('from', {}).get('emailAddress', {}).get('address', '未知')}] {msg.get('subject')}")
        return messages
    return None


# ==================== 邮件操作 ====================
def mark_as_read(access_token, message_id):
    """标记邮件为已读"""
    url = f"https://graph.microsoft.com/v1.0/me/messages/{message_id}"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    data = {"isRead": True}

    resp = requests.patch(url, headers=headers, json=data)
    if resp.status_code == 200:
        print("✅ 已标记为已读")
        return True
    else:
        print(f"❌ 标记失败: {resp.status_code}")
        return False


def delete_email(access_token, message_id):
    """删除邮件"""
    url = f"https://graph.microsoft.com/v1.0/me/messages/{message_id}"
    headers = {"Authorization": f"Bearer {access_token}"}

    resp = requests.delete(url, headers=headers)
    if resp.status_code == 204:
        print("✅ 邮件已删除")
        return True
    else:
        print(f"❌ 删除失败: {resp.status_code}")
        return False


# ==================== 主函数 ====================
def main():
    """主函数"""
    # 1. 获取 token
    access_token = load_token()
    if not access_token:
        print("🔑 需要重新授权")
        token_data = get_token_device_code()
        if token_data:
            save_token(token_data)
            access_token = token_data["access_token"]
        else:
            print("❌ 授权失败")
            return

    print("\n" + "=" * 60)
    print("✅ 已获取访问令牌，开始操作...")
    print("=" * 60)

    # 2. 查看收件箱统计
    total, unread = get_inbox_stats(access_token)
    if total is None:
        return

    # 3. 获取所有邮件
    all_emails = get_all_emails(access_token, batch_size=50, max_pages=None)

    # 4. 显示邮件
    print("\n" + "=" * 60)
    print("📬 收件箱所有邮件")
    print("=" * 60)

    display_emails(all_emails, show_body=False)

    # 5. 显示统计信息
    print("\n" + "=" * 60)
    print("📊 统计信息")
    print("=" * 60)
    print(f"   总邮件数: {len(all_emails)}")
    unread_count = sum(1 for m in all_emails if not m.get('isRead'))
    print(f"   未读邮件: {unread_count}")
    print(f"   已读邮件: {len(all_emails) - unread_count}")

    # 可选：显示最近的未读邮件
    if unread_count > 0:
        print("\n📌 最近未读:")
        unread = [m for m in all_emails if not m.get('isRead')][:5]
        for msg in unread:
            from_addr = msg.get('from', {}).get('emailAddress', {}).get('address', '未知')
            print(f"   [{from_addr}] {msg.get('subject')}")


if __name__ == "__main__":
    main()