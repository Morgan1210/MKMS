import json
import time
import uuid
import requests
import base64
import os
import time


# 辅助函数：将本地文件转换为Base64
def file_to_base64(file_path):
    with open(file_path, 'rb') as file:
        file_data = file.read()  # 读取文件内容
        base64_data = base64.b64encode(file_data).decode('utf-8')  # Base64 编码
    return base64_data


# recognize_task 函数
def recognize_task(file_url=None, file_path=None):
    recognize_url = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash"
    # 填入控制台获取的app id和access token
    appid = "6050850142"
    token = "oyaEb13gdq4UWUt99TGPePCOKvxWoxrC"

    headers = {
        "X-Api-App-Key": appid,
        "X-Api-Access-Key": token,
        "X-Api-Resource-Id": "volc.bigasr.auc_turbo",
        "X-Api-Request-Id": str(uuid.uuid4()),
        "X-Api-Sequence": "-1",
    }

    # 检查是使用文件URL还是直接上传数据
    audio_data = None
    if file_url:
        audio_data = {"url": file_url}
    elif file_path:
        base64_data = file_to_base64(file_path)  # 转换文件为 Base64
        audio_data = {"data": base64_data}  # 使用Base64编码后的数据

    if not audio_data:
        raise ValueError("必须提供 file_url 或 file_path 其中之一")

    request = {
        "user": {
            "uid": appid
        },
        "audio": audio_data,
        "request": {
            "model_name": "bigmodel",
            # "enable_itn": True,
            # "enable_punc": True,
            # "enable_ddc": True,
            # "enable_speaker_info": False,

        },
    }

    response = requests.post(recognize_url, json=request, headers=headers)
    if 'X-Api-Status-Code' in response.headers:
        print(f'recognize task response header X-Api-Status-Code: {response.headers["X-Api-Status-Code"]}')
        print(f'recognize task response header X-Api-Message: {response.headers["X-Api-Message"]}')
        print(time.asctime() + " recognize task response header X-Tt-Logid: {}".format(response.headers["X-Tt-Logid"]))
        print(f'recognize task response content is: {response.json()}\n')
        try:
            text = response.json()['result']['text']
        except KeyError:
            text = "KeyError!"
    else:
        print(f'recognize task failed and the response headers are:: {response.headers}\n')
        exit(1)
    # return response
    return text


CACHE_DIR = "/Users/morgan/PycharmProjects/MKMS/QuickChatGPT/cache"


def scan_cache():
    if not os.path.exists(CACHE_DIR):
        print("cache 目录不存在")
        return

    files = []
    for name in os.listdir(CACHE_DIR):
        path = os.path.join(CACHE_DIR, name)
        if os.path.isfile(path):
            stat = os.stat(path)
            files.append({
                "name": name,
                "path": path,
                "size_kb": round(stat.st_size / 1024, 2),
                "mtime": time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(stat.st_mtime)
                )
            })

    # 按时间倒序
    files.sort(key=lambda x: x["mtime"], reverse=True)

    for f in files:
        print(f"{f['mtime']} | {f['size_kb']:>7} KB | {f['name']}")


# if __name__ == '__main__':
    # main()
    # print(recognize_task(file_path='/Users/morgan/PycharmProjects/MKMS/QuickChatGPT/cache/1769692072725_recording.webm'))