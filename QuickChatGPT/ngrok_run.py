import os
import sys

# 把当前目录从路径里临时移除
current_dir = os.path.dirname(__file__)
if current_dir in sys.path:
    sys.path.remove(current_dir)

from pyngrok import ngrok
import time

# 把本地 4200 暴露出去
public_url = ngrok.connect(4200, "http")
print("🚀 Public URL:", public_url)

# 防止脚本退出
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    ngrok.disconnect(public_url)