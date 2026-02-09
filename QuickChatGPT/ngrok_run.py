from pyngrok import ngrok
import time

# 把本地 4200 暴露出去
print(1)
public_url = ngrok.connect(4200, "http")
print("🚀 Public URL:", public_url)

# 防止脚本退出
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    ngrok.disconnect(public_url)