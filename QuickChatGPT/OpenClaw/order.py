import json
import websocket

WS_URL = "ws://127.0.0.1:18789"

def recvj(ws):
    return json.loads(ws.recv())

ws = websocket.create_connection(WS_URL, timeout=10)

print("challenge:", recvj(ws))

ws.send(json.dumps({
    "type": "req",
    "id": "c1",
    "method": "connect",
    "params": {
        "minProtocol": 3,
        "maxProtocol": 3,
        "client": {
            "id": "cli",
            "version": "0.1",
            "platform": "macos",
            "mode": "operator"
        },
        "role": "operator",
        "scopes": ["operator.read", "operator.write"],
        "caps": [],
        "commands": [],
        "permissions": {}
    }
}))

while True:
    msg = recvj(ws)
    print(json.dumps(msg, ensure_ascii=False, indent=2))
    if msg.get("type") == "res" and msg.get("id") == "c1":
        break

ws.close()