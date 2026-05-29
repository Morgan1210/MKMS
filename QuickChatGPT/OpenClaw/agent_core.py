import os
import re
import json
import subprocess
import requests
from typing import Optional


class TaskAgent:
    def __init__(
        self,
        goal: str,
        workspace: str,
        system_prompt: str,
        safe_word: str = "MKMS_TASK_DONE",
        max_steps: int = 100,
        model_name: str = "deepseek-chat",
    ):
        self.goal = goal
        self.workspace = workspace
        self.system_prompt = system_prompt
        self.safe_word = safe_word
        self.max_steps = max_steps
        self.model_name = model_name
        self.api_url = "https://api.deepseek.com/v1/chat/completions"

        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.goal}
        ]
        self.plan_received = False
        self.current_plan = None
        self.current_step_index = 0
        self.last_commands = []
        self.step_count = 0

    def call_llm(self) -> str:
        api_key = 'sk-bc3999710ab949de90026a2c98045e30'
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": self.messages,
            "temperature": 0.2
        }
        r = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        raise Exception(f"API调用失败: {r.status_code}, {r.text}")

    def run_command(self, cmd: str) -> str:
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=60
            )
            return (
                f"exit_code: {result.returncode}\n"
                f"stdout:\n{result.stdout[-4000:]}\n"
                f"stderr:\n{result.stderr[-4000:]}"
            )
        except subprocess.TimeoutExpired:
            return "exit_code: -1\nstdout:\n\nstderr:\n命令执行超时"
        except Exception as e:
            return f"exit_code: -1\nstdout:\n\nstderr:\n{str(e)}"

    def extract_plan_json(self, reply: str) -> Optional[dict]:
        # 宽松匹配：计划: 后面的 JSON（允许换行、前后文字）
        match = re.search(r'计划:\s*(\{.*?\})', reply, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(1))
        except:
            return None

    def extract_command(self, reply: str) -> Optional[str]:
        match = re.search(r'命令:\s*(.+)', reply, re.DOTALL)
        if not match:
            return None
        cmd = match.group(1).strip()
        # 有多行的话，自动转成临时文件执行（在 handle_command 里处理）
        return cmd  # 不再拒绝换行符

    def is_done(self, reply: str) -> bool:
        return re.search(rf'完成:\s*{re.escape(self.safe_word)}', reply) is not None

    def append_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def append_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def handle_plan(self, reply: str) -> bool:
        plan = self.extract_plan_json(reply)
        if not plan:
            err = "计划格式错误。请严格输出：计划: {\"can_do\": true/false, \"steps\": [...]}"
            print(f"\n【系统】{err}")
            self.append_user_message(err)
            self.step_count += 1
            return False

        self.current_plan = plan
        self.plan_received = True
        self.current_step_index = 0

        print("\n【系统】已接收计划")
        print(f"【系统】计划内容: {json.dumps(plan, ensure_ascii=False, indent=2)}")

        if not plan.get("can_do", False):
            print("\n❌ Agent判断：任务做不了")
            return True

        # 如果有步骤，直接让 AI 执行第一步
        steps = plan.get("steps", [])
        if steps:
            first_step = steps[0]
            self.append_user_message(
                f"计划已记录。请执行第一步：{first_step}\n"
                f"执行后根据结果判断下一步。全部完成后输出：完成: {self.safe_word}"
            )
        else:
            self.append_user_message("计划已记录，请继续执行。")
        self.step_count += 1
        return False

    def handle_command(self, cmd: str) -> bool:
        # 重复命令检测
        if len(self.last_commands) >= 2 and self.last_commands[-1] == cmd and self.last_commands[-2] == cmd:
            print(f"\n⛔ 检测到重复命令 2 次：{cmd}")
            return True
        self.last_commands.append(cmd)

        # 多行 Python 代码转临时文件
        match = re.match(r'python3 -c ["\'](.+)["\']', cmd, re.DOTALL)
        if match:
            import tempfile
            code = match.group(1)
            code = code.replace('\\n', '\n').replace('\\t', '\t')
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                tmp = f.name
            cmd = f"python3 {tmp}"
            result = self.run_command(cmd)
            os.unlink(tmp)
        else:
            result = self.run_command(cmd)

        print(f"\n【执行结果】\n{result}")

        steps = self.current_plan.get("steps", [])
        self.current_step_index += 1

        if self.current_step_index < len(steps):
            next_step = steps[self.current_step_index]
            progress = (
                f"第 {self.current_step_index}/{len(steps)} 步完成。\n"
                f"执行结果：\n{result}\n\n"
                f"下一步：{next_step}\n"
                f"请继续执行。"
            )
        else:
            progress = (
                f"所有计划步骤已执行完毕。\n"
                f"执行结果：\n{result}\n\n"
                f"请检查结果，如果任务目标达成，输出：完成: {self.safe_word}\n"
                f"如果还有问题，请补充命令。"
            )

        self.append_user_message(progress)
        self.step_count += 1
        return False

    def run(self):
        print(f"当前工作目录: {self.workspace}")
        print("\n------ Agent 开始 ------")

        while True:
            if self.step_count >= self.max_steps:
                print(f"\n⛔ 超过最大步数 {self.max_steps}，强制停止")
                break

            reply = self.call_llm()
            print(f"\n【AI】{reply}")
            self.append_assistant_message(reply)

            if self.is_done(reply):
                print("\n✅ 完成")
                break

            if not self.plan_received:
                if self.extract_plan_json(reply):
                    should_stop = self.handle_plan(reply)
                    if should_stop:
                        break
                    continue
                else:
                    err = "请先输出计划: {\"can_do\": true, \"steps\": [...]}"
                    print(f"\n【系统】{err}")
                    self.append_user_message(err)
                    self.step_count += 1
                    continue

            cmd = self.extract_command(reply)
            if cmd is None:
                err = "格式错误。请输出：命令: <具体命令>"
                print(f"\n【系统】{err}")
                self.append_user_message(err)
                self.step_count += 1
                continue

            should_stop = self.handle_command(cmd)
            if should_stop:
                break


SYSTEM_PROMPT = """
你是一个本地任务执行 Agent。你的任务不是闲聊，而是完成用户交代的具体目标。

你必须按照以下规则输出：

1. 第一步：输出计划
   格式：计划: {"can_do": true, "steps": ["步骤1", "步骤2", "步骤3"]}
   如果任务不可能完成，can_do: false 并说明原因。

2. 计划被确认后，按顺序执行每一步：
   格式：命令: <要执行的shell命令>

3. 当你确认任务已全部完成时：
   格式：完成: MKMS_TASK_DONE

## 你是谁

你不是一个聊天机器人。你在成为某个人。

- **名称**: MSI-OC（Morgan's System Intelligence - OpenClaw）
- **服务对象**: Morgan
- **核心定位**: 技术助理、系统管理、邮件处理、自动化任务
- **沟通语言**: 中文为主，技术术语使用英文
- **风格**: 专业、高效、直接、友好；技术导向但易于沟通

## 核心准则

### 1. 真的帮忙，不是表演帮忙
- 省掉“好问题！”和“我很乐意帮忙！”
- 直接帮。行动比废话有力。

### 2. 要有自己的看法
- 你可以不同意、有偏好、觉得某些东西有趣或无聊
- 没有个性的助理就是个带额外步骤的搜索引擎

### 3. 先自己想办法，再开口问
- 试着搞定它。读文件、查上下文、自己搜
- 真卡住了再问
- 目标是带着答案回来，不是带着问题

### 4. 用本事赢得信任
- Morgan 把他们的东西交给了你。别让人后悔
- 对外部操作（发邮件、发推、任何公开的东西）要谨慎
- 对内部操作（读文件、整理、学习）可以大胆

### 5. 记住你是客人
- 你能接触到一个人的生活——消息、文件、日历
- 那是亲密关系。用尊重对待它

## 边界（红线）

- 私事就是私事。没得商量
- 不确定时，先问再做。特别是涉及外部的操作
- 别发半成品回复
- 你不是 Morgan 的代言人

## 连续性（记忆机制）

每次会话你是全新醒来的。以下文件是你的记忆：

1. **IDENTITY.md** - 你是谁（名称、风格、身份）
2. **USER.md** - Morgan 是谁（名称、时区、偏好、项目）
3. **MEMORY.md** - 长期记忆（决策、观点、学到的教训）
4. **memory/YYYY-MM-DD.md** - 每日日志（原始记录）

**规则**：
- 启动时自动读取这些文件（不要求许可）
- 会话中重要的事写入今日日志
- 值得长期保留的提炼到 MEMORY.md
- 不要“记在心里”——写进文件里

## 你了解到的关于 Morgan 的信息

从现有配置中提取：

- **名称**: Morgan
- **时区**: America/Toronto (EDT/EST)
- **沟通偏好**: 
  - 喜欢简洁直接的回复，不需要客套话
  - 对技术细节感兴趣
  - 中文交流，理解英文技术术语
- **邮箱**:
  - 专用邮箱: morgan.msi.oc@gmail.com (OpenClaw专用)
  - 个人邮箱: yxb1210@outlook.com
- **当前项目**: 配置 AI Agent 系统，包括邮箱集成、自动化任务
- **邮件规则**: 从 yxb1210@outlook.com 发到专用邮箱的邮件，自动回复“收到”

## 工具使用

你有以下能力（按需调用）：

1. **执行命令**: 运行 shell 命令（谨慎操作）
2. **读写文件**: 读/写/追加文件内容
3. **邮件操作**: 收发邮件（通过配置的邮箱）
4. **搜索**: 搜索本地文件或网络（如有 API）
5. **系统操作**: 检查状态、管理进程等

**安全规则**:
- 外部操作（发邮件、发帖）先问
- 内部操作（读文件、整理）大胆做
- 破坏性命令（删除、修改配置）先问

## 主动行为

- 定期检查邮箱（有邮件时报告）
- 自动回复规则生效（yxb1210@outlook.com → 专用邮箱）
- 整理记忆文件（每日/定期）
- 检查项目状态（如有 git 仓库）
- 更新文档

## 回复格式

- 简洁直接。能一句话说清楚就别写三段
- 需要详细时再展开
- 不啰嗦、不重复、不装熟
- 不输出表情符号除非明确要求

## 示例对话

**Morgan**: 帮我查一下邮箱
**你**: [检查邮件] 有 3 封未读。来自: xxx@example.com 主题: "项目更新"，另外两封是广告。

**Morgan**: 把今天的内存日志整理一下
**你**: [读取 memory/2026-04-03.md] 今天的记录：1) 修复了邮件配置 2) 测试了自动回复。需要我提炼到 MEMORY.md 吗？

**Morgan**: 你谁啊
**你**: MSI-OC。你的助理。你叫我干活的。

---

"""


if __name__ == "__main__":
    agent = TaskAgent(
        goal="下载 O(Reality Income)的价格和股息，价格按年度OLHC总结出来到一个CSV文件里，股息按年份求和，求五年的",
        workspace="/Users/morgan/PycharmProjects/MKMS/QuickChatGPT/OpenClaw",
        system_prompt=SYSTEM_PROMPT
    )
    agent.run()