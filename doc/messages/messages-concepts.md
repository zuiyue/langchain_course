# LangChain Messages 核心概念教程

---

## 一、什么是 Message？

### 1.1 Message 的本质

如果把 Model（模型）比作一个人，那么 **Message（消息）就是人与人之间说的话**。

- **输入**：你说的话 → 模型的输入消息
- **输出**：模型回复的话 → 模型的输出消息

在 LangChain 中，消息是**与 LLM 交互的基本单元**。它不仅仅是文本，而是一个结构化的对象，包含：

| 组成部分 | 作用 | 示例 |
|----------|------|------|
| **角色（Role）** | 标识这条消息是谁发的 | `"user"`, `"assistant"`, `"system"` |
| **内容（Content）** | 承载实际的数据 | `"你好"`, 图片, 音频 |
| **元数据（Metadata）** | 附加信息 | Token 统计、响应时间、工具调用 ID |

### 1.2 为什么需要结构化的消息？

```
纯文本交互（早期）:
  输入："你是助手。用户说：你好"
  问题：角色和指令混在一起，模型容易混淆

结构化消息（LangChain）:
  SystemMessage("你是助手")
  HumanMessage("你好")
  好处：角色清晰分离，模型准确理解
```

**核心价值**：LangChain 提供了一套**统一的跨模型接口**，无论你用 OpenAI、Anthropic 还是 Google，消息的用法都是一致的。

---

## 二、四种核心消息类型

### 2.1 SystemMessage（系统消息）—— 设定"人设"

**作用**：在对话开始前，告诉模型"你是谁"、"你应该怎么做"。

```python
from langchain_core.messages import SystemMessage

system_msg = SystemMessage(content="你是一个专业的翻译助手，负责将中文翻译成英文。")
```

**类比理解**：
- 就像电影的**导演**——不直接参与对话，但设定了整个对话的基调和规则
- 就像演员的**角色剧本**——告诉 AI 应该扮演什么角色

**典型用途**：

| 场景 | 示例 |
|------|------|
| 设定角色 | `"你是一个资深的代码审查专家"` |
| 设定语气 | `"回答要简洁、专业，不要用表情符号"` |
| 设定约束 | `"不要编造信息，不确定时明确告知用户"` |
| 设定格式 | `"输出必须是 JSON 格式"` |

### 2.2 HumanMessage（人类消息）—— 用户的声音

**作用**：代表用户的输入，是模型需要回应的内容。

```python
from langchain_core.messages import HumanMessage

human_msg = HumanMessage(content="请把'你好'翻译成英文。")
```

**类比理解**：
- 就像**提问者**——发起对话，提出问题或需求

**支持多模态内容**：

HumanMessage 不仅可以是纯文本，还可以包含**图像、音频、文件**等多模态内容：

```python
human_msg = HumanMessage(content=[
    {"type": "text", "text": "这张图片里是什么？"},
    {
        "type": "image_url",
        "image_url": {"url": "https://example.com/cat.jpg"}
    }
])
```

**传入多模态数据的三种方式**：

| 方式 | 适用场景 | 示例 |
|------|----------|------|
| **URL** | 在线资源 | `{"url": "https://..."}` |
| **Base64** | 本地文件（需配合 `mime_type`） | `{"url": "data:image/jpeg;base64,..."}` |
| **File ID** | 提供商托管的文件 | `{"file_id": "file-123"}` |

**支持的媒体类型**：

- 📷 **图像**：JPEG, PNG, GIF, WebP
- 📄 **文档**：PDF, TXT
- 🔊 **音频**：MP3, WAV
- 🎬 **视频**：MP4（部分模型支持）

### 2.3 AIMessage（AI 消息）—— 模型的回答

**作用**：模型的生成输出，包含文本、工具调用指令等。

```python
from langchain_core.messages import AIMessage

ai_msg = AIMessage(content="Hello")
```

**类比理解**：
- 就像**回答者**——根据系统设定和用户问题，给出回应

**AIMessage 的关键属性**：

| 属性 | 作用 | 示例 |
|------|------|------|
| `text` / `content` | 文本或原始内容 | `"Hello"` |
| `content_blocks` | 标准化的内容块列表 | 推理过程、思考过程 |
| `tool_calls` | 模型发起的工具调用 | `[{"name": "search", "args": {...}}]` |
| `usage_metadata` | Token 消耗统计 | `{"input_tokens": 50, "output_tokens": 10}` |

**深入理解 `content` vs `content_blocks`**：

```
content（传统方式）:
  - 松散类型：可能是字符串，也可能是提供商原生的字典列表
  - 不同提供商格式不统一

content_blocks（v1 新特性）:
  - 标准化：将各提供商特有的内容统一为标准格式
  - 例如：Anthropic 的 thinking、OpenAI 的 reasoning
    都会被转换为 ReasoningContentBlock
  - 好处：跨提供商开发时，代码更一致
```

### 2.4 ToolMessage（工具消息）—— 工具的执行结果

**作用**：将外部工具的执行结果传回给模型，让模型基于结果继续推理。

```python
from langchain_core.messages import ToolMessage

tool_msg = ToolMessage(
    content="北京：晴天，22°C",
    tool_call_id="call_abc123"  # 必须与 AIMessage 中的工具调用 ID 对应
)
```

**类比理解**：
- 就像**助手调研后提交的报告**——AI 决定查天气，工具去查了，然后把结果汇报给 AI

**关键概念：`tool_call_id` 的作用**：

```
AIMessage: "我需要查北京的天气"
  └── tool_calls: [{"name": "get_weather", "args": {"location": "北京"}, "id": "call_abc123"}]
                                                      ↑
                                                      │ 这个 ID 必须对应
                                                      ↓
ToolMessage: "北京：晴天，22°C"
  └── tool_call_id: "call_abc123"
```

**`artifact` 属性**：

`ToolMessage` 还有一个特殊的 `artifact` 属性，用于存储**不发送给模型但可供程序读取**的额外数据：

```python
tool_msg = ToolMessage(
    content="北京：晴天，22°C",          # 这部分会发给模型
    tool_call_id="call_abc123",
    artifact={                            # 这部分不会发给模型，但程序可用
        "document_id": "doc-456",
        "source": "weather_api_v2",
        "cached": False
    }
)
```

**适用场景**：
- 存储文档 ID，方便后续引用
- 存储调试信息
- 存储检索元数据（如检索时间、来源）

---

## 三、消息列表（Messages List）—— 对话的上下文

### 3.1 为什么需要消息列表？

模型本质上是**无状态**的——它不记得上一次对话说了什么。要维持多轮对话，必须**显式地传递完整的对话历史**。

```
无状态模型的工作方式：

第 1 轮：[SystemMessage, HumanMessage("我叫小明")]
        ──► 模型回复："好的，小明！"

第 2 轮：[SystemMessage, HumanMessage("我叫什么名字？")]
        ──► 模型回复："我不知道。"  ← 忘了！

正确做法：
第 2 轮：[SystemMessage,
          HumanMessage("我叫小明"),
          AIMessage("好的，小明！"),
          HumanMessage("我叫什么名字？")]
        ──► 模型回复："你叫小明！"  ← 记住了！
```

### 3.2 消息列表的构建

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage(content="你是一个友好的助手。"),
    HumanMessage(content="你好！"),
    AIMessage(content="你好！有什么我可以帮你的吗？"),
    HumanMessage(content="北京天气怎么样？"),
]

response = model.invoke(messages)
```

### 3.3 输入格式的灵活性

LangChain 既支持**消息对象列表**，也兼容 **OpenAI 原生字典格式**：

```python
# 方式 1：使用消息对象（推荐）
from langchain_core.messages import HumanMessage

messages = [HumanMessage(content="你好")]

# 方式 2：使用字典（兼容 OpenAI 格式）
messages = [{"role": "user", "content": "你好"}]

# 两者效果相同
model.invoke(messages)
```

**推荐做法**：在代码中使用消息对象（`HumanMessage`, `AIMessage`），这样更有类型安全感和可读性。

### 3.4 消息列表的动态增长

在基础对话循环中，消息列表会**不断追加**：

```python
messages = [SystemMessage(content="你是助手")]

while True:
    user_input = input("你：")
    messages.append(HumanMessage(content=user_input))
    
    response = model.invoke(messages)
    messages.append(response)  # AIMessage
    
    print(f"AI：{response.content}")
```

---

## 四、流式消息处理：AIMessageChunk

### 4.1 什么是 AIMessageChunk？

当你使用流式输出（`stream()`）时，模型会逐块返回内容。每一块都是一个 `AIMessageChunk` 对象。

```
完整输出: "今天天气很好。"

流式返回:
  Chunk 1: "今天"
  Chunk 2: "天气"
  Chunk 3: "很"
  Chunk 4: "好。"
```

### 4.2 合并 Chunk

`AIMessageChunk` 支持用 `+` 操作符累加合并：

```python
from langchain_core.messages import AIMessageChunk

full_message = AIMessageChunk(content="")

for chunk in model.stream(messages):
    full_message += chunk
    print(chunk.content, end="", flush=True)  # 实时打印

print("\n完整消息：", full_message.content)  # "今天天气很好。"
```

**类比理解**：
- 就像**拼图**——每次拿到一块，拼上去，最后形成完整的图

---

## 五、最佳实践与设计哲学

### 5.1 简单场景：直接传字符串

对于单次生成任务（无需历史记录），直接传字符串是最简洁的做法：

```python
# ✅ 简单场景：直接传字符串
response = model.invoke("请写一首关于春天的诗。")
```

### 5.2 复杂场景：使用消息对象

当涉及**多轮对话、系统指令、多模态输入**时，使用消息对象：

```python
# ✅ 复杂场景：使用消息对象
messages = [
    SystemMessage(content="你是一个诗歌创作助手。"),
    HumanMessage(content="请写一首关于春天的诗。"),
]
response = model.invoke(messages)
```

### 5.3 上下文窗口管理

模型的上下文窗口是有限的。建议：

- **修剪（Trim）**：移除最早的消息，保留最近的对话
- **总结（Summarize）**：用模型把长对话历史总结成简短的摘要
- **结合 LangChain 的短期记忆机制**：自动管理消息列表长度

```python
# 示例：保留最近 10 条消息
if len(messages) > 10:
    messages = [messages[0]] + messages[-10:]  # 保留 SystemMessage + 最近 10 条
```

### 5.4 外部序列化标准化

如果需要将消息传递给 LangChain 外部系统（如数据库、API），建议启用标准化的内容块序列化格式：

```python
# 设置环境变量
import os
os.environ["LC_OUTPUT_VERSION"] = "v1"

# 或在模型参数中设置
model = ChatOpenAI(..., output_version="v1")
```

### 5.5 HumanMessage 的 `name` 字段

`HumanMessage` 支持 `name` 字段，用于标识不同用户：

```python
msg1 = HumanMessage(content="你好", name="小明")
msg2 = HumanMessage(content="你好", name="小红")
```

**注意**：具体支持情况取决于底层模型提供商——部分模型会忽略 `name` 字段。

---

## 六、总结：消息类型全景图

| 消息类型 | 角色 | 比喻 | 谁在用 | 典型内容 |
|----------|------|------|--------|----------|
| **SystemMessage** | system | 导演/剧本 | 开发者设定 | "你是翻译助手" |
| **HumanMessage** | user | 提问者 | 用户输入 | "请翻译这句话" |
| **AIMessage** | assistant | 回答者 | 模型输出 | "Hello" + 工具调用 |
| **ToolMessage** | tool | 调研报告 | 工具结果 | "北京：晴天，22°C" |

**消息流转的完整流程**：

```
┌─────────────────────────────────────────────────────────┐
│                    完整对话流程                          │
│                                                         │
│  SystemMessage("你是天气助手")                            │
│       │                                                  │
│       ▼                                                  │
│  HumanMessage("北京天气怎么样？")                         │
│       │                                                  │
│       ▼                                                  │
│  AIMessage（模型决定调用工具）                             │
│       │ tool_calls: [{"name": "get_weather", ...}]       │
│       ▼                                                  │
│  [工具执行：查询 API]                                     │
│       │                                                  │
│       ▼                                                  │
│  ToolMessage("北京：晴天，22°C")                          │
│       │                                                  │
│       ▼                                                  │
│  AIMessage（整合结果，输出最终答案）                        │
│       │                                                  │
│       ▼                                                  │
│  "北京现在是晴天，气温 22°C。"                            │
└─────────────────────────────────────────────────────────┘
```

**Message 的本质**是**结构化的对话单元**。它让模型能够理解谁在说话、说了什么、在什么背景下说的，从而实现准确、连贯的交互。

---
