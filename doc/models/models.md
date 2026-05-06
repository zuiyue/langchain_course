# LangChain Models 核心概念教程

---


## 一、什么是 Model？

Model 是 Agent  里最底层的 AI 调用抽象，核心作用是：

- 统一不同大模型厂商的调用方式
- 屏蔽不同模型 API 的差异
- 让上层代码可以基于同一套接口切换模型实现

这套设计的重点不是“只支持某个模型”，而是提供一个统一、可移植、可替换的模型调用层。

一句话理解：

- Model 解决“怎么调用模型”【统一】
- Agent 解决“怎么围绕模型完成任务”

### 1.1 Model 在智能体中的角色

如果把 Agent（智能体）比作一个人，那么 **Model（模型）就是这个人的"大脑"**。

- **眼睛和耳朵**：接收用户的输入，理解要做什么
- **大脑皮层**：进行推理、分析、决策
- **嘴巴和手**：输出答案或决定调用哪个工具

在 LangChain 中，模型是 Agent 的**推理引擎**，驱动整个决策过程：
- 决定调用什么工具
- 如何解释工具返回的结果
- 何时输出最终答案

### 1.2 从 LLM 到 Chat Model 的演进

```
传统 LLM（文本补全模型）:
  输入："今天天气"
  输出："很好，适合出去玩。"  ← 只是补全文本

Chat Model（对话模型）:
  输入：[{"role": "user", "content": "今天天气怎么样？"}]
  输出：[{"role": "assistant", "content": "今天天气很好..."}]  ← 结构化对话
```

**关键区别**：

| 特性 | 传统 LLM | Chat Model |
|------|----------|------------|
| **输入格式** | 纯文本字符串 | 带角色的消息列表 |
| **输出格式** | 纯文本字符串 | 结构化的消息对象 |
| **对话历史** | 需要手动拼接 | 天然支持多轮对话 |
| **工具调用** | 不支持 | 原生支持 |
| **主流程度** | 已过时 | 当前主流 |

> **建议**：现代应用应该使用带 `Chat` 前缀的模型（如 `ChatOpenAI`, `ChatAnthropic`），而不是传统的 `OpenAI`, `AnthropicLLM`。

---

## 二、Model 的核心类型

### 2.1 Chat Models（对话模型）—— 主角

这是你将会最常使用的模型类型。它的特点是：

- **消息交互**：接收输入消息（可包含历史对话角色），生成输出消息
- **角色感知**：理解 `user`（用户）、`assistant`（助手）、`system`（系统）等不同角色
- **多轮对话**：天然支持带上下文的对话

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")

# 输入是一条消息
response = model.invoke([{"role": "user", "content": "你好！"}])
print(response.content)  # "你好！有什么我可以帮你的吗？"
```

**常见的 Chat Model 提供商**：

| 提供商 | 类名 | 特点 |
|--------|------|------|
| OpenAI | `ChatOpenAI` | 通用性强，工具调用成熟 |
| Anthropic | `ChatAnthropic` | 长上下文理解优秀 |
| Google | `ChatGoogleGenerativeAI` | Gemini 系列，多模态强 |
| 本地部署 | `ChatOllama` | 隐私保护，无 API 成本 |

### 2.2 Embedding Models（嵌入模型）—— 幕后英雄

嵌入模型的作用是**将文本转换为数字向量**，让计算机能够"理解"文本的语义。

```
"今天天气很好"  ──►  [0.23, -0.45, 0.67, ...]  ← 1536 维的向量
"天气真不错"    ──►  [0.22, -0.44, 0.68, ...]  ← 非常接近的向量

"我喜欢编程"    ──►  [0.89, 0.12, -0.33, ...]  ← 完全不同的向量
```

**为什么这很重要？**

- **语义搜索**：两个向量越接近，文本含义越相似
- **RAG（检索增强生成）**：用它来找到与问题最相关的文档片段
- **去重和聚类**：自动发现相似的文本

### 2.3 Local Models（本地模型）—— 隐私与控制的守护者

当你有以下需求时，本地模型是很好的选择：

- **隐私敏感**：数据不能离开公司网络
- **成本控制**：不想按调用次数付费
- **离线运行**：需要在无网络环境下工作

```python
from langchain_ollama import ChatOllama

# 在本地运行 Llama 模型
model = ChatOllama(model="llama3.1", base_url="http://localhost:11434")
```

---

## 三、Model 的调用方式

### 3.1 三种调用模式

#### 模式 1：`invoke()` —— 同步等待

```
你 ──► 模型 ──► （等待...）──► 完整答案
```

```python
response = model.invoke([{"role": "user", "content": "写一首诗"}])
print(response.content)  # 等待完整生成后才返回
```

**适用场景**：短文本、需要完整结果后再处理

#### 模式 2：`stream()` —— 流式输出

```
你 ──► 模型 ──► "床" ──► "前" ──► "明" ──► "月" ──► ...  ← 逐字返回
```

```python
for chunk in model.stream([{"role": "user", "content": "写一首诗"}]):
    print(chunk.content, end="", flush=True)  # 逐字打印
```

**适用场景**：长文本生成、需要实时反馈改善用户体验

#### 模式 3：`batch()` —— 并行处理

```
请求1 ──┐
请求2 ──┼──► 模型并行处理 ──► [结果1, 结果2, 结果3]
请求3 ──┘
```

```python
responses = model.batch([
    [{"role": "user", "content": "翻译：Hello"}],
    [{"role": "user", "content": "翻译：World"}],
    [{"role": "user", "content": "翻译：Python"}],
])
```

**适用场景**：批量处理、提升吞吐量并降低成本

### 3.2 自动流式（Auto-streaming）

在 LangGraph 等框架中，即使你调用的是 `invoke()`，如果应用处于流式模式，LangChain 也会在底层**自动启用流式传输**。这是透明处理的，你不需要额外配置。

---

## 四、Model 的核心配置

### 4.1 初始化参数

推荐使用的初始化方式是 `init_chat_model()`：

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "anthropic:claude-sonnet-4-5",
    temperature=0.7,      # 控制输出的创造性
    max_tokens=1024,      # 限制最大输出长度
    timeout=30,           # 请求超时（秒）
    max_retries=3,        # 失败重试次数
)
```

**核心参数解析**：

| 参数 | 作用 | 比喻 | 典型值 |
|------|------|------|--------|
| `model` | 指定用哪个模型 | 选择哪个大脑 | `"gpt-4o"`, `"claude-sonnet-4-5"` |
| `temperature` | 控制输出的随机性 | 创造力旋钮 | `0`（确定）~ `1`（创意） |
| `max_tokens` | 限制输出长度 | 字数限制 | `1024`, `4096` |
| `timeout` | 请求超时时间 | 耐心限制 | `30`, `60`（秒） |
| `max_retries` | 失败重试次数 | 再试几次 | `3`, `6`（默认），长任务 `10-15` |

### 4.2 模型档案（Profile）

每个模型都有一个**能力档案**，告诉你它能做什么：

```python
profile = model.profile

print(profile)
# {
#     "max_input_tokens": 200000,      # 最大输入 token 数
#     "max_output_tokens": 4096,       # 最大输出 token 数
#     "tool_calling": True,            # 是否支持工具调用
#     "image_inputs": True,            # 是否支持图像输入
#     "audio_inputs": False,           # 是否支持音频输入
#     "video_inputs": False,           # 是否支持视频输入
# }
```

**为什么这有用？**

你可以根据模型的能力**动态适配**你的应用——如果模型不支持工具调用，就不要给它绑工具。

### 4.3 运行时配置（RunnableConfig）

除了初始化参数，你还可以在每次调用时传入运行时配置：

```python
result = model.invoke(
    [{"role": "user", "content": "你好"}],
    config={
        "run_name": "greeting_call",           # 调用名称（用于追踪）
        "tags": ["production", "user-facing"], # 标签（用于过滤）
        "metadata": {"user_id": "123"},        # 元数据（用于关联）
        "callbacks": [my_callback],            # 回调（用于监控）
    }
)
```

这些配置主要用于**调试、追踪和监控**，特别是与 LangSmith 集成时。

---

## 五、Model 的高级能力

### 5.1 工具调用（Tool Calling）—— 让模型学会"动手"

这是 Chat Model 最重要的能力之一。通过 `bind_tools()`，你可以让模型决定何时调用外部工具。

```python
from langchain_core.tools import tool

@tool
def get_weather(location: str) -> str:
    """查询天气。"""
    return f"{location}: 晴天，22°C"

# 绑定工具到模型
model_with_tools = model.bind_tools([get_weather])

# 调用
response = model_with_tools.invoke([
    {"role": "user", "content": "北京天气怎么样？"}
])

# 模型会返回工具调用请求
print(response.tool_calls)
# [{'name': 'get_weather', 'args': {'location': '北京'}, 'id': 'call_123'}]
```

**工具调用的执行流程**：

```
┌─────────────────────────────────────────────────────────┐
│                    工具调用流程                           │
│                                                         │
│  用户: "北京和上海天气哪个更好？"                         │
│       │                                                  │
│       ▼                                                  │
│  ┌─────────┐                                             │
│  │  模型    │  分析：需要查两个城市的天气                  │
│  └────┬────┘                                             │
│       │                                                  │
│       ├──► get_weather("北京")  ──►  "北京: 晴天,22°C"   │
│       │                                                  │
│       └──► get_weather("上海")  ──►  "上海: 阴天,18°C"   │
│       │                                                  │
│       ▼                                                  │
│  ┌─────────┐                                             │
│  │  模型    │  整合结果：北京天气更好                      │
│  └────┬────┘                                             │
│       │                                                  │
│       ▼                                                  │
│  "北京天气更好，晴天22°C，而上海阴天18°C"                  │
└─────────────────────────────────────────────────────────┘
```

**关键概念：并行工具调用**

模型可以**同时调用多个工具**，而不是一个一个来。这就像一个人可以同时查天气和看日历，大大提升了效率。

### 5.2 结构化输出（Structured Output）—— 让模型按格式回答

有时你需要模型返回**特定格式的数据**，而不是自由文本。

```python
from pydantic import BaseModel

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

# 方式 1：使用 with_structured_output()
structured_model = model.with_structured_output(ContactInfo)
result = structured_model.invoke("提取：张三，邮箱 zhangsan@example.com，电话 13800138000")

print(result)
# ContactInfo(name='张三', email='zhangsan@example.com', phone='13800138000')
```

**三种结构化输出策略**：

| 策略 | 原理 | 可靠性 | 兼容性 |
|------|------|--------|--------|
| `json_schema` | 使用模型厂商的原生 JSON Schema 支持 | 最高 | 有限（仅支持部分厂商） |
| `function_calling` | 通过工具调用模拟，让模型"调用"一个返回 JSON 的虚拟工具 | 高 | 广泛 |
| `json_mode` | 仅保证输出是合法 JSON 格式 | 一般 | 最广泛 |

**获取原始消息和 Token 统计**：

```python
structured_model = model.with_structured_output(ContactInfo, include_raw=True)
result = structured_model.invoke("提取：张三...")

print(result["parsed"])   # ContactInfo 对象
print(result["raw"])      # 原始 AIMessage（含 token 使用量）
```

---

## 六、最佳实践与设计哲学

### 6.1 优先使用流式传输

对于长文本生成，流式传输可以**显著改善用户体验**——用户不需要等待完整答案出来就能看到部分内容。

```python
# ✅ 推荐：流式输出
for chunk in model.stream(messages):
    print(chunk.content, end="", flush=True)
```

### 6.2 利用 batch 并行化

当你需要处理多个独立请求时，`batch()` 比逐个调用更高效，也更便宜。

```python
# ✅ 推荐：批量处理
results = model.batch([messages1, messages2, messages3])

# ❌ 不推荐：逐个调用
results = [model.invoke(m) for m in [messages1, messages2, messages3]]
```

### 6.3 使用 Prompt Caching 降低成本

很多模型提供商支持**提示词缓存**——如果你重复发送相同的前缀内容，缓存部分的处理会更快、更便宜。

```
第一次："请分析这段代码..."  ──► 全价
第二次："请分析这段代码..."  ──► 缓存命中，降价
```

### 6.4 配置回调和元数据追踪

在生产环境中，建议配置回调和元数据，方便与 LangSmith 等平台集成，用于：

- **排查问题**：追踪每次调用的输入输出
- **监控性能**：统计延迟、token 消耗、错误率
- **审计合规**：记录谁在什么时候调用了什么

```python
result = model.invoke(
    messages,
    config={
        "metadata": {"user_id": "123", "feature": "chat"},
        "tags": ["production"],
        "callbacks": [UsageMetadataCallbackHandler()],
    }
)
```

### 6.5 Token 使用量追踪

你可以通过回调或上下文管理器来追踪 token 消耗：

```python
from langchain_core.callbacks import UsageMetadataCallbackHandler

with UsageMetadataCallbackHandler() as cb:
    model.invoke(messages)
    print(cb.total_tokens)  # 总 token 消耗
```

---

## 七、可配置模型 —— 运行时动态切换

通过 `configurable_fields` 和 `config_prefix`，你可以在**运行时动态切换模型**，而无需重新实例化。

```python
from langchain_core.runnables import ConfigurableField

model = ChatOpenAI(model="gpt-4o", temperature=0.7)

# 让 model 和 temperature 可以在运行时配置
configurable_model = model.configurable_fields(
    model=ConfigurableField(id="model"),
    temperature=ConfigurableField(id="temperature"),
)

# 使用不同的配置调用
result = configurable_model.invoke(
    messages,
    config={"configurable": {"model": "gpt-3.5-turbo", "temperature": 0.3}}
)
```

**适用场景**：

- 同一套代码，不同环境用不同模型（开发用便宜模型，生产用强模型）
- 根据用户权限动态切换（普通用户用小模型，VIP 用户用大模型）
- A/B 测试不同模型的效果

---

## 八、总结：Model 的本质

| 概念 | 比喻 | 作用 |
|------|------|------|
| **Chat Model** | 大脑 | 推理、决策、理解 |
| **Embedding Model** | 翻译官 | 将文本转为数字向量 |
| **工具调用** | 手脚 | 让模型能执行实际操作 |
| **结构化输出** | 表格 | 强制模型按格式回答 |
| **多模态** | 眼睛和耳朵 | 理解图像、音频、视频 |
| **流式输出** | 打字机 | 逐字返回，改善体验 |
| **并行批处理** | 多线程 | 同时处理多个请求 |

**Model 的本质**是智能体的**推理引擎**。它不仅理解文本，还能决定行动、调用工具、整合信息，最终输出有价值的答案。


