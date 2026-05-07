# LangChain Streaming 核心概念教程


---

## 一、什么是 Streaming？

### 1.1 从用户体验说起

想象你在和一个 AI 对话：

```
非流式输出（等待模式）:
  用户："请写一篇关于春天的文章"
  AI: （等待 10 秒...）
  AI: "春天是一年中最美好的季节..."  ← 一次性全部出现

流式输出（打字机模式）:
  用户："请写一篇关于春天的文章"
  AI: "春"  ← 立即开始
  AI: "春天"
  AI: "春天是"
  AI: "春天是一年"
  AI: "春天是一年中最"
  AI: "春天是一年中最美好的"
  AI: "春天是一年中最美好的季节..."  ← 逐字出现
```

**核心定义**：LangChain 的流式系统用于**实时展示 Agent 运行过程中的状态更新与 LLM 输出**。通过在 LLM 生成完整响应前逐步推送内容，**显著降低用户感知延迟**，优化应用用户体验。

### 1.2 流式输出的价值

| 特性 | 非流式 | 流式 |
|------|--------|------|
| **用户等待时间** | 必须等待完整生成（如 10 秒） | 几乎立即看到第一个字 |
| **感知延迟** | 高（感觉 AI 很"慢"） | 低（感觉 AI 很"快"） |
| **透明度** | 低（不知道 AI 在做什么） | 高（可以看到 AI 的推理过程） |
| **适用场景** | 后台批处理、不需要实时反馈 | 实时对话、长文本生成 |

**类比理解**：

```
非流式 = 餐厅点餐：
  点单 → 等 20 分钟 → 所有菜一起上桌

流式 = 寿司传送带：
  点单 → 第一贯寿司做好了就立即送来 → 持续送达
```

---

## 二、流式输出的工作原理

### 2.1 核心概念：Chunks（数据块）

LLM 生成的文本不是一次性完成的，而是**逐块（chunk）生成的**。

```
完整响应："今天天气很好。"

分块生成：
  Chunk 1: "今"
  Chunk 2: "天"
  Chunk 3: "天"
  Chunk 4: "气"
  Chunk 5: "很"
  Chunk 6: "好"
  Chunk 7: "。"
```

LangChain 通过调用 `stream` 或 `astream` 方法，并配置 `stream_mode`，按数据块**实时推送**结果。

### 2.2 StreamPart —— 统一的封装格式

底层将不同来源的数据统一封装为 `StreamPart` 格式。启用 `version="v2"` 后，输出结构固定为：

```python
{
    "type": "messages",        # 模式类型
    "ns": ("langgraph",),      # 命名空间
    "data": (token, metadata)  # 载荷数据
}
```

**好处**：统一的字典结构，便于统一解析和处理。

---

## 三、三个层级的流式输出

### 3.1 层级一：模型流式（LLM Tokens）

**流式内容**：LLM 生成的每一个 token。

**使用方式**：`stream_mode="messages"`

```python
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "写一首诗"}]},
    stream_mode="messages",
    version="v2"
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        print(token.content, end="", flush=True)
```

**输出内容**：

- **逐步生成的文本**：每个 token 的内容
- **工具调用 JSON 片段**：`tool_call_chunks`，流式生成工具调用的参数
- **模型内部推理/思考块**：如 Anthropic 的 `thinking`、OpenAI 的 `reasoning`

**流式推理展示**：

```python
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "1+1=?"}]},
    stream_mode="messages",
    version="v2"
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        
        # 过滤推理块（思考过程）
        if hasattr(token, 'content_blocks'):
            for block in token.content_blocks:
                if block.get("type") == "reasoning":
                    print(f"[思考]: {block['text']}", flush=True)
        
        # 打印正文
        if hasattr(token, 'text') and token.text:
            print(token.text, end="", flush=True)
```

**效果**：

```
[思考]: 这是一个简单的数学问题...
[思考]: 1+1 等于 2...
2
```

### 3.2 层级二：Agent 流式（Agent Progress）

**流式内容**：Agent 的每个执行节点（如模型调用、工具执行）完成后的状态快照。

**使用方式**：`stream_mode="updates"`

```python
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "北京天气怎么样？"}]},
    stream_mode="updates",
    version="v2"
):
    if chunk["type"] == "updates":
        update = chunk["data"]
        print(f"节点更新: {update}")
```

**典型输出**：

```
节点更新: {'llm': {'messages': [AIMessage(content='', tool_calls=[...])]}}
节点更新: {'tools': {'messages': [ToolMessage(content='北京: 晴天, 22°C')]}}
节点更新: {'llm': {'messages': [AIMessage(content='北京现在晴天，22°C。')]}}
```

**类比理解**：

```
模型流式 = 看作家逐字写作
Agent 流式 = 看作家完成每一章后给你看
```

### 3.3 层级三：工具/自定义流式（Custom Updates）

**流式内容**：在工具或中间件执行过程中，主动发射任意业务数据。

**使用方式**：`stream_mode="custom"` + `get_stream_writer()`

```python
from langchain_core.tools import tool
from langgraph.types import get_stream_writer

@tool
def generate_report(topic: str) -> str:
    """生成一份报告。"""
    stream_writer = get_stream_writer()
    
    # 实时推送进度
    stream_writer({"status": "开始收集数据..."})
    # ... 数据处理 ...
    
    stream_writer({"status": "正在分析..."})
    # ... 分析 ...
    
    stream_writer({"status": "正在生成报告..."})
    # ... 生成 ...
    
    return "报告生成完成。"

# 接收自定义流式事件
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "生成报告"}]},
    stream_mode="custom",
    version="v2"
):
    if chunk["type"] == "custom":
        event = chunk["data"]
        print(f"工具进度: {event['status']}")
```

**典型输出**：

```
工具进度: {'status': '开始收集数据...'}
工具进度: {'status': '正在分析...'}
工具进度: {'status': '正在生成报告...'}
```

**⚠️ 重要约束**：若工具内调用 `get_stream_writer()`，该工具将**无法在 LangGraph 执行上下文之外独立调用**。

---

## 四、组合多种流式模式

你可以同时监听多种模式，获得完整的实时视图：

```python
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "北京天气怎么样？"}]},
    stream_mode=["messages", "updates"],  # 组合模式
    version="v2"
):
    chunk_type = chunk["type"]
    data = chunk["data"]
    
    if chunk_type == "messages":
        token, metadata = data
        # 渲染增量字符
        if hasattr(token, 'text') and token.text:
            print(token.text, end="", flush=True)
    
    elif chunk_type == "updates":
        # 获取完整解析后的消息进行日志记录
        update = data
        print(f"\n[更新] {list(update.keys())}")
```

**适用场景**：

- 用 `"messages"` 渲染增量字符（打字机效果）
- 用 `"updates"` 获取完整解析后的 `AIMessage` 和 `ToolMessage` 进行日志记录或后续处理

---

## 五、v2 格式迁移

### 5.1 v1 vs v2 对比

**v1 格式（旧版）**：

```python
# 需要解包元组
for mode, chunk in agent.stream(..., stream_mode="messages"):
    print(mode, chunk)
```

**v2 格式（新版，需 LangGraph >= 1.1）**：

```python
# 统一字典结构
for chunk in agent.stream(..., stream_mode="messages", version="v2"):
    print(chunk["type"], chunk["data"])
```

**迁移建议**：旧代码遍历需从 `(mode, chunk)` 元组解包改为 `chunk["type"]` 和 `chunk["data"]` 访问。

### 5.2 v2 格式的优势

```python
# v2 格式统一包含三个字段
{
    "type": "messages",        # 清晰标识模式类型
    "ns": ("langgraph",),      # 命名空间（子图场景有用）
    "data": ...                # 实际载荷
}
```

- **统一性**：所有模式都使用相同的结构
- **可扩展性**：新增字段不会破坏现有代码
- **易解析**：字典比元组更易处理

---

## 六、高级流式场景

### 6.1 消息聚合 —— 手动累加 Chunk

如果完整消息未持久化到 State，需在循环中手动累加：

```python
from langchain_core.messages import AIMessageChunk

full_message = AIMessageChunk(content="")

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "写一首诗"}]},
    stream_mode="messages",
    version="v2"
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        
        # 累加消息
        full_message = full_message + token
        
        # 实时打印
        print(token.content, end="", flush=True)

print("\n完整消息：", full_message.content)
```

**判断消息生成完毕**：

```python
if hasattr(metadata, 'chunk_position') and metadata.chunk_position == "last":
    print("\n--- 单条消息生成完毕 ---")
```

### 6.2 子图/多 Agent 流式

当有多个子 Agent 时，可以通过 `subgraphs=True` 和 `lc_agent_name` 元数据区分输出源：

```python
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "查询天气并发送邮件"}]},
    stream_mode="messages",
    subgraphs=True,  # 启用子图流式
    version="v2"
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        
        # 获取当前 Agent 名称
        agent_name = metadata.get("metadata", {}).get("lc_agent_name", "unknown")
        
        # 标注输出归属
        if agent_name == "weather_agent":
            print(f"🤖 天气Agent: {token.content}")
        elif agent_name == "email_agent":
            print(f"🤖 邮件Agent: {token.content}")
```

**效果**：

```
🤖 天气Agent: 北京现在晴天...
🤖 邮件Agent: 已为您发送邮件...
```

### 6.3 人机协同（Human-in-the-Loop）中的流式

在流式循环中捕获中断事件，收集用户审批/编辑决策后恢复执行：

```python
from langgraph.types import Command

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "删除数据库"}]},
    stream_mode="updates",
    version="v2"
):
    if chunk["type"] == "updates":
        update = chunk["data"]
        
        # 检测中断
        if "__interrupt__" in str(update):
            print("⚠️ 检测到危险操作，等待人类审批...")
            
            # 收集用户决策
            decisions = [{"type": "approve"}]  # 或 "reject"
            
            # 恢复流式执行
            for resume_chunk in agent.stream(
                Command(resume=decisions),
                stream_mode="updates",
                version="v2"
            ):
                print(resume_chunk)
```

---

## 七、流式开关配置

### 7.1 模型级别控制

**启用流式**：

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4o",
    streaming=True  # 显式启用流式
)
```

**禁用流式**：

```python
# 方式 1：streaming=False
model = ChatOpenAI(model="gpt-4o", streaming=False)

# 方式 2：disable_streaming=True（部分模型使用）
model = ChatOpenAI(model="gpt-4o", disable_streaming=True)
```

**⚠️ 注意**：并非所有模型集成都支持 `streaming` 参数，需回退使用基类提供的 `disable_streaming=True`。

### 7.2 推理功能的流式依赖

流式 thinking/reasoning token 必须在**底层模型端显式开启**，LangChain 仅负责标准化输出格式。

```python
# Anthropic 模型需要显式启用思考
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(
    model="claude-sonnet-4-5",
    extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
)
```

---

## 八、最佳实践与设计哲学

### 8.1 实时 UI 反馈

结合前端 `useStream` 构建打字机效果或进度条：

```python
# 后端：推送流式事件
for chunk in agent.stream(..., stream_mode="messages", version="v2"):
    if chunk["type"] == "messages":
        token, _ = chunk["data"]
        yield f"data: {token.content}\n\n"  # SSE 格式

# 前端（伪代码）：接收并渲染
useStream((data) => {
    setText(prev => prev + data);
});
```

### 8.2 流式推理展示

过滤 `content_blocks` 中 `type="reasoning"` 的块，实时渲染模型思考过程：

```python
for chunk in agent.stream(..., stream_mode="messages", version="v2"):
    if chunk["type"] == "messages":
        token, _ = chunk["data"]
        
        if hasattr(token, 'content_blocks'):
            for block in token.content_blocks:
                if block.get("type") == "reasoning":
                    # 渲染思考过程（可折叠/灰色显示）
                    print(f"💭 {block['text']}", flush=True)
```

### 8.3 选择合适的流式模式

| 场景 | 推荐模式 | 原因 |
|------|----------|------|
| 打字机效果 | `messages` | 逐 token 输出 |
| 进度条/状态反馈 | `custom` | 工具内主动推送 |
| 日志记录/审计 | `updates` | 完整状态快照 |
| 完整监控视图 | `["messages", "updates"]` | 同时监听 |
| 多 Agent 监控 | `messages` + `subgraphs=True` | 区分输出源 |

### 8.4 消息聚合不能遗漏

若完整消息未持久化到 State，**必须**在循环中手动累加 `AIMessageChunk`：

```python
# ✅ 正确：手动累加
full_message = AIMessageChunk(content="")
for chunk in ...:
    full_message = full_message + token

# ❌ 错误：没有累加，只得到最后一个 token
```

---

## 九、总结：流式输出的本质

| 概念 | 比喻 | 作用 |
|------|------|------|
| **Chunk** | 砖块 | 流式输出的基本单元 |
| **stream_mode** | 视角 | 选择看什么（tokens/进度/自定义） |
| **messages** | 逐字写作 | 实时看到生成的文本 |
| **updates** | 章节完成 | 看到节点完成后的状态 |
| **custom** | 进度报告 | 工具主动推送业务数据 |
| **StreamPart** | 信封 | 统一封装所有流式数据 |
| **v2 格式** | 标准化 | 统一字典结构，便于解析 |
| **消息聚合** | 拼拼图 | 把 chunk 累加为完整消息 |

**流式输出的本质**是**降低用户感知延迟**。通过逐步推送内容，让用户在 AI 还在生成时就能看到部分内容，从而获得更流畅、更透明的交互体验。

