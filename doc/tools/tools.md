# LangChain Tools 核心概念教程

---

## 一、什么是 Tool？

### 1.1 Tool 的本质

如果把 Agent（智能体）比作一个人，那么 **Tool（工具）就是这个人的手脚和感官**。

```
LLM 本身的能力:
  ✅ 理解语言、翻译、总结、生成文本
  ❌ 无法获取实时数据（天气、新闻、股价）
  ❌ 无法执行外部操作（发邮件、查数据库、调用 API）
  ❌ 无法读取文件、访问互联网

LLM + Tools:
  ✅ 获取实时数据 ──► 调用天气 API
  ✅ 执行外部操作 ──► 发送邮件、创建工单
  ✅ 查询数据库 ──► 检索用户信息
  ✅ 读写文件 ──► 读取配置文件、写入日志
```

**核心定义**：工具是**扩展 Agent 能力的可调用函数**。它让模型能够获取实时数据、执行代码、查询外部数据库，并采取现实世界的行动。

### 1.2 工具的工作机制

```
传统函数调用:
  程序员决定何时调用 → 写死在代码中 → 执行

Agent 中的工具:
  模型自主决定何时调用 → 根据对话上下文推理 → 执行
  ↑
  这是关键！工具不是被程序员调用的，而是被模型"选择"调用的
```

**完整决策流程**：

```
用户："北京现在天气怎么样？适合跑步吗？"
      │
      ▼
┌─────────────┐
│   LLM 推理   │  思考：我需要查北京的天气
└──────┬──────┘
       │
       ▼ 模型自主决定调用工具
┌─────────────┐
│ get_weather │  参数：{"location": "北京"}
└──────┬──────┘
       │
       ▼ 执行工具，返回结果
  "北京：晴天，22°C，微风"
       │
       ▼
┌─────────────┐
│   LLM 推理   │  思考：22°C 微风，适合跑步
└──────┬──────┘
       │
       ▼
  "北京现在晴天 22°C，微风，非常适合跑步！"
```

---

## 二、工具的定义方式

### 2.1 基础方式：`@tool` 装饰器

这是最常见的工具定义方式。使用 `@tool` 装饰 Python 函数，默认将函数的 **docstring（文档字符串）** 作为工具描述。

```python
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """查询指定城市的实时天气。
    
    Args:
        location: 城市名称，如"北京"、"上海"
    """
    # 实际场景中这里会调用天气 API
    return f"{location}: 晴天，22°C"
```

**这个简单的定义包含了四个关键部分**：

| 部分 | 来源 | 示例值 |
|------|------|--------|
| **名称** | 函数名 | `get_weather` |
| **描述** | docstring | "查询指定城市的实时天气..." |
| **参数** | 类型提示 | `location: str` |
| **返回值** | 函数返回 | `"北京: 晴天，22°C"` |

### 2.2 自定义名称和描述

你可以覆盖默认的函数名和 docstring，提供更精准的描述来优化模型的引导。

```python
# 自定义名称
@tool("query_weather")
def get_weather(location: str) -> str:
    """查询指定城市的实时天气。"""
    ...

# 自定义名称 + 描述
@tool("search", description="搜索互联网获取最新的新闻、事实和实时信息。当你需要 2024 年之后的最新数据时使用此工具。")
def search_web(query: str) -> str:
    """这个 docstring 会被上面的 description 覆盖。"""
    ...
```

### 2.3 高级 Schema：复杂参数结构

当工具需要**嵌套的、结构化的输入**时，可以通过 `args_schema` 参数传入 Pydantic 模型：

```python
from pydantic import BaseModel, Field
from langchain.tools import tool

class WeatherQuery(BaseModel):
    """天气查询的参数。"""
    location: str = Field(description="城市名称，如'北京'、'上海'")
    date: str = Field(description="日期，格式为 YYYY-MM-DD。不填则查当前天气")

@tool("get_weather", args_schema=WeatherQuery)
def get_weather(location: str, date: str = None) -> str:
    """查询指定城市在指定日期的天气。"""
    if date:
        return f"{location} {date}: 晴天，22°C"
    return f"{location}: 晴天，22°C"
```

---

## 三、工具的四大组成部分

### 3.1 名称（Name）—— 工具的"身份证"

**默认取自函数名**。模型根据名称来识别和调用工具。

**命名规范**：

```
✅ 推荐：snake_case（下划线分隔）
   - get_weather
   - search_web
   - send_email

❌ 避免：空格或特殊字符
   - get weather     ← 空格可能导致解析问题
   - getWeather      ← camelCase 在某些模型上兼容性差
   - get-weather     ← 连字符可能被误解为运算符
```

**为什么 `snake_case` 最好？**

因为不同模型厂商对名称的解析方式不同。`snake_case` 是**兼容性最强**的格式，确保工具名在所有提供商下都能正确解析。

### 3.2 描述（Description）—— 工具的"说明书"

描述是**模型决定是否使用该工具的关键依据**。

```
描述质量对比：

❌ 差的描述：
   "获取信息"  ← 太模糊，模型不知道什么时候该用

✅ 好的描述：
   "搜索互联网获取最新的新闻、事实和实时信息。
    当你需要 2024 年之后的最新数据时使用此工具。
    不适合用于数学计算或代码生成。"
    
   ↑ 清晰说明了：
   - 用途：搜索新闻和事实
   - 触发条件：需要最新数据
   - 不适用的场景：数学计算
```

**描述的最佳实践**：

- **简洁精准**：不要太长，但要把用途和触发条件说清楚
- **包含使用场景**：告诉模型"在什么情况下应该用这个工具"
- **包含限制**：告诉模型"在什么情况下不应该用"

### 3.3 参数（Parameters）—— 工具的"输入接口"

**必须提供 Python 类型提示**，LangChain 会根据类型提示自动生成输入 Schema（JSON Schema）。

```python
@tool
def search(query: str, max_results: int = 5) -> str:
    """搜索互联网。
    
    Args:
        query: 搜索关键词
        max_results: 最大返回结果数，默认 5
    """
    ...
```

**生成的 JSON Schema**：

```json
{
  "type": "object",
  "properties": {
    "query": {"type": "string", "description": "搜索关键词"},
    "max_results": {"type": "integer", "description": "最大返回结果数，默认 5"}
  },
  "required": ["query"]
}
```

**⚠️ 重要约束：保留字冲突**

参数名 **绝对不能** 使用 `config` 或 `runtime`！这两个是 LangChain 内部保留字，用作业务参数会导致运行时崩溃。

```python
# ❌ 错误：使用了保留字
@tool
def bad_tool(query: str, config: dict) -> str:
    ...

# ✅ 正确：使用其他名称
@tool
def good_tool(query: str, options: dict) -> str:
    ...
```

### 3.4 返回值（Return Values）—— 工具的"输出格式"

工具可以返回三种类型的值，各有不同的用途：

#### 类型 1：`str`（字符串）—— 给模型看的

```python
@tool
def get_weather(location: str) -> str:
    """查询天气。"""
    return f"{location}: 晴天，22°C"
```

**作用**：转为人类可读的 `ToolMessage`，**供模型阅读和推理**。

**适用场景**：工具结果最终需要模型理解并整合到回答中。

#### 类型 2：`object / dict`（结构化数据）—— 给模型精准解析的

```python
@tool
def get_weather(location: str) -> dict:
    """查询天气。"""
    return {
        "city": location,
        "temperature": 22,
        "condition": "晴天",
        "humidity": 45
    }
```

**作用**：返回结构化数据，**供模型精准解析特定字段**。

**适用场景**：下游程序或模型需要提取特定字段（如只取温度数值做计算）。

#### 类型 3：`Command`（命令对象）—— 直接变更 Agent 状态的

```python
from langgraph.types import Command
from langchain.tools import tool

@tool
def update_user_name(new_name: str, runtime) -> Command:
    """更新用户名称。"""
    return Command(
        update={"user_name": new_name},  # 更新 Agent 状态
        graph=Command.PARENT
    )
```

**作用**：直接变更 Agent/Graph 状态，可附带 `ToolMessage` 告知模型执行结果。

**适用场景**：需要修改会话状态或长期记忆，而不只是返回一个结果。

---

## 四、工具的运行时上下文：ToolRuntime

### 4.1 什么是 ToolRuntime？

`ToolRuntime` 是工具的**运行时上下文对象**，它让工具能够访问：
- 当前的对话状态
- 用户身份信息
- 长期记忆存储
- 流式输出通道

**如何在工具中使用**：

在函数签名中添加 `runtime: ToolRuntime` 参数，LangChain 会**自动注入**，并且该参数**对模型隐藏**（不会暴露给 LLM 的 JSON Schema）。

```python
from langchain.tools import tool
from langchain_core.tools import ToolRuntime

@tool
def search(query: str, runtime: ToolRuntime) -> str:
    """搜索互联网。"""
    # 访问短期会话状态（消息历史）
    messages = runtime.state
    
    # 访问不可变的调用时配置（如用户身份）
    user_id = runtime.context.get("user_id")
    
    # 访问长期跨会话持久化存储
    preferences = runtime.store.get(f"user:{user_id}:preferences")
    
    # 执行搜索
    return f"搜索结果（用户 {user_id}）: ..."
```

### 4.2 ToolRuntime 的四大能力

```
┌─────────────────────────────────────────────────────────┐
│                  Tool Runtime Context                    │
│                                                         │
│   Tool Call ──► ToolRuntime                              │
│                      │                                   │
│        ┌─────────────┼─────────────┐                    │
│        ▼             ▼             ▼                    │
│     State        Context       Store                   │
│   (消息历史)    (用户身份)    (长期记忆)                 │
│                                                         │
│   赋能四大能力：                                         │
│   1. Context-Aware Tools（上下文感知）                    │
│   2. Stateful Tools（有状态）                            │
│   3. Memory-Enabled Tools（记忆启用）                    │
│   4. Streaming Tools（流式输出）                         │
└─────────────────────────────────────────────────────────┘
```

#### 能力 1：上下文感知（Context-Aware）

工具可以读取当前的对话上下文，做出更智能的决策：

```python
@tool
def search(query: str, runtime: ToolRuntime) -> str:
    """搜索。"""
    # 读取最近的消息，判断用户意图
    recent_messages = runtime.state["messages"][-3:]
    
    # 如果用户在讨论代码，自动增强搜索的编程相关性
    if any("代码" in msg.content for msg in recent_messages):
        query = f"{query} programming"
    
    return web_search(query)
```

#### 能力 2：有状态（Stateful）

工具可以读写会话状态：

```python
@tool
def set_language(lang: str, runtime: ToolRuntime) -> str:
    """设置用户的语言偏好。"""
    runtime.state["user_language"] = lang
    return f"语言已设置为 {lang}"
```

#### 能力 3：记忆启用（Memory-Enabled）

工具可以访问长期存储，跨会话保留信息：

```python
@tool
def get_user_preferences(runtime: ToolRuntime) -> str:
    """获取用户的偏好设置。"""
    user_id = runtime.context.get("user_id")
    store = runtime.store
    
    # 从长期存储读取
    prefs = store.get(f"user:{user_id}:preferences")
    
    if prefs is None:
        return "用户尚未设置偏好"
    return prefs
```

#### 能力 4：流式输出（Streaming）

对于长耗时任务，工具可以通过 `runtime.stream_writer` 实时反馈进度：

```python
@tool
def generate_report(topic: str, runtime: ToolRuntime) -> str:
    """生成一份详细的报告。"""
    stream = runtime.stream_writer
    
    stream({"status": "开始收集数据..."})
    # ... 数据处理 ...
    
    stream({"status": "正在分析..."})
    # ... 分析 ...
    
    stream({"status": "正在生成报告..."})
    # ... 生成 ...
    
    return "报告生成完成。"
```

**⚠️ 注意**：`runtime.stream_writer` 仅在 **LangGraph 执行图运行时**有效，本地独立调用时无效。

---

## 五、工具的执行机制：ToolNode 与路由

### 5.1 ToolNode —— 工具的执行节点

LangGraph 提供了预置的 `ToolNode`，用于执行工具调用。它原生支持：

- **并行调用**：模型可以同时调用多个工具，`ToolNode` 会并行执行
- **错误拦截**：工具异常时，自动转换为 `ToolMessage` 返回错误信息
- **状态注入**：执行结果自动注入回图状态，供 LLM 继续推理

### 5.2 条件路由：`tools_condition`

配合 `tools_condition` 可以实现智能路由：

```
┌─────────────────────────────────────────────────────────┐
│                    工具路由流程                           │
│                                                         │
│   用户输入 ──► LLM                                      │
│                    │                                     │
│            ┌───────┴───────┐                            │
│            │               │                             │
│     需要工具？         不需要工具？                       │
│            │               │                             │
│            ▼               ▼                             │
│       ToolNode         直接输出答案                       │
│            │                                            │
│            ▼                                            │
│       回到 LLM（基于工具结果继续推理）                     │
└─────────────────────────────────────────────────────────┘
```

**在 StateGraph 中的集成**：

```python
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph

builder = StateGraph(State)

# 注册工具节点
tool_node = ToolNode(tools=[get_weather, search])
builder.add_node("tools", tool_node)

# 添加 LLM 节点
builder.add_node("llm", llm_node)

# 建立条件路由：LLM → tools → LLM 循环
builder.add_conditional_edges(
    "llm",
    tools_condition,
    {True: "tools", False: "__end__"}
)

# 工具执行后回到 LLM
builder.add_edge("tools", "llm")
```

---

## 六、最佳实践与设计哲学

### 6.1 Docstring 和类型提示是必须的

```python
# ❌ 错误：缺少类型提示和 docstring
@tool
def bad_tool(query, max_results):
    result = search(query)
    return result[:max_results]

# ✅ 正确：完整的类型提示和 docstring
@tool
def search_web(query: str, max_results: int = 5) -> str:
    """搜索互联网获取最新信息。
    
    Args:
        query: 搜索关键词（2-10 个词）
        max_results: 最大返回结果数，默认 5
    """
    result = web_search(query)
    return result[:max_results]
```

### 6.2 使用 ToolRuntime 获取上下文，避免硬编码

```python
# ❌ 错误：硬编码用户 ID
@tool
def get_preferences() -> str:
    user_id = "123"  # 硬编码！
    return store.get(f"user:{user_id}:preferences")

# ✅ 正确：使用 runtime.context 获取
@tool
def get_preferences(runtime: ToolRuntime) -> str:
    user_id = runtime.context.get("user_id")
    return runtime.store.get(f"user:{user_id}:preferences")
```

### 6.3 并发冲突处理：定义 Reducer

LLM 可以**并行调用多个工具**。如果多个工具并发修改同一个状态字段，**必须**为该字段定义 Reducer，否则数据会被覆盖。

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    # 使用 add_messages Reducer 安全地追加消息
    messages: Annotated[list, add_messages]
    
    # 如果多个工具可能同时更新这个字段，也需要定义 Reducer
    search_results: Annotated[list, lambda a, b: a + b]
```

### 6.4 生产环境使用持久化存储

```python
# ❌ 开发环境：内存存储（重启后丢失）
from langgraph.store.memory import InMemoryStore
store = InMemoryStore()

# ✅ 生产环境：Postgres 持久化存储
from langgraph.store.postgres import PostgresStore
store = PostgresStore(connection_string="postgresql://...")
```

### 6.5 根据下游需求选择返回类型

| 下游需求 | 返回类型 | 示例场景 |
|----------|----------|----------|
| 模型需要理解并整合到回答 | `str` | 天气查询、翻译 |
| 程序需要提取特定字段计算 | `dict` | 数值计算、数据过滤 |
| 需要修改 Agent 状态 | `Command` | 更新用户偏好、设置标记 |

---

## 七、总结：工具的本质

| 概念 | 比喻 | 作用 |
|------|------|------|
| **工具函数** | 手脚 | 执行实际操作 |
| **名称** | 身份证 | 让模型识别和引用 |
| **描述** | 说明书 | 告诉模型何时使用 |
| **参数** | 输入接口 | 规定如何调用 |
| **返回值** | 输出接口 | 传递执行结果 |
| **ToolRuntime** | 神经系统 | 连接上下文、状态、记忆 |
| **ToolNode** | 执行引擎 | 并行执行、错误处理 |
| **tools_condition** | 路由器 | 决定是否需要工具 |

**Tool 的本质**是**模型与外部世界的桥梁**。它让模型不再局限于训练数据的知识，能够获取实时信息、执行实际操作，从而完成复杂的现实世界任务。

