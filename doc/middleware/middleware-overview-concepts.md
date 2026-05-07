# LangChain Middleware Overview 核心概念教程

---

## 一、什么是 Middleware？

### 1.1 从生活场景说起

想象你在经营一家餐厅：

```
没有中间件的 Agent:
  厨师（模型）直接做菜 → 端给客户
  
  问题：
  - 没人检查菜品质量
  - 没人控制出菜速度
  - 没人拦截不安全的食材

有中间件的 Agent:
  厨师（模型）做菜 → 质检员检查（before_model）→ 端菜 → 服务员确认（after_model）
                    ↓
              发现问题？退回重做！
```

**核心定义**：Middleware（中间件）是一种用于在 Agent 执行的每一步进行**精确控制和定制化处理**的机制。它允许开发者在**不修改核心业务逻辑**的前提下，紧密干预 Agent 的运行时行为。

### 1.2 中间件的本质

中间件是 Agent 的**非侵入式运行时插件**。它不替代 Agent 的决策逻辑，而是作为"**拦截器**"附加在 Agent 内部。

```
中间件的特点：
  ✅ 非侵入式：不需要修改 Agent 的核心代码
  ✅ 插件化：可以随意添加、移除、替换
  ✅ 精确控制：在特定的执行节点介入
  ✅ 可组合：多个中间件可以串联使用
```

---

## 二、中间件的工作原理

### 2.1 核心 Agent 循环

理解中间件之前，先理解 Agent 的核心执行循环：

```
┌─────────────────────────────────────────────────────────┐
│                    核心 Agent 循环                        │
│                                                         │
│   用户输入 ──► ┌──────────┐                              │
│                │  模型调用 │ ← 模型推理，决定是否调用工具  │
│                └─────┬────┘                              │
│                      │                                    │
│              ┌───────▼───────┐                           │
│              │ 需要调用工具？ │                           │
│              └───┬───────┬───┘                           │
│                  │ 是     │ 否                            │
│                  │        │                               │
│            ┌─────▼┐   ┌──▼──────┐                        │
│            │执行工具│   │输出答案 │                        │
│            └─────┬┘   └─────────┘                        │
│                  │                                        │
│            ┌─────▼────┐                                  │
│            │观察结果   │── 回到模型调用                     │
│            └──────────┘                                  │
└─────────────────────────────────────────────────────────┘
```

### 2.2 中间件如何介入？

中间件通过在核心循环的每个步骤**之前和之后**暴露钩子（hooks）来拦截、修改或增强数据流与控制流。

```
┌─────────────────────────────────────────────────────────┐
│                    中间件工作流程                          │
│                                                         │
│   用户输入                                               │
│      │                                                  │
│      ▼                                                  │
│  ┌──────────────┐                                       │
│  │ before_model │  ← 中间件：模型推理前介入               │
│  └──────┬───────┘                                       │
│         │                                               │
│         ▼                                               │
│  ┌──────────┐                                           │
│  │  模型调用 │                                           │
│  └────┬─────┘                                           │
│       │                                                 │
│       ▼                                                 │
│  ┌──────────────┐                                       │
│  │ after_model  │  ← 中间件：模型推理后介入               │
│  └──────┬───────┘                                       │
│         │                                               │
│         ▼                                               │
│  ┌──────────────┐                                       │
│  │ wrap_tool_call│ ← 中间件：工具调用前后介入             │
│  └──────┬───────┘                                       │
│         │                                               │
│         ▼                                               │
│      输出 / 回到循环                                     │
└─────────────────────────────────────────────────────────┘
```

### 2.3 中间件的执行顺序

当有多个中间件时，它们按**注册的顺序**依次执行：

```
middleware=[middleware_A, middleware_B, middleware_C]

执行流程：
  before_model: A → B → C → 模型调用
  after_model:  模型输出 → C → B → A → 输出
```

**类比理解**：

```
工厂流水线上的质检环节：

原料进入 → A检查 → B检查 → C检查 → 加工 → C检查 → B检查 → A检查 → 成品
```

---

## 三、中间件的类型

### 3.1 高级中间件（开箱即用）

LangChain 提供了可以直接实例化使用的高级中间件：

#### SummarizationMiddleware（摘要中间件）

**作用**：当对话历史过长时，自动用 LLM 总结早期消息，避免超出上下文窗口。

```python
from langchain.agents.middleware import SummarizationMiddleware

summarizer = SummarizationMiddleware(
    model="anthropic:claude-sonnet-4-5",
    max_tokens_before_summary=5000,  # 超过 5000 token 时触发总结
    max_messages_to_keep=5           # 保留最近 5 条消息
)

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[search],
    middleware=[summarizer],
)
```

**工作流程**：

```
对话历史超过阈值？
  ├── 否 ──► 保持原样
  └── 是
       │
       ▼
  ┌──────────┐
  │  LLM 总结 │  总结早期历史为简洁的概述
  └────┬────┘
       │
       ▼
  用总结替换早期消息 ──► 继续对话
```

#### HumanInTheLoopMiddleware（人工介入中间件）

**作用**：在特定工具调用前暂停，等待人类审批。

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware

hitl = HumanInTheLoopMiddleware(
    interrupt_on={"delete_record": True, "send_email": True}  # 这些工具需要审批
)

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[delete_record, send_email],
    middleware=[hitl],
    checkpointer=checkpointer,  # 需要 checkpointer 支持
)
```

**工作流程**：

```
Agent 决定调用 delete_record
  │
  ▼
HumanInTheLoopMiddleware 拦截
  │
  ▼
暂停执行，等待人类审批
  │
  ├── 批准 ──► 执行工具
  └── 拒绝 ──► 返回拒绝消息给模型
```

### 3.2 自定义中间件（通过钩子）

你可以通过底层钩子创建自定义中间件：

#### @before_model —— 模型推理前处理

在模型推理**之前**拦截消息，适合做裁剪、注入动态提示词。

```python
from langchain.agents.middleware import before_model

@before_model
def inject_dynamic_prompt(state):
    """在模型推理前注入动态系统提示。"""
    user_role = state.get("user_role", "default")
    
    if user_role == "admin":
        dynamic_prompt = SystemMessage("你是管理员助手，可以访问所有数据。")
    else:
        dynamic_prompt = SystemMessage("你是普通用户助手。")
    
    return {
        "messages": [dynamic_prompt] + state["messages"]
    }
```

#### @after_model —— 模型推理后处理

在模型推理**之后**拦截输出，适合做验证、清理、日志记录。

```python
from langchain.agents.middleware import after_model

@after_model
def validate_output(state):
    """验证模型输出是否符合安全要求。"""
    last_message = state["messages"][-1]
    
    # 检查是否包含敏感信息
    if "password" in last_message.content.lower():
        return {
            "messages": [
                {"role": "assistant", "content": "抱歉，我无法提供密码信息。"}
            ]
        }
    
    return state
```

#### @wrap_tool_call —— 工具调用包裹

在工具调用**前后**介入，适合做错误处理、重试、日志记录。

```python
from langchain.agents.middleware import wrap_tool_call

@wrap_tool_call
def handle_tool_errors(state, tool_call, runtime):
    """处理工具调用异常，自动重试。"""
    try:
        result = tool_call.invoke()
        return result
    except Exception as e:
        # 返回错误消息给模型
        return f"工具调用失败：{str(e)}。请重试或换其他方式。"
```

#### @dynamic_prompt —— 动态提示词生成

动态生成系统提示词，适合根据运行时状态调整模型行为。

```python
from langchain.agents.middleware import dynamic_prompt

@dynamic_prompt
def generate_prompt(state):
    """根据用户身份动态生成提示词。"""
    user_id = state.get("user_id", "unknown")
    
    if user_id == "admin":
        return SystemMessage("你是管理员助手，可以执行删除、修改等操作。")
    else:
        return SystemMessage("你是普通用户助手，只能查询信息。")
```

---

## 四、中间件的使用场景

### 4.1 行为追踪

用于日志记录、数据分析和运行时调试。

```python
from langchain.agents.middleware import before_model, after_model
import logging

@before_model
def log_input(state):
    """记录模型输入。"""
    logging.info(f"收到用户输入: {state['messages'][-1].content}")
    return state

@after_model
def log_output(state):
    """记录模型输出。"""
    logging.info(f"模型回复: {state['messages'][-1].content}")
    return state
```

### 4.2 数据转换

动态修改 Prompt、干预工具选择及格式化模型输出。

```python
@before_model
def translate_input(state):
    """将非中文输入翻译为中文。"""
    last_msg = state["messages"][-1]
    if not is_chinese(last_msg.content):
        # 插入翻译指令
        translated = translate_to_chinese(last_msg.content)
        state["messages"][-1].content = translated
    return state
```

### 4.3 流程控制

添加重试机制、模型/工具回退以及提前终止逻辑。

```python
@wrap_tool_call
def retry_on_failure(state, tool_call, runtime):
    """工具调用失败时自动重试。"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return tool_call.invoke()
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(1 * (attempt + 1))  # 指数退避
```

### 4.4 安全与限制

应用调用速率限制、安全护栏和 PII（个人敏感信息）检测。

```python
@after_model
def detect_pii(state):
    """检测并过滤个人敏感信息。"""
    last_message = state["messages"][-1]
    content = last_message.content
    
    # 简单的 PII 检测
    import re
    phone_pattern = r'\b1[3-9]\d{9}\b'
    id_pattern = r'\b\d{17}[\dXx]\b'
    
    content = re.sub(phone_pattern, "[手机号码已隐藏]", content)
    content = re.sub(id_pattern, "[身份证号已隐藏]", content)
    
    state["messages"][-1].content = content
    return state
```

---

## 五、中间件与 Agent 的关系

### 5.1 集成方式

通过 `create_agent` 函数的 `middleware` 参数注入：

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    HumanInTheLoopMiddleware,
)

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[search, delete_record],
    middleware=[
        SummarizationMiddleware(...),      # 消息摘要
        HumanInTheLoopMiddleware(...),     # 人工介入
    ],
)
```

### 5.2 中间件在 Agent 架构中的位置

```
┌─────────────────────────────────────────────────────────┐
│                    Agent 架构全貌                          │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │                Middleware 层                       │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐              │  │
│  │  │ Summarize│ │  HITL   │ │ Custom  │  ...         │  │
│  │  └─────────┘ └─────────┘ └─────────┘              │  │
│  └───────────────────────────────────────────────────┘  │
│                          │                              │
│  ┌───────────────────────────────────────────────────┐  │
│  │                核心 Agent 层                        │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │  │
│  │  │  模型     │  │  工具    │  │  状态    │         │  │
│  │  └──────────┘  └──────────┘  └──────────┘         │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │               Checkpointer 层                      │  │
│  │         （状态持久化，支持中断恢复）                  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 5.3 中间件不替代什么？

```
中间件不是：
  ❌ 不是工具的替代品 —— 工具负责执行外部操作
  ❌ 不是模型的替代品 —— 模型负责推理和决策
  ❌ 不是提示词的替代品 —— 提示词定义模型的角色和行为

中间件是：
  ✅ 是 Agent 执行流程的"拦截器"和"增强器"
  ✅ 提供横切关注点（Cross-cutting concerns）的统一处理
```

---

## 六、最佳实践与设计哲学

### 6.1 统一注册中间件

在初始化 Agent 时，将所有中间件实例打包成列表，统一传入配置：

```python
# ✅ 推荐：统一注册
agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[search],
    middleware=[
        SummarizationMiddleware(...),
        HumanInTheLoopMiddleware(...),
        my_custom_middleware,
    ],
)

# ❌ 不推荐：分散注册
agent = create_agent(...)
agent.add_middleware(...)  # 没有这种 API！
```

### 6.2 中间件的顺序很重要

中间件按注册顺序依次执行，**先注册的 before_model 先执行，后注册的 after_model 先执行**（栈式）。

```python
middleware=[A, B, C]

执行顺序：
  before: A → B → C → 模型
  after:  模型 → C → B → A → 输出
```

**设计建议**：把"最外层"的逻辑先注册（如日志记录），把"最内层"的逻辑后注册（如安全检测）。

### 6.3 HumanInTheLoopMiddleware 需要 Checkpointer

人工介入中间件需要 `checkpointer` 和 `thread_id` 支持，因为需要暂停和恢复状态：

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[delete_record],
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"delete_record": True})],
    checkpointer=checkpointer,  # 必须有！
)

config = {"configurable": {"thread_id": "session-1"}}
```

### 6.4 中间件的职责单一原则

每个中间件应该只做一件事：

```python
# ✅ 好的中间件：职责单一
@before_model
def trim_messages(state):
    """只负责裁剪消息。"""
    ...

@after_model
def detect_pii(state):
    """只负责 PII 检测。"""
    ...

# ❌ 坏的中间件：职责过多
@before_model
def do_everything(state):
    """又裁剪、又翻译、又检测..."""
    ...
```

---

## 七、总结：中间件的本质

| 概念 | 比喻 | 作用 |
|------|------|------|
| **Middleware** | 质检员 | 在执行流程中介入控制 |
| **before_model** | 事前检查 | 模型推理前拦截 |
| **after_model** | 事后确认 | 模型推理后拦截 |
| **wrap_tool_call** | 工具包装器 | 工具调用前后拦截 |
| **dynamic_prompt** | 动态剧本 | 运行时生成提示词 |
| **SummarizationMiddleware** | 速记员 | 压缩对话历史 |
| **HumanInTheLoopMiddleware** | 审批人 | 等待人类批准 |

**中间件的本质**是**非侵入式的运行时插件**。它通过在 Agent 核心循环的关键节点暴露钩子，让你能够在不修改核心代码的前提下，精确控制 Agent 的行为——从日志记录、安全检测到人工审批，一切皆可通过中间件实现。

---
