# LangChain Prebuilt Middleware 核心概念教程


---

## 一、什么是 Prebuilt Middleware？

### 1.1 从"重复造轮子"到"开箱即用"

在开发 Agent 应用时，很多问题是**共性的**：

```
开发者常遇到的问题：
  ❓ "对话太长了，超出上下文窗口怎么办？"
  ❓ "模型调用工具失败了，要不要重试？"
  ❓ "用户给了危险指令，要不要让人审批？"
  ❓ "输出里包含手机号等敏感信息，怎么过滤？"
  ❓ "主模型挂了，有没有备用的？"

解决方案：
  以前：每个开发者自己写代码解决
  现在：LangChain 内置了预构建的中间件，直接拿来用！
```

**核心定义**：内置中间件是专为 LangChain 和 Deep Agents 代理设计的**预构建、生产级拦截模块**。它们直接作用于代理的模型调用与工具执行生命周期，用于解决上下文溢出、成本控制、安全合规、错误恢复、工具筛选等通用工程问题，**开发者无需重复编写样板代码**。

### 1.2 内置中间件的分类

内置中间件分为两大类：

| 类别 | 说明 | 示例 |
|------|------|------|
| **通用型 (Provider-agnostic)** | 适用于所有模型提供商 | Summarization, HITL, Retry, Limits |
| **供应商特定型 (Provider-specific)** | 仅适用于特定厂商的模型 | Anthropic 提示词缓存、OpenAI 内容审核 |

---

## 二、内置中间件全景图

### 2.1 所有内置中间件一览

| 中间件名称 | 核心功能 | 一句话解释 |
|:---|:---|:---|
| **SummarizationMiddleware** | 对话摘要压缩 | 接近 Token 限制时，自动压缩旧对话历史 |
| **HumanInTheLoopMiddleware** | 人工审批 | 暂停执行，等待人类对工具调用进行审批、编辑或拒绝 |
| **ToolCallLimitMiddleware** | 调用次数限制 | 限制模型或工具的调用次数，防死循环与超支 |
| **ModelFallbackMiddleware** | 模型回退 | 主模型失败时，自动切换至备用模型 |
| **PIIDetectionMiddleware** | 敏感信息检测 | 扫描并处理输入/输出中的个人敏感信息 |
| **TodoListMiddleware** | 待办列表 | 赋予代理内置的待办工具，用于多步骤任务规划 |
| **LLMToolSelectorMiddleware** | 工具筛选 | 先用轻量 LLM 筛选最相关的工具，降低 Token 消耗 |
| **ToolRetryMiddleware** | 工具重试 | 捕获异常，对失败的调用执行自动重试 |
| **LLMToolEmulatorMiddleware** | 工具模拟 | 拦截真实工具调用，用 LLM 生成模拟响应（用于测试） |
| **ContextEditingMiddleware** | 上下文编辑 | 超出 Token 阈值时，精准清除历史中的旧工具输出 |
| **ShellToolMiddleware** | Shell 会话 | 暴露持久化 Shell 会话，允许代理安全执行系统命令 |
| **FileSearchMiddleware** | 文件搜索 | 提供文件名匹配和内容正则搜索工具 |
| **FilesystemMiddleware** | 文件系统 | 提供文件读写工具，支持代理将中间结果存入记忆 |
| **SubagentMiddleware** | 子代理 | 允许主代理动态生成子代理，隔离复杂任务的上下文 |

### 2.2 按功能分类

为了更好地理解，我们可以按功能将它们分组：

```
┌─────────────────────────────────────────────────────────┐
│                  内置中间件功能分组                        │
│                                                         │
│  📝 上下文管理：                                         │
│     Summarization, Context Editing                       │
│                                                         │
│  🔒 安全与合规：                                         │
│     Human-in-the-loop, PII Detection, Limits             │
│                                                         │
│  🔄 容错与恢复：                                         │
│     Model Fallback, Tool Retry                           │
│                                                         │
│  🛠️ 工具增强：                                           │
│     Tool Selector, Tool Emulator, To-do List             │
│                                                         │
│  💻 环境模拟：                                           │
│     Shell, Filesystem, File Search                       │
│                                                         │
│  🧩 架构扩展：                                           │
│     Subagent                                             │
│                                                         │
│  🏢 供应商专属：                                         │
│     Anthropic (提示词缓存), OpenAI (内容审核), AWS        │
└─────────────────────────────────────────────────────────┘
```

---

## 三、核心内置中间件详解

### 3.1 SummarizationMiddleware —— 对话摘要压缩

**问题**：长对话会超出模型的上下文窗口。

**解决方案**：当对话接近 Token 限制时，自动用 LLM 总结早期历史并替换原文。

```python
from langchain.agents.middleware import SummarizationMiddleware

summarizer = SummarizationMiddleware(
    model="gpt-4-mini",                        # 用于总结的模型
    trigger=("tokens", 4000),                  # 超过 4000 token 时触发
    keep=("messages", 20),                     # 保留最近 20 条消息
)

agent = create_agent(
    model="gpt-4",
    tools=[search],
    middleware=[summarizer],
)
```

**工作原理**：

```
对话历史：
  [msg1, msg2, msg3, ..., msg50]  ← 总 token 数超过阈值

触发总结：
  1. 用 LLM 总结 msg1 到 msg30 为一段概述
  2. 用概述替换原文
  3. 保留最近 20 条消息（msg31 到 msg50）

结果：
  [AIMessage("早期对话概述..."), msg31, ..., msg50]
```

**触发条件配置**：

| 方式 | 示例 | 说明 |
|------|------|------|
| **Token 数** | `trigger=("tokens", 4000)` | 超过 4000 token 时触发 |
| **消息数** | `trigger=("messages", 50)` | 超过 50 条消息时触发 |
| **比例** | `trigger=("fraction", 0.8)` | 达到上下文窗口 80% 时触发（需 `langchain>=1.1`） |

**保留策略配置**：

| 方式 | 示例 | 说明 |
|------|------|------|
| **消息数** | `keep=("messages", 20)` | 保留最近 20 条 |
| **Token 数** | `keep=("tokens", 2000)` | 保留最近 2000 token |

**⚠️ 废弃参数警告**：`max_tokens_before_summary` 和 `messages_to_keep` 已废弃，必须改用 `trigger` 和 `keep` 元组。

### 3.2 HumanInTheLoopMiddleware —— 人工审批

**问题**：Agent 可能执行危险操作（删除数据库、发送邮件、转账等）。

**解决方案**：在特定工具调用前暂停，等待人类审批。

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

hitl = HumanInTheLoopMiddleware(
    interrupt_on={
        "delete_record": {"allowed_decisions": ["approve", "reject"]},
        "send_email": {"allowed_decisions": ["approve", "edit", "reject"]},
        "transfer_money": False,  # False 表示总是需要审批
    }
)

agent = create_agent(
    model="gpt-4",
    tools=[delete_record, send_email, transfer_money],
    middleware=[hitl],
    checkpointer=checkpointer,  # ⚠️ 必须有！
)

config = {"configurable": {"thread_id": "session-1"}}
```

**工作流程**：

```
Agent 决定调用 delete_record
  │
  ▼
HumanInTheLoopMiddleware 拦截
  │
  ▼
暂停执行，保存状态到 Checkpointer
  │
  ▼
等待人类决策：
  ├── approve（批准）──► 执行工具
  ├── edit（编辑）──► 修改参数后执行
  └── reject（拒绝）──► 返回拒绝消息给模型

⚠️ 必须配置 checkpointer，否则状态无法跨中断保存！
```

**审批模式说明**：

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| **approve** | 直接批准执行 | 低风险操作 |
| **edit** | 编辑参数后执行 | 需要调整参数的场景 |
| **reject** | 拒绝并返回模型 | 高风险操作 |

### 3.3 ToolCallLimitMiddleware —— 调用次数限制

**问题**：Agent 可能陷入无限循环，导致成本失控。

**解决方案**：限制模型或工具的调用次数。

```python
from langchain.agents.middleware import ToolCallLimitMiddleware

limiter = ToolCallLimitMiddleware(
    thread_limit=50,     # 整个线程最多调用 50 次（需 checkpointer）
    run_limit=10,        # 单轮对话最多调用 10 次
    exit_behavior="error",  # 超出限制时的行为："error" 或 "warn"
)

agent = create_agent(
    model="gpt-4",
    tools=[search],
    middleware=[limiter],
    checkpointer=checkpointer,  # thread_limit 需要
)
```

**限制类型**：

| 参数 | 说明 | 是否需要 Checkpointer |
|------|------|----------------------|
| `thread_limit` | 整个线程的累计调用次数 | ✅ 需要 |
| `run_limit` | 单轮对话的调用次数 | ❌ 不需要 |

**超出限制的行为**：

| `exit_behavior` | 行为 | 适用场景 |
|-----------------|------|----------|
| `"error"` | 抛出异常，终止执行 | 生产环境（安全） |
| `"warn"` | 发出警告，继续执行 | 开发调试 |

### 3.4 ModelFallbackMiddleware —— 模型回退

**问题**：主模型可能因为网络、限流或服务中断而失败。

**解决方案**：主模型失败时，自动按顺序切换至备用模型。

```python
from langchain.agents.middleware import ModelFallbackMiddleware

fallback = ModelFallbackMiddleware(
    fallback_models=[
        "gpt-4-mini",       # 第一备用
        "gpt-3.5-turbo",    # 第二备用
    ]
)

agent = create_agent(
    model="gpt-4",          # 主模型
    tools=[search],
    middleware=[fallback],
)
```

**工作流程**：

```
调用 gpt-4 ──► 失败（超时/限流）
  │
  ▼
自动切换到 gpt-4-mini ──► 成功！
  │
  （如果也失败）
  ▼
自动切换到 gpt-3.5-turbo ──► 成功！
```

**适用场景**：

- 生产高可用：应对网络瞬态故障或模型限流
- 成本控制：默认用便宜模型，失败时才用贵模型

### 3.5 PIIDetectionMiddleware —— 敏感信息检测

**问题**：Agent 的输出可能包含个人敏感信息（手机号、身份证、邮箱等）。

**解决方案**：自动扫描并处理敏感信息。

```python
from langchain.agents.middleware import PIIDetectionMiddleware

pii = PIIDetectionMiddleware(
    strategy="redact",     # 处理策略
    entities=["phone", "email", "id_card"],  # 检测的实体类型
)

agent = create_agent(
    model="gpt-4",
    tools=[search],
    middleware=[pii],
)
```

**处理策略**：

| 策略 | 行为 | 示例 |
|------|------|------|
| **block** | 抛出异常，阻断输出 | 适用于严格合规场景 |
| **redact** | 替换为 `[REDACTED]` | "手机号 [REDACTED]" |
| **mask** | 部分掩码 | "138****8000" |
| **hash** | 哈希化 | "a1b2c3d4..." |

**自定义检测规则**：

支持传入正则字符串、编译后的 Regex 对象，或自定义函数：

```python
import re

pii = PIIDetectionMiddleware(
    strategy="redact",
    custom_patterns=[
        r"\b\d{4}-\d{4}-\d{4}-\d{4}\b",  # 信用卡号
        re.compile(r"\b\d{17}[\dXx]\b"),   # 身份证号
    ],
)

# 或使用自定义函数
def custom_detector(text):
    """返回 [{text, start, end}] 格式的匹配结果。"""
    matches = []
    # 自定义检测逻辑...
    return matches

pii = PIIDetectionMiddleware(
    strategy="mask",
    custom_detector=custom_detector,
)
```

### 3.6 TodoListMiddleware —— 待办列表

**问题**：Agent 处理复杂多步骤任务时，容易迷失或遗漏步骤。

**解决方案**：赋予代理内置的 `write_todos` 工具，用于规划和追踪任务。

```python
from langchain.agents.middleware import TodoListMiddleware

agent = create_agent(
    model="gpt-4",
    tools=[search, calculate],
    middleware=[TodoListMiddleware()],
)
```

**Agent 如何使用**：

```
用户："帮我调研三家竞争对手的优缺点，并写一份报告。"

Agent 自主规划：
  1. write_todos([
       {"content": "调研公司 A", "status": "pending"},
       {"content": "调研公司 B", "status": "pending"},
       {"content": "调研公司 C", "status": "pending"},
       {"content": "撰写报告", "status": "pending"},
     ])
  2. 执行每个 todo
  3. 完成后标记 status 为 "completed"
```

**适用场景**：

- 长周期对话：配合 Summarization 管理复杂多步任务
- 项目调研：多步骤数据收集和分析
- 自动化流程：需要明确步骤和进度的任务

### 3.7 LLMToolSelectorMiddleware —— 工具筛选

**问题**：当工具池很大（>10 个）时，把所有工具描述发给模型会消耗大量 Token。

**解决方案**：先用轻量、便宜的 LLM 筛选出最相关的工具，再传给主模型。

```python
from langchain.agents.middleware import LLMToolSelectorMiddleware

selector = LLMToolSelectorMiddleware(
    model="gpt-4-mini",           # 轻量筛选模型
    always_include=["search"],    # 始终包含的工具
    max_tools=5,                  # 最多筛选 5 个工具
)

agent = create_agent(
    model="gpt-4",
    tools=[search, weather, email, calendar, calculator, ...],  # 20+ 工具
    middleware=[selector],
)
```

**工作流程**：

```
用户输入 ──► 轻量模型筛选器 ──► 选出 5 个最相关的工具
                                      │
                                      ▼
                                主模型 + 筛选后的工具 ──► 执行
```

**好处**：

- **降低 Token 消耗**：不需要把所有工具描述都发给主模型
- **提高准确性**：减少不相关工具的干扰，模型更专注

### 3.8 ToolRetryMiddleware —— 工具重试

**问题**：工具调用可能因为网络、API 限流等原因失败。

**解决方案**：捕获特定异常，对失败的调用执行带指数退避的自动重试。

```python
from langchain.agents.middleware import ToolRetryMiddleware

retry = ToolRetryMiddleware(
    max_retries=3,                # 最多重试 3 次
    initial_delay=1,              # 初始等待 1 秒
    backoff_factor=2,             # 每次等待时间翻倍
    retry_on=(TimeoutError, ConnectionError),  # 只重试这些异常
)

agent = create_agent(
    model="gpt-4",
    tools=[search],
    middleware=[retry],
)
```

**重试策略**：

```
第 1 次失败 ──► 等待 1 秒 ──► 重试
第 2 次失败 ──► 等待 2 秒 ──► 重试
第 3 次失败 ──► 等待 4 秒 ──► 重试
第 4 次失败 ──► 放弃，返回错误

等待时间公式：initial_delay * (backoff_factor ** retry_count)
```

**指数退避 + 抖动**：

```python
retry = ToolRetryMiddleware(
    max_retries=3,
    initial_delay=1,
    backoff_factor=2,
    jitter=True,  # 添加随机抖动，避免多个请求同时重试造成雪崩
)
```

### 3.9 LLMToolEmulatorMiddleware —— 工具模拟

**问题**：开发测试时，外部依赖（API、数据库）可能未就绪或调用成本高。

**解决方案**：拦截真实工具调用，用 LLM 生成模拟响应。

```python
from langchain.agents.middleware import LLMToolEmulatorMiddleware

emulator = LLMToolEmulatorMiddleware(
    model="gpt-4-mini",          # 用于模拟的模型
    emulate_tools=["search", "weather"],  # 需要模拟的工具
)

agent = create_agent(
    model="gpt-4",
    tools=[search, weather],
    middleware=[emulator],
)
```

**工作流程**：

```
Agent 决定调用 search("北京天气")
  │
  ▼
Emulator 拦截
  │
  ▼
用 LLM 生成模拟响应："北京现在晴天，22°C"
  │
  ▼
返回模拟的 ToolMessage（不调用真实 API）
```

**适用场景**：

- 开发与测试：外部依赖未就绪时验证 Agent 逻辑
- 原型开发：快速验证流程，无需配置真实 API
- 演示环境：避免演示时调用真实服务

### 3.10 ContextEditingMiddleware —— 上下文编辑

**问题**：当对话超出 Token 阈值时，需要清理历史中的旧工具输出。

**解决方案**：精准清除历史消息中的旧 `ToolMessage`，保留核心对话。

```python
from langchain.agents.middleware import ContextEditingMiddleware

context_editor = ContextEditingMiddleware(
    trigger=("tokens", 8000),     # 超过 8000 token 时触发
    strategy="remove_tool_calls", # 清除工具调用和结果
)

agent = create_agent(
    model="gpt-4",
    tools=[search],
    middleware=[context_editor],
)
```

**与 Summarization 的区别**：

| 特性 | Summarization | Context Editing |
|------|---------------|-----------------|
| **处理方式** | 用 LLM 总结早期历史 | 直接删除旧工具输出 |
| **成本** | 需要调用 LLM 总结 | 不需要额外调用 |
| **信息保留** | 保留摘要概要 | 丢失被删除的内容 |
| **适用场景** | 需要保留上下文语义 | 只需要清理工具噪音 |

---

## 四、环境模拟类中间件

### 4.1 ShellTool Middleware —— Shell 会话

**作用**：暴露持久化 Shell 会话，允许代理安全地执行系统命令。

```python
from langchain.agents.middleware import ShellToolMiddleware

shell = ShellToolMiddleware(
    execution_policy="host",  # 执行策略
    timeout=30,               # 超时时间（秒）
)

agent = create_agent(
    model="gpt-4",
    middleware=[shell],
)
```

**执行策略**：

| 策略 | 说明 | 安全性 |
|------|------|--------|
| `HostExecutionPolicy` | 直接在宿主机执行 | ⚠️ 完全权限，有风险 |
| `DockerExecutionPolicy` | 在 Docker 容器内执行 | ✅ 隔离，更安全 |

**⚠️ 安全警告**：

- `HostExecutionPolicy` 提供宿主机完全权限
- `redaction_rules` 仅在命令执行**后**过滤输出，**无法**防止中间过程的数据外泄
- **生产环境务必使用 Docker 或沙箱策略**
- 持久化 Shell **目前不支持**与 `Human-in-the-loop` 的 interrupts 共用

### 4.2 FileSearch Middleware —— 文件搜索

**作用**：提供文件搜索工具，支持文件名匹配和内容正则搜索。

```python
from langchain.agents.middleware import FileSearchMiddleware

file_search = FileSearchMiddleware(
    root_dir="/workspace/project",  # 搜索根目录
    max_results=20,                 # 最大返回结果数
)

agent = create_agent(
    model="gpt-4",
    middleware=[file_search],
)
```

**提供的工具**：

| 工具 | 功能 | 示例 |
|------|------|------|
| `glob_search` | 文件名匹配 | `"**/*.py"` → 找到所有 Python 文件 |
| `grep_search` | 内容正则搜索 | `"def .*"` → 找到所有函数定义 |

### 4.3 Filesystem Middleware —— 文件系统

**作用**：提供文件读写工具，支持代理将中间结果存入短/长期记忆。

```python
from langchain.agents.middleware import FilesystemMiddleware

fs = FilesystemMiddleware(
    root_dir="/workspace/data",
    backend="state",  # 默认内存，可配持久化
)

agent = create_agent(
    model="gpt-4",
    middleware=[fs],
)
```

**后端配置**：

| 后端 | 说明 | 适用场景 |
|------|------|----------|
| `StateBackend` | 内存存储（默认） | 开发/短期任务 |
| `StoreBackend` | 持久化存储 | 跨线程长期记忆 |
| `CompositeBackend` | 组合路由 | 将特定路径路由至持久化 |

```python
# CompositeBackend 示例：将 /memories/ 路由至持久化
from langgraph.store.composite import CompositeBackend

fs = FilesystemMiddleware(
    backend=CompositeBackend(
        routes={
            "/memories/": StoreBackend(...),
            "/tmp/": StateBackend(),
        }
    ),
)
```

---

## 五、架构扩展类中间件

### 5.1 SubagentMiddleware —— 子代理

**作用**：允许主代理动态生成子代理，隔离复杂任务的上下文。

```python
from langchain.agents.middleware import SubagentMiddleware
from langchain.agents import create_agent

# 定义子代理
research_agent = create_agent(
    model="gpt-4-mini",
    tools=[search],
    system_prompt="你是一个专业的调研助手。"
)

subagent = SubagentMiddleware(
    subagents={"research": research_agent},
)

agent = create_agent(
    model="gpt-4",
    tools=[...],
    middleware=[subagent],
)
```

**工作流程**：

```
主 Agent: "我需要调研三家公司的信息"
  │
  ▼
创建子代理: research_agent
  │
  ▼
子代理独立执行调研任务（上下文隔离）
  │
  ▼
返回结果给主 Agent
  │
  ▼
主 Agent 整合并输出
```

**好处**：

- **上下文隔离**：子代理的对话不会污染主代理的历史
- **并行处理**：多个子代理可以同时执行
- **成本优化**：子代理可以用更便宜的模型

---

## 六、供应商专属中间件

### 6.1 Anthropic 专属

| 中间件 | 功能 |
|--------|------|
| **提示词缓存** | 利用 Anthropic 的 prompt caching 功能，降低重复处理的延迟和成本 |

```python
from langchain.agents.middleware import AnthropicPromptCachingMiddleware

anthropic = AnthropicPromptCachingMiddleware(
    cache_prefix=["system_prompt"],  # 需要缓存的提示词前缀
)
```

### 6.2 OpenAI 专属

| 中间件 | 功能 |
|--------|------|
| **内容审核** | 利用 OpenAI 的 Moderation API 检测有害内容 |

```python
from langchain.agents.middleware import OpenAIModerationMiddleware

moderation = OpenAIModerationMiddleware(
    action="block",  # 检测到有害内容时阻断
)
```

### 6.3 AWS 专属

| 中间件 | 功能 |
|--------|------|
| **Bash/文本编辑器集成** | 与 AWS 的 Bedrock 和 SageMaker 工具链集成 |

---

## 七、组合使用：生产级 Agent 配置

### 7.1 高可用生产环境示例

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    HumanInTheLoopMiddleware,
    ToolCallLimitMiddleware,
    ModelFallbackMiddleware,
    ToolRetryMiddleware,
    PIIDetectionMiddleware,
    TodoListMiddleware,
)
from langgraph.checkpoint.postgres import PostgresSaver

# 1. 配置持久化
checkpointer = PostgresSaver(connection_string="postgresql://...")

# 2. 组合中间件
middleware = [
    SummarizationMiddleware(
        model="gpt-4-mini",
        trigger=("tokens", 4000),
        keep=("messages", 20),
    ),
    HumanInTheLoopMiddleware(
        interrupt_on={"delete_record": False, "send_email": False}
    ),
    ToolCallLimitMiddleware(
        thread_limit=50,
        run_limit=10,
        exit_behavior="error",
    ),
    ModelFallbackMiddleware(
        fallback_models=["gpt-4-mini", "gpt-3.5-turbo"],
    ),
    ToolRetryMiddleware(
        max_retries=3,
        initial_delay=1,
        backoff_factor=2,
        jitter=True,
    ),
    PIIDetectionMiddleware(
        strategy="redact",
    ),
    TodoListMiddleware(),
]

# 3. 创建 Agent
agent = create_agent(
    model="gpt-4",
    tools=[search, delete_record, send_email],
    middleware=middleware,
    checkpointer=checkpointer,
)
```

### 7.2 开发与测试环境示例

```python
from langchain.agents.middleware import (
    LLMToolEmulatorMiddleware,
    LLMToolSelectorMiddleware,
)

middleware = [
    LLMToolEmulatorMiddleware(
        model="gpt-4-mini",
        emulate_tools=["search", "weather"],
    ),
    LLMToolSelectorMiddleware(
        model="gpt-4-mini",
        always_include=["search"],
        max_tools=5,
    ),
]

agent = create_agent(
    model="gpt-4",
    tools=[search, weather, email, calendar, calculator, ...],
    middleware=middleware,
)
```

---

## 八、最佳实践与设计哲学

### 8.1 长周期对话

**组合**：`Summarization` + `Context Editing` + `To-do List`

```python
middleware = [
    SummarizationMiddleware(...),
    ContextEditingMiddleware(...),
    TodoListMiddleware(),
]
```

### 8.2 生产高可用

**组合**：`Model Fallback` + `Tool Retry` + `Limits`

```python
middleware = [
    ModelFallbackMiddleware(...),
    ToolRetryMiddleware(...),
    ToolCallLimitMiddleware(...),
]
```

### 8.3 安全与合规

**金融/医疗场景必开**：`PII Detection`

**数据库写入或资金操作必挂**：`Human-in-the-loop`

```python
middleware = [
    PIIDetectionMiddleware(strategy="block"),
    HumanInTheLoopMiddleware(interrupt_on={"transfer_money": False}),
]
```

### 8.4 工具路由优化

**工具池 >10 个时启用**：`LLM Tool Selector`

**设置 `always_include`**：保证核心工具必选

```python
middleware = [
    LLMToolSelectorMiddleware(
        always_include=["search", "calculator"],
        max_tools=5,
    ),
]
```

### 8.5 开发与测试

**外部依赖未就绪时**：使用 `LLM Tool Emulator` 模拟数据流

```python
middleware = [
    LLMToolEmulatorMiddleware(emulate_tools=["search", "weather"]),
]
```

---

## 九、重要的技术细节和常见错误

### 9.1 ⚠️ Checkpointer 强依赖

`HumanInTheLoopMiddleware` 和 `ToolCallLimitMiddleware`（使用 `thread_limit` 时）**必须**配置 `checkpointer`，否则状态无法跨中断保存，会导致功能失效。

```python
# ✅ 正确
agent = create_agent(
    middleware=[HumanInTheLoopMiddleware(...)],
    checkpointer=MemorySaver(),  # 必须有！
)

# ❌ 错误
agent = create_agent(
    middleware=[HumanInTheLoopMiddleware(...)],
    # 没有 checkpointer ──► 功能失效
)
```

### 9.2 ⚠️ 废弃参数警告

`SummarizationMiddleware` 的 `max_tokens_before_summary` 和 `messages_to_keep` 已废弃，必须改用 `trigger` 和 `keep` 元组。

```python
# ✅ 正确
SummarizationMiddleware(
    trigger=("tokens", 4000),
    keep=("messages", 20),
)

# ❌ 错误（已废弃）
SummarizationMiddleware(
    max_tokens_before_summary=4000,
    messages_to_keep=20,
)
```

### 9.3 ⚠️ Fraction 触发限制

使用 `trigger=("fraction", 0.8)` 依赖 `langchain>=1.1` 的模型 profile 数据。若模型无 profile，需手动指定或改用 `tokens`/`messages`。

### 9.4 ⚠️ Shell 工具与 HITL 不兼容

持久化 Shell **目前不支持**与 `Human-in-the-loop` 的 interrupts 共用。

### 9.5 ⚠️ 安全风险

`HostExecutionPolicy` 提供宿主机完全权限。`redaction_rules` 仅在命令执行**后**过滤输出，**无法**防止中间过程的数据外泄。**生产环境务必使用 Docker 或沙箱策略**。

---

## 十、总结：内置中间件的本质

| 类别 | 中间件 | 比喻 | 作用 |
|------|--------|------|------|
| **上下文管理** | Summarization | 速记员 | 压缩旧对话 |
| **上下文管理** | Context Editing | 清洁工 | 清除工具噪音 |
| **安全合规** | HITL | 审批人 | 人工批准危险操作 |
| **安全合规** | PII Detection | 安检仪 | 检测敏感信息 |
| **安全合规** | Limits | 预算员 | 限制调用次数 |
| **容错恢复** | Fallback | 备胎 | 模型失败时切换 |
| **容错恢复** | Retry | 重试按钮 | 失败后重新尝试 |
| **工具增强** | Selector | 调度员 | 筛选相关工具 |
| **工具增强** | Emulator | 模拟器 | 模拟工具响应 |
| **工具增强** | To-do List | 记事本 | 规划多步骤任务 |
| **环境模拟** | Shell | 终端 | 执行系统命令 |
| **环境模拟** | Filesystem | 文件柜 | 读写文件 |
| **架构扩展** | Subagent | 分包商 | 动态生成子代理 |

**内置中间件的本质**是**生产级的预构建拦截器**。它们解决了 Agent 开发中的共性问题——上下文管理、安全合规、容错恢复、工具优化——让你无需重复造轮子，专注于业务逻辑。

---
