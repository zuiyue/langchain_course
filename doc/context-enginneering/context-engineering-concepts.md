# LangChain Context Engineering 核心概念教程

---

## 一、什么是 Context Engineering？

### 1.1 从"模型很强但表现很差"说起

想象一个场景：

```
你问一个顶尖专家（强模型）：
  "帮我修好这台机器。"

专家失败了。为什么？
  ❌ 没有机器的型号信息
  ❌ 没有故障现象描述
  ❌ 没有维修手册
  ❌ 没有可用的工具清单

不是专家能力不足，而是缺乏**上下文**！

同样的专家，有了上下文：
  "帮我修好这台机器。型号是 X-200，故障是发出异常噪音。
   这是维修手册第 3 章。可用工具有：扳手、螺丝刀、万用表。"

专家立刻解决问题。
```

**核心定义**：Context Engineering（上下文工程）是指以**正确的格式**为 LLM 提供**正确的信息和工具**，使其能够成功完成特定任务。这是 AI 工程师的首要工作，也是 LangChain 智能体（Agent）抽象设计的核心目标。

### 1.2 为什么智能体会失败？

```
智能体失败的原因：
  ❌ 10% 是因为底层 LLM 能力不足
  ✅ 90% 是因为没有向 LLM 传递"正确的"上下文

缺乏合适的上下文 = 构建可靠智能体的最大障碍
```

**关键认知**：大多数 Agent 问题不是模型不够强，而是上下文管理不到位。

---

## 二、Context Engineering 的定位

### 2.1 概念层 vs 工程层

Context Engineering 是一个**综合性的上位概念**，它不是单一技术，而是多种技术的统一：

```
Context Engineering 包含：
  ├── Prompt Engineering（提示词工程）
  ├── Memory Management（记忆管理）
  ├── RAG（检索增强生成）
  ├── Tool Calling（工具调用）
  ├── State Management（状态管理）
  └── Middleware（中间件）

但它 ≠ 其中任何一个
  它是这些技术的"编排者"
```

### 2.2 它和 Prompt Engineering 的区别

很多人会把 Context Engineering 简化为 Prompt Engineering，但其实不够：

```
Prompt Engineering:
  关注："怎么写提示词让模型表现更好"
  范围：单次提示词的内容和结构

Context Engineering:
  关注："整个上下文系统怎么设计"
  范围：给什么、为什么给、给多少、什么时候给、如何压缩和维护

关系：
  Prompt Engineering ⊂ Context Engineering
  （提示词工程是上下文工程的一个子集）
```

### 2.3 层次理解

| 层次 | 回答的问题 | 示例 |
|------|-----------|------|
| **概念层** | "模型需要什么上下文，才能在当前任务里表现最好？" | 模型需要用户身份、对话历史、可用工具 |
| **工程层** | "这些上下文怎么构建、筛选、压缩、注入、维护、验证？" | 如何从数据库读取用户偏好、如何裁剪超长对话、如何动态过滤工具 |

---

## 三、核心 Agent 循环

理解上下文工程之前，先理解 Agent 的核心执行循环：

```
┌─────────────────────────────────────────────────────────┐
│                    核心 Agent 循环                        │
│                                                         │
│   用户输入 ──► ┌──────────┐                              │
│                │ Model    │  调用 LLM（传入提示词和工具）   │
│                │ call     │  返回响应或工具执行请求         │
│                └─────┬────┘                              │
│                      │                                    │
│              ┌───────▼───────┐                           │
│              │ 需要调用工具？ │                           │
│              └───┬───────┬───┘                           │
│                  │ 是     │ 否                            │
│                  │        │                               │
│            ┌─────▼┐   ┌──▼──────┐                        │
│            │Tool   │   │输出答案 │                        │
│            │execution│  └─────────┘                        │
│            │(执行工具)│                                    │
│            └─────┬┘                                        │
│                  │                                         │
│            ┌─────▼────┐                                   │
│            │工具结果   │── 回到 Model call                 │
│            └──────────┘                                   │
└─────────────────────────────────────────────────────────┘
```

**上下文工程就是在循环的每个步骤中，决定传入什么信息、何时传入、如何更新。**

---

## 四、三大控制维度

上下文管理主要分为三大控制维度，各自依赖不同的数据源：

### 4.1 维度一：Model Context（模型上下文）

**性质**：**瞬态（Transient）**，仅影响单次调用。

**控制内容**：

| 内容 | 说明 | 示例 |
|------|------|------|
| **系统提示词** | 定义模型的角色和行为准则 | `"你是客服助手"` |
| **消息历史** | 当前的对话上下文 | `[User: "你好", AI: "你好！"]` |
| **可用工具** | 模型可以选择调用的工具列表 | `[search, weather, email]` |
| **模型选择** | 使用哪个 LLM | `gpt-4` 或 `gpt-4-mini` |
| **响应格式** | 结构化输出的 Schema | `ContactInfo` Pydantic 模型 |

**类比理解**：

```
模型上下文 = 演员的剧本
  - 每次演出可能不同（瞬态）
  - 包含角色设定、台词、舞台说明
  - 演完就结束，不影响下一场
```

### 4.2 维度二：Tool Context（工具上下文）

**性质**：**持久（Persistent）**，跨轮次保存。

**控制内容**：工具对数据源的读取与写入。

```
工具执行时：
  读取：查询数据库、获取用户偏好、读取配置
  写入：更新会话状态、修改长期记忆、记录日志

效果：
  本次工具执行的结果会影响后续的模型推理
```

**类比理解**：

```
工具上下文 = 演员的道具和布景
  - 一旦放置，整场演出都可用（持久）
  - 道具可以被修改，后续演出看到新状态
  - 比如：演员写了一封信，后续演员可以看到这封信
```

### 4.3 维度三：Life-cycle Context（生命周期上下文）

**性质**：**持久（Persistent）**，跨步骤生效。

**控制内容**：控制模型调用与工具执行之间的逻辑。

```
生命周期中间件可以做：
  - 对话摘要（压缩历史消息）
  - 护栏检查（安全合规）
  - 日志记录（审计追踪）
  - 动态提示词注入（个性化）
  - 模型切换（根据成本/复杂度）
```

**类比理解**：

```
生命周期上下文 = 导演和舞台管理
  - 控制演出流程，不直接参与表演
  - 决定何时换场景、何时休息、何时结束
  - 确保演出顺利进行
```

### 4.4 三大维度对比

```
┌─────────────────────────────────────────────────────────┐
│              三大控制维度全景图                            │
│                                                         │
│  Model Context          Tool Context         Life-cycle  │
│  (模型上下文)           (工具上下文)          (生命周期)   │
│                                                         │
│  瞬态 ←────────────────→ 持久 ←────────────→ 持久        │
│  单次有效               跨轮次保存           跨步骤生效    │
│                                                         │
│  系统提示词             数据库查询           对话摘要      │
│  消息历史               会话状态更新         护栏检查      │
│  可用工具               长期记忆读写         日志记录      │
│  模型选择                                  动态提示词      │
│  响应格式                                  模型切换       │
└─────────────────────────────────────────────────────────┘
```

---

## 五、三大数据源

上下文工程依赖三种数据源，它们的生命周期和作用域不同：

```
┌─────────────────────────────────────────────────────────┐
│                    三大数据源                              │
│                                                         │
│  Runtime Context        State                Store       │
│  (运行时上下文)         (状态)               (存储)       │
│                                                         │
│  静态配置               短期记忆             长期记忆      │
│  会话级作用域           会话级作用域         跨会话作用域  │
│                                                         │
│  User ID               当前消息            用户偏好       │
│  API Key              上传文件            历史洞察       │
│  环境标志              认证状态            用户画像       │
└─────────────────────────────────────────────────────────┘
```

### 5.1 Runtime Context（运行时上下文）

**性质**：静态配置，会话级作用域。

```python
from dataclasses import dataclass

@dataclass
class AppContext:
    user_id: str           # 用户身份
    api_key: str           # API 密钥
    environment: str       # 环境标志（dev/prod）
    role: str              # 用户角色（admin/viewer）

# 在 Agent 调用时传入
context = AppContext(
    user_id="user-123",
    api_key="sk-...",
    environment="prod",
    role="admin",
)
result = agent.invoke({"messages": [...]}, context=context)
```

### 5.2 State（状态）

**性质**：短期记忆，会话级作用域。

```python
# State 在会话中持续存在，但会话结束后消失
state = {
    "messages": [...],        # 当前消息历史
    "uploaded_files": [...],  # 上传的文件
    "auth_status": True,      # 认证状态
}
```

### 5.3 Store（存储）

**性质**：长期记忆，跨会话作用域。

```python
# Store 在多个会话之间持久保存
runtime.store.put(
    ("users", "user-123"),
    "preferences",
    {"language": "zh", "theme": "dark"}
)

# 下次会话中读取
prefs = runtime.store.get(("users", "user-123"), "preferences")
```

### 5.4 三大数据源对比

| 数据源 | 生命周期 | 作用域 | 示例 |
|--------|----------|--------|------|
| **Runtime Context** | 单次调用 | 会话内 | 用户 ID、API Key |
| **State** | 会话期间 | 会话内 | 消息历史、认证状态 |
| **Store** | 永久 | 跨会话 | 用户偏好、历史洞察 |

---

## 六、上下文工程的实现机制：中间件

LangChain 通过 **Middleware（中间件）** 机制实现上下文工程。中间件允许开发者钩入（hook）智能体生命周期的任何步骤。

### 6.1 中间件能做什么

```
中间件允许你：
  ✅ 更新上下文：修改 State、Store 或单次调用的输入
  ✅ 控制生命周期跳转：根据当前上下文决定跳过工具执行、
     重复模型调用或进入其他步骤
  ✅ 动态修改：提示词、工具列表、模型选择
  ✅ 横切关注点：日志、安全、摘要、重试
```

### 6.2 中间件在 Agent 循环中的位置

```
┌─────────────────────────────────────────────────────────┐
│              中间件在 Agent 循环中的钩子                    │
│                                                         │
│   用户输入                                                │
│      │                                                  │
│      ▼                                                  │
│   before_agent ────────────►  会话级钩子                 │
│      │                                                  │
│      ▼                                                  │
│   before_model ────────────►  模型调用前                  │
│      │                                                  │
│      ▼                                                  │
│   wrap_model_call ────────►  包裹模型调用                 │
│      │                                                  │
│      ▼                                                  │
│   模型调用                                               │
│      │                                                  │
│      ▼                                                  │
│   after_model ────────────►  模型调用后                  │
│      │                                                  │
│      ▼                                                  │
│   wrap_tool_call ─────────►  包裹工具调用                │
│      │                                                  │
│      ▼                                                  │
│   工具执行                                               │
│      │                                                  │
│      ▼                                                  │
│   after_agent ─────────────►  会话级钩子                 │
│      │                                                  │
│      ▼                                                  │
│   输出 / 回到循环                                        │
└─────────────────────────────────────────────────────────┘
```

### 6.3 典型中间件示例

#### SummarizationMiddleware —— 自动会话摘要

```python
from langchain.agents.middleware import SummarizationMiddleware

# 对话超限时自动压缩历史记录
summarizer = SummarizationMiddleware(
    model="gpt-4-mini",
    trigger=("tokens", 4000),   # 超过 4000 token 时触发
    keep=("messages", 20),       # 保留最近 20 条消息
)

agent = create_agent(
    model="gpt-4",
    middleware=[summarizer],
)
```

#### LLMToolSelectorMiddleware —— 动态工具过滤

```python
from langchain.agents.middleware import LLMToolSelectorMiddleware

# 根据用户权限动态展示或隐藏工具
selector = LLMToolSelectorMiddleware(
    model="gpt-4-mini",
    always_include=["search"],  # 始终包含
    max_tools=5,                 # 最多展示 5 个工具
)
```

---

## 七、使用场景与最佳实践

### 7.1 典型使用场景

#### 场景 1：动态提示词/模型切换

根据对话长度或环境成本动态调整系统指令或切换模型。

```python
@before_model
def dynamic_model_switch(state, runtime):
    """根据对话长度切换模型。"""
    total_tokens = count_tokens(state["messages"])
    
    if total_tokens > 10000:
        # 长对话使用大窗口模型
        request.override(model="claude-sonnet-4-5")
    else:
        request.override(model="gpt-4-mini")
    
    return request
```

#### 场景 2：动态工具过滤

根据用户认证状态、权限角色或功能开关，动态展示或隐藏特定工具。

```python
@before_model
def filter_tools_by_role(state, runtime):
    """根据用户角色过滤工具。"""
    role = runtime.context.role
    
    if role != "admin":
        # 非管理员隐藏敏感工具
        request.override(
            tools=[t for t in request.tools 
                   if t.name not in ["delete_record", "admin_tool"]]
        )
    
    return request
```

#### 场景 3：合规与安全注入

运行时动态注入 GDPR/HIPAA 等合规规则到提示词末尾。

```python
@before_model
def inject_compliance(state):
    """注入合规规则。"""
    compliance_prompt = SystemMessage(
        "请注意遵守 GDPR 和 HIPAA 法规。"
        "不要泄露个人敏感信息。"
        "涉及医疗信息时，请先验证用户身份。"
    )
    
    return {
        "messages": [compliance_prompt] + state["messages"]
    }
```

#### 场景 4：自动会话摘要

对话超限时自动压缩历史记录并持久化到 State。

```python
# 使用内置 SummarizationMiddleware
summarizer = SummarizationMiddleware(
    model="gpt-4-mini",
    trigger=("tokens", 4000),
    keep=("messages", 20),
)
```

#### 场景 5：结构化输出动态切换

根据场景动态切换 Pydantic Schema，确保返回数据符合下游系统要求。

```python
@before_model
def switch_response_format(state):
    """根据用户意图切换输出格式。"""
    last_msg = state["messages"][-1].content
    
    if "提取联系人" in last_msg:
        request.override(response_format=ContactInfo)
    elif "分析情感" in last_msg:
        request.override(response_format=SentimentAnalysis)
    
    return request
```

### 7.2 最佳实践

#### 实践 1：从简开始

先使用静态提示和工具，仅在必要时引入动态逻辑。

```python
# ✅ 第一步：静态配置（最简单）
agent = create_agent(
    model="gpt-4",
    system_prompt="你是客服助手。",
    tools=[search, email],
)

# ✅ 第二步：仅在验证需要后添加动态逻辑
agent = create_agent(
    model="gpt-4",
    middleware=[dynamic_prompt_middleware],
    tools=[search, email],
)
```

#### 实践 2：增量测试

一次只添加一项上下文工程特性，便于定位问题。

```
推荐流程：
  1. 先用静态提示词 + 固定工具 → 验证基本流程
  2. 添加动态提示词 → 验证提示词变化
  3. 添加工具过滤 → 验证工具动态性
  4. 添加 Store 记忆 → 验证跨会话保留
```

#### 实践 3：监控性能

跟踪模型调用次数、Token 消耗和延迟。

```python
@after_model
def log_performance(state, runtime):
    """记录性能指标。"""
    exec_info = runtime.execution_info
    
    logging.info(
        f"run_id: {exec_info.run_id}, "
        f"messages: {len(state['messages'])}, "
        f"tokens: {estimate_tokens(state['messages'])}"
    )
    
    return state
```

#### 实践 4：善用内置中间件

优先使用官方提供的组件，而非重复造轮子。

```python
# ✅ 推荐：使用内置中间件
from langchain.agents.middleware import (
    SummarizationMiddleware,
    LLMToolSelectorMiddleware,
    HumanInTheLoopMiddleware,
)

# ❌ 不推荐：自己实现摘要逻辑
@after_model
def my_own_summarization(state):
    # 自己写摘要逻辑...
```

#### 实践 5：理解瞬态与持久

明确单次修改是仅影响本次调用还是会永久改变 State。

```python
# 瞬态修改：仅影响本次模型调用
@wrap_model_call
def transient_change(state, request, handler):
    # 修改 request 中的消息，但不改变 State 中的历史
    request.messages = [new_msg] + request.messages
    return handler(request)

# 持久修改：永久改变 State
@before_model
def persistent_change(state):
    # 通过返回字典更新 State
    return {"messages": state["messages"] + [new_msg]}
```

---

## 八、重要的技术细节

### 8.1 瞬态 vs 持久更新

这是上下文工程中最容易混淆的概念：

```
瞬态更新（Transient）:
  - 使用 wrap_model_call 修改 Messages
  - 仅影响本次调用
  - 不会改变 State 中的历史记录

持久更新（Persistent）:
  - 通过 ExtendedModelResponse 返回 Command
  - 或使用生命周期钩子（before_model/after_model）
  - 永久改变 State
```

```python
# 瞬态：不改变 State
@wrap_model_call
def transient_change(state, request, handler):
    request.messages = [SystemMessage("额外指令")] + request.messages
    return handler(request)  # State 不变

# 持久：改变 State
@before_model
def persistent_change(state):
    return {
        "messages": [SystemMessage("额外指令")] + state["messages"]
    }  # State 被更新
```

### 8.2 工具定义规范

工具的名称、描述和参数说明**不仅是元数据，更是引导 LLM 推理的关键**。

```python
# ✅ 好的工具定义：描述清晰、参数有说明
@tool
def search_web(query: str, max_results: int = 5) -> str:
    """搜索互联网获取最新信息。
    
    当你需要实时数据、新闻或事实时使用此工具。
    
    Args:
        query: 搜索关键词（2-10 个词）
        max_results: 最大返回结果数
    """
    ...

# ❌ 差的工具定义：描述模糊
@tool
def search(query: str) -> str:
    """搜索。"""
    ...
```

### 8.3 工具过载 vs 工具不足

```
工具过载:
  ❌ 向 LLM 暴露过多工具（如 50+ 个）
  后果：
    - 上下文过载，浪费 Token
    - 模型混淆，增加幻觉错误
    - 选择困难，调用不相关工具

工具不足:
  ❌ 向 LLM 暴露过少工具（如只给 1 个）
  后果：
    - 能力受限，无法完成任务
    - 模型尝试用文本生成替代工具调用

正确做法:
  ✅ 根据场景动态过滤工具集（5-10 个为佳）
```

### 8.4 Schema 设计精度

定义 `Response Format` 时，字段名、类型和描述必须极其精确。

```python
# ✅ 好的 Schema：字段名清晰、类型精确、描述详细
class ContactInfo(BaseModel):
    """从文本中提取的联系人信息。"""
    name: str = Field(description="完整姓名，如'张三'")
    email: str = Field(description="邮箱地址，必须符合 email 格式")
    phone: str = Field(description="11 位手机号码")

# ❌ 差的 Schema：字段名模糊、类型不明确
class BadContact(BaseModel):
    info: str  # "info" 太模糊
    data: Any  # Any 类型，模型不知道输出什么
```

---

## 九、常见错误与陷阱

### 陷阱 1：混淆瞬态与持久更新

**现象**：在 `wrap_model_call` 中修改消息，期望改变 State 历史，但实际只影响本次调用。

```python
# 误解：以为这会改变 State 历史
@wrap_model_call
def wrong_approach(state, request, handler):
    request.messages.append(SystemMessage("记住这个"))
    return handler(request)
    
# 结果：State 中的历史消息不变，只有本次调用看到这条消息

# 正确：使用 before_model 持久更新
@before_model
def right_approach(state):
    return {
        "messages": state["messages"] + [SystemMessage("记住这个")]
    }
```

### 陷阱 2：工具定义不规范

**现象**：工具描述模糊，模型不知道该什么时候用。

```python
# ❌ 差描述
@tool
def get_data(query: str) -> str:
    """获取数据。"""

# ✅ 好描述
@tool
def get_data(query: str) -> str:
    """从数据库中查询用户、订单或产品信息。
    
    当用户询问具体的数据记录时使用此工具。
    不适合用于搜索互联网或执行计算。
    
    Args:
        query: SQL 查询语句或自然语言描述
    """
```

### 陷阱 3：上下文过载

**现象**：一次性把所有信息都塞给模型，超出上下文窗口或导致模型迷失。

```
错误做法：
  - 传入 100,000 字的文档全文
  - 暴露 50 个工具给模型
  - 传入 500 条消息的历史

正确做法：
  - 使用 RAG 检索相关片段
  - 动态过滤工具（5-10 个）
  - 使用 SummarizationMiddleware 压缩历史
```

---

## 十、总结：Context Engineering 的本质

| 概念 | 比喻 | 作用 |
|------|------|------|
| **Context Engineering** | 舞台导演 | 编排整个演出的上下文 |
| **Model Context** | 演员剧本 | 单次调用的输入 |
| **Tool Context** | 道具布景 | 工具读写的数据 |
| **Life-cycle Context** | 舞台管理 | 控制执行流程 |
| **Runtime Context** | 演员身份卡 | 静态配置信息 |
| **State** | 当前舞台状态 | 短期会话记忆 |
| **Store** | 道具仓库 | 长期跨会话记忆 |
| **Middleware** | 幕后工作人员 | 钩入生命周期各步骤 |

**Context Engineering 的本质**是**围绕模型输入上下文进行系统化设计与治理**。它不是单点技巧（如只关注提示词怎么写），而是一个综合性工程问题——在 Agent 运行过程中，如何管理上下文的构建、筛选、压缩、注入、维护和验证，确保模型在每次调用时都获得最合适、最精简、最相关的信息。

```
一句话总结：

Context Engineering 是一个上位概念，核心是"给模型什么上下文、
为什么给、给多少、什么时候给、如何压缩和维护"。

它本质上是综合性的工程问题，不是单点技巧。
```

---


 