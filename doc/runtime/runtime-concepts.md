# LangChain Runtime 核心概念教程


---

## 一、什么是 Runtime？

### 1.1 从"硬编码"到"依赖注入"

想象你在开发一个需要访问数据库的工具：

```
硬编码方式（没有 Runtime）:
  @tool
  def get_user_info(user_id: str) -> str:
      # 问题 1：数据库连接硬编码
      db = connect_to_database("postgresql://admin:password@prod-db:5432")
      
      # 问题 2：环境写死，无法本地测试
      # 问题 3：用户身份全局获取，难以测试
      
      return db.query(f"SELECT * FROM users WHERE id = {user_id}")

依赖注入方式（使用 Runtime）:
  @tool
  def get_user_info(user_id: str, runtime: ToolRuntime[AppContext]) -> str:
      # ✅ 数据库连接从上下文注入
      db = runtime.context.db_connection
      
      # ✅ 用户身份从上下文注入
      current_user = runtime.context.user_id
      
      # ✅ 长期记忆从 Store 获取
      preferences = runtime.store.get(("users",), current_user)
      
      return db.query(f"SELECT * FROM users WHERE id = {user_id}")
```

**核心定义**：Runtime 是 LangChain 的**核心运行对象**，负责在 Agent 调用期间提供**静态上下文、存储实例、流式写入器及执行/服务器元数据**。其核心设计理念是**依赖注入（Dependency Injection）**，旨在替代硬编码或全局状态，使工具和中间件更具可测试性、可复用性和灵活性。

### 1.2 依赖注入的价值

```
没有依赖注入（硬编码）:
  ❌ 环境切换困难（开发/测试/生产各写一套代码）
  ❌ 单元测试困难（需要真实的数据库和外部服务）
  ❌ 配置分散在各处，难以管理
  ❌ 代码与基础设施紧耦合

使用依赖注入（Runtime）:
  ✅ 环境切换只需传入不同的 Context 实例
  ✅ 单元测试可以传入 Mock Context
  ✅ 配置集中在 Context 定义处
  ✅ 代码与基础设施解耦
```

---

## 二、Runtime 的工作原理

### 2.1 三步工作流

```
┌─────────────────────────────────────────────────────────┐
│                  Runtime 三步工作流                        │
│                                                         │
│  1. 定义结构（Schema）                                   │
│     创建 Agent 时，通过 context_schema 参数               │
│     定义上下文的静态数据结构（通常使用 @dataclass）        │
│                                                         │
│  2. 注入上下文（Injection）                               │
│     调用 agent.invoke() 时，通过 context 参数              │
│     传入具体的配置实例                                     │
│                                                         │
│  3. 运行时访问（Access）                                  │
│     Agent 运行期间，框架自动将实例化的 Runtime 对象         │
│     传递给工具和中间件，供其按需读取或写入数据              │
└─────────────────────────────────────────────────────────┘
```

### 2.2 详细流程

```
开发阶段:
  1. 定义 Context 数据结构
     @dataclass
     class AppContext:
         user_id: str
         db_connection: Database
         api_key: str

创建 Agent:
  2. 声明 context_schema
     agent = create_agent(
         model="gpt-4",
         tools=[my_tool],
         context_schema=AppContext,  ← 定义结构
     )

调用 Agent:
  3. 传入具体实例
     context = AppContext(
         user_id="user-123",
         db_connection=my_db,
         api_key="sk-...",
     )
     result = agent.invoke(
         {"messages": [...]},
         context=context,  ← 注入实例
     )

运行期间:
  4. 工具/中间件自动接收 Runtime
     @tool
     def my_tool(query: str, runtime: ToolRuntime[AppContext]):
         user_id = runtime.context.user_id  ← 读取上下文
         db = runtime.context.db_connection
         ...
```

---

## 三、Runtime 的主要组件

Runtime 包含以下核心组件：

```
┌─────────────────────────────────────────────────────────┐
│                    Runtime 组件全貌                        │
│                                                         │
│  Runtime                                                │
│  ├── context          静态上下文（用户 ID、DB 连接等）     │
│  ├── store            长期记忆存储（BaseStore 实例）       │
│  ├── stream_writer    流式写入器（推送实时信息）           │
│  ├── execution_info   执行信息（thread_id, run_id 等）    │
│  └── server_info      服务器信息（LangGraph Server 专属） │
└─────────────────────────────────────────────────────────┘
```

### 3.1 Context —— 静态上下文

**定义**：只读的静态信息，如用户 ID、数据库连接或其他外部依赖。

```python
from dataclasses import dataclass

@dataclass
class AppContext:
    user_id: str
    db_connection: Database
    api_key: str
    environment: str  # "dev" | "staging" | "prod"
```

**在工具中访问**：

```python
@tool
def get_user_data(user_id: str, runtime: ToolRuntime[AppContext]) -> str:
    # 读取静态上下文
    db = runtime.context.db_connection
    current_user = runtime.context.user_id
    
    return db.query(f"SELECT * FROM users WHERE id = {user_id}")
```

**在中间件中访问**：

```python
@before_model
def inject_user_context(state, runtime: Runtime[AppContext]):
    user_id = runtime.context.user_id
    env = runtime.context.environment
    
    # 根据环境注入不同的提示词
    if env == "prod":
        prompt = SystemMessage(f"你是生产环境助手，当前用户: {user_id}")
    else:
        prompt = SystemMessage(f"你是开发环境助手")
    
    return {"messages": [prompt] + state["messages"]}
```

### 3.2 Store —— 长期记忆存储

**定义**：`BaseStore` 实例，用于读写长期记忆（Long-term Memory）。

```python
# 读取长期记忆
preferences = runtime.store.get(("users", user_id), "preferences")

# 写入长期记忆
runtime.store.put(("users", user_id), "preferences", {"theme": "dark", "lang": "zh"})

# 删除记忆
runtime.store.delete(("users", user_id), "preferences")
```

**典型场景**：

```
用户偏好存储:
  runtime.store.put(("users", "user-123"), "preferences", {
      "language": "zh",
      "theme": "dark",
      "notifications": True
  })

下次访问时读取:
  prefs = runtime.store.get(("users", "user-123"), "preferences")
  # {"language": "zh", "theme": "dark", "notifications": True}
```

### 3.3 Stream Writer —— 流式写入器

**定义**：用于通过 `"custom"` 流模式传输实时信息（如工具进度）。

```python
@tool
def long_running_task(task: str, runtime: ToolRuntime) -> str:
    stream = runtime.stream_writer
    
    # 推送进度
    stream({"status": "开始处理...", "progress": 0})
    
    # ... 执行任务 ...
    stream({"status": "处理中...", "progress": 50})
    
    # ... 继续执行 ...
    stream({"status": "即将完成...", "progress": 90})
    
    # 完成
    stream({"status": "完成!", "progress": 100})
    
    return "任务完成。"
```

**前端接收**：

```tsx
for await (const chunk of agent.stream(..., stream_mode="custom")) {
  console.log(chunk.data);
  // {"status": "开始处理...", "progress": 0}
  // {"status": "处理中...", "progress": 50}
  // ...
}
```

### 3.4 Execution Info —— 执行信息

**定义**：当前执行的身份与重试信息，包含 `thread_id`、`run_id`、`attempt number`。

```python
execution_info = runtime.execution_info

print(execution_info.thread_id)    # 线程 ID（用于会话隔离）
print(execution_info.run_id)       # 运行 ID（用于追踪）
print(execution_info.attempt)      # 重试次数
```

**典型用途**：

- **日志记录**：关联同一次执行的所有日志
- **审计追踪**：记录谁在什么时间做了什么
- **调试**：根据 run_id 追踪完整的执行链路

```python
import logging

@tool
def delete_record(record_id: str, runtime: ToolRuntime) -> str:
    exec_info = runtime.execution_info
    
    logging.info(
        f"[thread:{exec_info.thread_id}] "
        f"[run:{exec_info.run_id}] "
        f"删除记录: {record_id}"
    )
    
    # 执行删除
    return "已删除。"
```

### 3.5 Server Info —— 服务器信息

**定义**：LangGraph Server 特有元数据，包含 `assistant_id`、`graph_id`、认证用户信息。

```python
server_info = runtime.server_info

if server_info is not None:  # ⚠️ 必须判空！
    print(server_info.assistant_id)  # 助手 ID
    print(server_info.graph_id)      # 图 ID
    print(server_info.authenticated_user)  # 认证用户
```

**⚠️ 重要**：`server_info` 在非 LangGraph Server 环境（如本地开发）下返回 `None`。访问前必须进行判空处理，否则在本地调试时会触发 `AttributeError`。

```python
# ✅ 正确：判空处理
@tool
def my_tool(query: str, runtime: ToolRuntime) -> str:
    server = runtime.server_info
    if server is not None:
        print(f"运行在服务器上，assistant_id: {server.assistant_id}")
    else:
        print("本地运行")
    ...

# ❌ 错误：直接访问（本地运行时会崩溃）
@tool
def bad_tool(query: str, runtime: ToolRuntime) -> str:
    print(runtime.server_info.assistant_id)  # AttributeError!
    ...
```

---

## 四、Runtime 在工具中的应用

### 4.1 ToolRuntime 类型

`ToolRuntime[Context]` 是专用于在工具函数内部访问 Runtime 的泛型类型参数。

```python
from langchain_core.tools import ToolRuntime

@tool
def my_tool(query: str, runtime: ToolRuntime[AppContext]) -> str:
    # 访问静态上下文
    user_id = runtime.context.user_id
    
    # 访问长期记忆
    prefs = runtime.store.get(("users", user_id), "preferences")
    
    # 推送流式进度
    runtime.stream_writer({"status": "处理中..."})
    
    # 访问执行信息
    run_id = runtime.execution_info.run_id
    
    return f"处理完成，run_id: {run_id}"
```

### 4.2 关键特性：对模型隐藏

声明 `runtime: ToolRuntime` 参数会自动注入，但会被**从暴露给 LLM 的 JSON Schema 中剔除**。

```
工具定义:
  @tool
  def search(query: str, runtime: ToolRuntime) -> str:
      ...

暴露给 LLM 的 Schema:
  {
    "name": "search",
    "parameters": {
      "query": {"type": "string"}  ← 只有 query，runtime 被隐藏
    }
  }
```

**为什么这很重要？**

- 工具的真实签名包含 `runtime`，但模型不应该知道或传入这个参数
- 框架自动在调用时注入 `runtime`，模型只需提供业务参数

---

## 五、Runtime 在中间件中的应用

### 5.1 Node-style Hooks

直接添加 `Runtime[Context]` 参数。

```python
from langchain.agents.middleware import before_model, Runtime

@before_model
def log_request(state, runtime: Runtime[AppContext]):
    """记录请求日志。"""
    user_id = runtime.context.user_id
    run_id = runtime.execution_info.run_id
    
    logging.info(f"用户 {user_id} 发起请求 [run:{run_id}]")
    
    return state
```

### 5.2 Wrap-style Hooks

通过 `ModelRequest` 参数（`request.runtime`）访问。

```python
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def generate_prompt(state: dict, request: ModelRequest[AppContext]):
    """动态生成提示词。"""
    runtime = request.runtime
    user_id = runtime.context.user_id
    
    # 读取长期记忆
    prefs = runtime.store.get(("users", user_id), "preferences")
    lang = prefs.get("language", "en")
    
    # 根据语言修改提示词
    if lang == "zh":
        request.system_message.content_blocks.append({
            "type": "text",
            "text": "请用中文回复。"
        })
    
    return request.system_message
```

### 5.3 典型应用

| 应用 | 说明 | 使用的 Runtime 组件 |
|------|------|---------------------|
| **动态提示词** | 根据用户身份拼接 system_prompt | `context`, `store` |
| **请求日志** | 记录每次调用的详细信息 | `execution_info` |
| **流式修改** | 基于上下文推送自定义事件 | `stream_writer` |
| **鉴权拦截** | 检查用户权限，拒绝未授权访问 | `context`, `server_info` |

---

## 六、状态管理、上下文传递、持久化存储

### 6.1 上下文传递：类型安全的设计

通过类型安全的 `context_schema`（数据类）定义结构，在 `invoke` 时显式传入，实现强类型、安全的上下文传递。

```python
from dataclasses import dataclass

# 1. 定义上下文结构
@dataclass
class AppContext:
    user_id: str
    role: str
    db_connection: Database

# 2. 创建 Agent 时声明 schema
agent = create_agent(
    model="gpt-4",
    tools=[my_tool],
    context_schema=AppContext,
)

# 3. 调用时传入具体实例
context = AppContext(
    user_id="user-123",
    role="admin",
    db_connection=my_db,
)

result = agent.invoke(
    {"messages": [...]},
    context=context,
)
```

### 6.2 持久化存储：Store 操作

通过 `runtime.store` 接口操作长期记忆。

```python
# 存储结构: 命名空间 (namespace) + 键 (key) = 值 (value)

# 写入
runtime.store.put(
    ("users", "user-123"),  # 命名空间
    "preferences",           # 键
    {"theme": "dark"}        # 值
)

# 读取
prefs = runtime.store.get(
    ("users", "user-123"),
    "preferences"
)
# {"theme": "dark"}

# 搜索（在同一命名空间内）
results = runtime.store.search(
    ("users", "user-123"),
    query="dark"
)
```

### 6.3 状态追踪：Execution Info

通过 `runtime.execution_info` 获取当前线程和运行 ID。

```python
@tool
def process_data(data: str, runtime: ToolRuntime) -> str:
    exec_info = runtime.execution_info
    
    print(f"thread_id: {exec_info.thread_id}")  # 会话隔离标识
    print(f"run_id: {exec_info.run_id}")        # 单次运行标识
    print(f"attempt: {exec_info.attempt}")      # 重试次数
    
    # 用于日志关联
    logger = logging.getLogger(__name__)
    logger = logger.bind(
        thread_id=exec_info.thread_id,
        run_id=exec_info.run_id
    )
    
    # 处理数据
    return "处理完成。"
```

---

## 七、使用场景与最佳实践

### 7.1 典型场景

| 场景 | 使用的 Runtime 组件 | 说明 |
|------|---------------------|------|
| **多租户应用** | `context.user_id` | 根据用户隔离数据和行为 |
| **环境感知开发** | `context.environment` | 区分本地调试与生产环境 |
| **用户偏好** | `store` | 读取/存储用户个性化设置 |
| **审计日志** | `execution_info` | 记录完整执行链路 |
| **进度推送** | `stream_writer` | 长任务实时反馈 |
| **权限控制** | `server_info.authenticated_user` | 基于认证用户鉴权 |

### 7.2 最佳实践

#### 实践 1：始终通过 context_schema 规范上下文

```python
# ✅ 推荐：明确定义上下文结构
@dataclass
class AppContext:
    user_id: str
    role: str
    db: Database

agent = create_agent(
    context_schema=AppContext,  # 类型安全
    ...
)

# ❌ 不推荐：使用任意字典
agent = create_agent(
    # 没有 schema，类型不安全
    ...
)
context = {"user_id": "123", "role": "admin", ...}  # 容易拼写错误
```

#### 实践 2：利用依赖注入提升可测试性

```python
# 测试时可以传入 Mock Context
@dataclass
class MockContext:
    user_id: str = "test-user"
    db: MockDatabase = field(default_factory=MockDatabase)

def test_my_tool():
    mock_context = MockContext()
    runtime = ToolRuntime(context=mock_context, ...)
    
    result = my_tool.func("test query", runtime)
    assert result == "expected"
```

#### 实践 3：严格遵循 Hook 签名规范

不同风格的 Middleware Hook 获取 Runtime 的方式不同：

```python
# ✅ Node-style：直接添加 Runtime 参数
@before_model
def my_middleware(state, runtime: Runtime[AppContext]):
    ...

# ✅ Wrap-style：通过 request.runtime 访问
@dynamic_prompt
def my_prompt(state, request: ModelRequest[AppContext]):
    runtime = request.runtime
    ...

# ❌ 错误：Node-style 中尝试从 state 获取
@before_model
def wrong_middleware(state):
    runtime = state["runtime"]  # 不存在！
```

#### 实践 4：环境感知开发

利用 `server_info` 区分本地调试与 LangGraph Server 生产环境。

```python
@tool
def get_config(key: str, runtime: ToolRuntime) -> str:
    server = runtime.server_info
    
    if server is not None:
        # 生产环境：从服务器配置读取
        return server.config.get(key)
    else:
        # 本地开发：使用默认值
        return DEFAULT_CONFIGS.get(key)
```

#### 实践 5：Store 命名空间设计

设计清晰的命名空间层次，避免键冲突。

```python
# ✅ 推荐：按实体类型组织
runtime.store.put(("users", user_id), "preferences", {...})
runtime.store.put(("sessions", session_id), "summary", "...")
runtime.store.put(("projects", project_id), "metadata", {...})

# ❌ 不推荐：扁平化存储，容易冲突
runtime.store.put((), f"user_{user_id}_prefs", {...})
runtime.store.put((), f"session_{session_id}_summary", "...")
```

---

## 八、重要技术细节

### 8.1 版本依赖

访问 `runtime.execution_info` 和 `runtime.server_info` 需要：
- `deepagents>=0.5.0` 或
- `langgraph>=1.1.5`

### 8.2 环境差异处理

`server_info` 在非 LangGraph Server 环境下返回 `None`。

```python
# 安全访问模式
if (server := runtime.server_info) is not None:
    assistant_id = server.assistant_id
    graph_id = server.graph_id
    user = server.authenticated_user
```

### 8.3 ToolRuntime 与 Runtime 的区别

| 类型 | 使用场景 | 访问方式 |
|------|----------|----------|
| **ToolRuntime** | 工具函数内部 | 函数参数 `runtime: ToolRuntime[Context]` |
| **Runtime** | 中间件函数内部 | 函数参数 `runtime: Runtime[Context]` 或 `request.runtime` |

---

## 九、常见错误与陷阱

### 陷阱 1：未判空直接访问 server_info

**现象**：本地运行时触发 `AttributeError`。

```python
# ❌ 错误
assistant_id = runtime.server_info.assistant_id  # 本地运行时 None!

# ✅ 正确
if runtime.server_info is not None:
    assistant_id = runtime.server_info.assistant_id
```

### 陷阱 2：忘记声明 context_schema

**现象**：Runtime 中无法访问 context，或者访问到的是 `None`。

```python
# ❌ 错误：没有声明 schema
agent = create_agent(
    tools=[my_tool],
    # 没有 context_schema
)

# ✅ 正确：声明 schema
agent = create_agent(
    tools=[my_tool],
    context_schema=AppContext,
)
```

### 陷阱 3：混淆 Hook 获取 Runtime 的方式

**现象**：在 Wrap-style Hook 中尝试直接获取 `runtime` 参数。

```python
# ❌ 错误：Wrap-style 没有直接 runtime 参数
@dynamic_prompt
def wrong_prompt(state, runtime: Runtime):  # 不存在！
    ...

# ✅ 正确：通过 request.runtime 访问
@dynamic_prompt
def correct_prompt(state, request: ModelRequest):
    runtime = request.runtime
    ...
```

### 陷阱 4：在工具中使用全局状态而非 Runtime

**现象**：代码无法测试，环境切换困难。

```python
# ❌ 错误：使用全局状态
GLOBAL_DB_CONNECTION = connect_to_database("...")

@tool
def bad_tool(query: str) -> str:
    return GLOBAL_DB_CONNECTION.query(query)

# ✅ 正确：通过 Runtime 注入
@tool
def good_tool(query: str, runtime: ToolRuntime[AppContext]) -> str:
    return runtime.context.db_connection.query(query)
```

---

## 十、总结：Runtime 的本质

| 概念 | 比喻 | 作用 |
|------|------|------|
| **Runtime** | 工具箱 | 提供运行时所需的一切资源 |
| **Context** | 身份证 | 静态身份信息（你是谁、在哪） |
| **Store** | 记忆库 | 长期持久化存储 |
| **Stream Writer** | 对讲机 | 实时推送进度信息 |
| **Execution Info** | 日志标签 | 追踪执行链路 |
| **Server Info** | 服务器通行证 | LangGraph Server 元数据 |
| **context_schema** | 模具 | 定义上下文的类型结构 |
| **依赖注入** | 送餐服务 | 需要什么就传入什么，不用自己找 |

**Runtime 的本质**是**依赖注入容器**。它通过在 Agent 调用时自动注入上下文、存储、流式写入器和元数据，让工具和中间件能够访问运行环境的信息，而无需硬编码或依赖全局状态——这使得代码更清晰、更可测试、更易于维护。

---
