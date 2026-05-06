# LangChain Long-term Memory 核心概念教程

---

## 一、什么是 Long-term Memory？

### 1.1 从"金鱼记忆"到"大象记忆"

想象你和两个人打交道：

```
金鱼记忆（没有长期记忆）:
  你："我叫小明，我喜欢喝咖啡。"
  AI："好的，小明！我记住了。"
  
  （第二天，新的对话）
  你："你好！"
  AI："你好！请问怎么称呼？"  ← 完全忘了！
  
  你："我叫小明啊，昨天说过的"
  AI："哦抱歉，我不记得了..."

大象记忆（有长期记忆）:
  你："我叫小明，我喜欢喝咖啡。"
  AI："好的，小明！我记住了。"
  
  （第二天，新的对话）
  你："你好！"
  AI："你好小明！今天想聊些什么？
       要聊聊你喜欢的咖啡吗？"  ← 记住了！
```

**核心定义**：Long-term Memory（长期记忆）允许 LangChain Agent **跨不同的对话（Threads）和会话**持久化存储并召回信息。与仅限单线程的短期记忆不同，它能在**任意时间**被访问。

---

## 二、为什么需要 Long-term Memory？

### 2.1 短期记忆的局限

```
┌─────────────────────────────────────────────────────────┐
│              短期记忆 vs 长期记忆                          │
│                                                         │
│  短期记忆（Short-term Memory）                           │
│  ├── 范围：单个会话内                                    │
│  ├── 内容：完整的对话历史                                 │
│  ├── 存储：Checkpointer（检查点）                        │
│  ├── 生命周期：会话结束后丢弃                            │
│  └── 类比：人的工作记忆（短期记忆）                       │
│                                                         │
│  长期记忆（Long-term Memory）                            │
│  ├── 范围：跨多个会话                                    │
│  ├── 内容：提炼的关键信息（偏好、画像、重要事实）          │
│  ├── 存储：Store（存储）                                 │
│  ├── 生命周期：持久保留，可跨会话访问                     │
│  └── 类比：人的知识库（长期记忆）                         │
└─────────────────────────────────────────────────────────┘
```

### 2.2 长期记忆的价值

```
让 Agent 突破单次会话的限制：
  ✅ 记住用户的历史交互
  ✅ 记住用户的偏好和习惯
  ✅ 记住重要的上下文信息
  ✅ 在多次独立对话中提供连贯、个性化的体验

典型场景：
  - "上次你推荐的那家餐厅叫什么来着？"  ← 跨会话回忆
  - "我还是喜欢上次那种风格"            ← 记住偏好
  - "我之前说过我对海鲜过敏"             ← 记住重要事实
```

---

## 三、Long-term Memory 的工作原理和架构

### 3.1 基于 LangGraph Store 构建

长期记忆基于 LangGraph Store 构建。数据以 **JSON 文档**形式存储，采用**命名空间（namespace）**和**键（key）**的层级结构组织。

```
Store 的层级结构：

类比文件系统：
  namespace（命名空间） ≈ 文件夹路径
  key（键）             ≈ 文件名
  value（值）           ≈ 文件内容（JSON 文档）

示例结构：
  /users/user-123/preferences
    └── language: "zh"
        coffee_preference: "latte"
  
  /users/user-123/profile
    └── name: "小明"
        birthday: "1990-01-01"
  
  /users/user-456/preferences
    └── language: "en"
        coffee_preference: "americano"
```

### 3.2 命名空间隔离

```
┌─────────────────────────────────────────────────────────┐
│              命名空间隔离机制                              │
│                                                         │
│  namespace 的作用：                                      │
│  ├── 隔离不同用户的数据                                   │
│  ├── 隔离不同应用的数据                                   │
│  └── 隔离不同类型的数据                                  │
│                                                         │
│  示例：                                                  │
│  ("users", "user-123", "preferences")  ← 用户 123 的偏好 │
│  ("users", "user-456", "preferences")  ← 用户 456 的偏好 │
│  ("apps", "app-abc", "config")         ← 应用 abc 的配置  │
│  ("global", "rules")                   ← 全局规则         │
└─────────────────────────────────────────────────────────┘
```

---

## 四、Store 的机制和使用方式

### 4.1 两种 Store 实现

```
┌─────────────────────────────────────────────────────────┐
│              两种 Store 实现                               │
│                                                         │
│  InMemoryStore（内存存储）               PostgresStore（数据库存储）│
│                                                         │
│  基于内存字典                            基于 PostgreSQL 数据库     │
│  适用于开发环境                           适用于生产环境              │
│  服务重启后数据丢失                       数据持久化                  │
│  无需额外配置                            需安装依赖并初始化表结构      │
│                                                         │
│  ⚠️ 严禁用于生产！                      ✅ 生产环境推荐               │
└─────────────────────────────────────────────────────────┘
```

### 4.2 InMemoryStore（开发环境）

```python
from langgraph.store.memory import InMemoryStore

# 创建内存存储
store = InMemoryStore()

# 特点：
# ✅ 开箱即用，无需配置
# ✅ 适合快速原型开发
# ❌ 数据仅存在于进程内存
# ❌ 服务重启后数据丢失
```

### 4.3 PostgresStore（生产环境）

```python
from langgraph.store.postgres import PostgresStore

# 创建数据库存储
store = PostgresStore(
    connection_string="postgresql://user:pass@localhost:5432/mydb"
)

# 初始化底层表结构（首次使用时必须调用）
store.setup()

# 特点：
# ✅ 数据持久化，重启不丢失
# ✅ 支持多实例共享
# ✅ 适合生产环境
# ❌ 需要安装依赖：pip install langgraph-checkpoint-postgres
# ❌ 需要调用 setup() 初始化表结构
```

---

## 五、如何在 LangChain 中实现 Long-term Memory

### 5.1 核心三步

```
┌─────────────────────────────────────────────────────────┐
│              长期记忆实现三步曲                            │
│                                                         │
│  1. 初始化 Store 实例                                     │
│     根据环境选择 InMemoryStore 或 PostgresStore           │
│                                                         │
│  2. 传入 Agent                                           │
│     将 store 实例作为 store 参数传入 create_agent          │
│                                                         │
│  3. 在工具中访问                                          │
│     声明 runtime: ToolRuntime[Context] 参数                │
│     通过 runtime.store 进行读写操作                        │
└─────────────────────────────────────────────────────────┘
```

### 5.2 完整示例

```python
from langchain.agents import create_agent
from langgraph.store.memory import InMemoryStore
from langchain_core.tools import tool
from langchain_core.tools import ToolRuntime
from dataclasses import dataclass

# ──────────────────────────────────────────────────────
# 步骤 1：初始化 Store
# ──────────────────────────────────────────────────────

# 开发环境
store = InMemoryStore()

# 生产环境（推荐）
# store = PostgresStore(connection_string="postgresql://...")
# store.setup()

# ──────────────────────────────────────────────────────
# 步骤 2：定义上下文结构
# ──────────────────────────────────────────────────────

@dataclass
class AppContext:
    user_id: str  # 用户标识符，用于定位命名空间

# ──────────────────────────────────────────────────────
# 步骤 3：创建工具并访问 Store
# ──────────────────────────────────────────────────────

@tool
def save_preference(pref_name: str, pref_value: str, runtime: ToolRuntime[AppContext]) -> str:
    """保存用户偏好设置。
    
    Args:
        pref_name: 偏好名称，如 "language"、"coffee"
        pref_value: 偏好值
    """
    # 从上下文获取用户 ID
    user_id = runtime.context.user_id
    
    # 写入长期记忆
    runtime.store.put(
        ("users", user_id, "preferences"),  # 命名空间
        pref_name,                          # 键
        {"value": pref_value}               # 值（JSON 文档）
    )
    
    return f"已保存偏好: {pref_name} = {pref_value}"

@tool
def get_preference(pref_name: str, runtime: ToolRuntime[AppContext]) -> str:
    """获取用户偏好设置。
    
    Args:
        pref_name: 偏好名称
    """
    user_id = runtime.context.user_id
    
    # 从长期记忆读取
    result = runtime.store.get(
        ("users", user_id, "preferences"),  # 命名空间
        pref_name                           # 键
    )
    
    if result is None:
        return f"未找到偏好: {pref_name}"
    
    return result.value.get("value", "未设置")

# ──────────────────────────────────────────────────────
# 步骤 4：创建 Agent 并注入 Store
# ──────────────────────────────────────────────────────

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[save_preference, get_preference],
    store=store,  # 注入 Store
    context_schema=AppContext,
    system_prompt="""你是个人助手，可以记住用户的偏好。
    
    使用 save_preference 保存用户偏好。
    使用 get_preference 查询用户偏好。
    """
)

# ──────────────────────────────────────────────────────
# 步骤 5：调用 Agent
# ──────────────────────────────────────────────────────

# 第一次对话：保存偏好
context = AppContext(user_id="user-123")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "我喜欢喝拿铁咖啡"}]},
    context=context,
)

# 第二次对话（新的对话，不同的 thread）：查询偏好
result = agent.invoke(
    {"messages": [{"role": "user", "content": "我喜欢喝什么咖啡？"}]},
    context=context,  # 相同的 user_id，可以读取之前的保存的偏好
)
# AI 应该回答："你喜欢喝拿铁咖啡。" ← 从长期记忆中读取！
```

---

## 六、记忆的生命周期管理

### 6.1 创建/写入（Put）

```python
# 写入记忆
store.put(
    ("users", "user-123", "preferences"),  # 命名空间（类似文件夹路径）
    "coffee",                               # 键（类似文件名）
    {"value": "latte", "updated_at": "2024-01-01"}  # 值（JSON 文档）
)
```

### 6.2 读取（Get）

```python
# 读取记忆
result = store.get(
    ("users", "user-123", "preferences"),  # 命名空间
    "coffee"                                # 键
)

if result is not None:
    print(result.value)        # {"value": "latte", "updated_at": "2024-01-01"}
    print(result.value["value"])  # "latte"
    print(result.metadata)     # 元数据（如创建时间、更新时间等）
```

### 6.3 搜索（Search）

```python
# 搜索记忆
# 方式 A：基于键值过滤
results = store.search(
    ("users", "user-123", "preferences"),
    filter={"value": "latte"}
)

# 方式 B：基于向量相似度的语义检索（需配置 IndexConfig）
results = store.search(
    ("users", "user-123", "preferences"),
    query="我喜欢喝的咖啡类型"  # 语义搜索，找到最相关的记忆
)

for result in results:
    print(f"键: {result.key}")
    print(f"值: {result.value}")
```

### 6.4 更新（Update）

```python
# 更新记忆：使用相同的 namespace 和 key 再次调用 put 覆盖原有数据
store.put(
    ("users", "user-123", "preferences"),
    "coffee",
    {"value": "americano", "updated_at": "2024-01-02"}  # 覆盖旧值
)
```

### 6.5 删除（Delete）

```python
# 注意：当前文档未提供明确的删除方法
# 但通常可以通过覆盖为空值或标记为删除来实现逻辑删除

store.put(
    ("users", "user-123", "preferences"),
    "coffee",
    {"deleted": True}  # 逻辑删除标记
)
```

---

## 七、使用场景与最佳实践

### 7.1 典型场景

| 场景 | 存储内容 | 命名空间示例 |
|------|---------|-------------|
| **用户画像** | 姓名、语言偏好、时区 | `("users", user_id, "profile")` |
| **用户偏好** | 咖啡偏好、音乐品味、风格喜好 | `("users", user_id, "preferences")` |
| **历史上下文** | 上次讨论的话题、重要决策 | `("users", user_id, "history")` |
| **业务规则** | 跨会话的业务逻辑配置 | `("apps", app_id, "rules")` |
| **全局知识** | 全系统共享的配置或规则 | `("global", "config")` |

### 7.2 最佳实践

#### 实践 1：生产环境务必使用数据库支持的 Store

```python
# ❌ 开发环境：内存存储（重启后丢失）
store = InMemoryStore()

# ✅ 生产环境：PostgreSQL 持久化
from langgraph.store.postgres import PostgresStore

store = PostgresStore(
    connection_string="postgresql://user:pass@host:5432/db"
)
store.setup()  # 首次使用必须初始化表结构
```

#### 实践 2：利用 namespace 逻辑隔离数据

```python
# ✅ 好的命名空间设计
store.put(("users", "user-123", "preferences"), "coffee", {...})
store.put(("users", "user-456", "preferences"), "coffee", {...})
# 不同用户的数据完全隔离

# ✅ 按类型隔离
store.put(("users", user_id, "preferences"), "coffee", {...})
store.put(("users", user_id, "profile"), "name", {...})
store.put(("users", user_id, "history"), "last_topic", {...})

# ❌ 差的命名空间设计（所有数据混在一起）
store.put(("data",), f"user_{user_id}_coffee", {...})
store.put(("data",), f"user_{user_id}_name", {...})
```

#### 实践 3：通过 context 动态传入标识符

```python
@dataclass
class AppContext:
    user_id: str
    org_id: str

@tool
def get_user_pref(key: str, runtime: ToolRuntime[AppContext]) -> str:
    """获取用户偏好。"""
    # 从上下文动态获取用户 ID，而非硬编码
    user_id = runtime.context.user_id
    
    result = runtime.store.get(
        ("users", user_id, "preferences"),
        key
    )
    return result.value["value"] if result else "未设置"
```

#### 实践 4：启用向量语义搜索

```python
from langgraph.store.memory import InMemoryStore
from langchain_openai import OpenAIEmbeddings
from langgraph.store.base import IndexConfig

# 配置嵌入模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 创建支持语义搜索的 Store
store = InMemoryStore(
    index=IndexConfig(
        embed=embeddings.embed_query,  # 嵌入函数
        dims=1536,                     # 向量维度
    )
)

# 现在可以使用语义搜索
results = store.search(
    ("users", "user-123", "preferences"),
    query="我喜欢喝什么样的咖啡"  # 语义匹配，非精确键匹配
)
```

---

## 八、重要技术细节

### 8.1 InMemoryStore 严禁用于生产

```
InMemoryStore 的数据仅存在于进程内存：
  ❌ 服务重启后数据全部丢失
  ❌ 不支持多实例共享
  ❌ 不支持持久化

生产环境必须使用 PostgresStore：
  ✅ 数据持久化到数据库
  ✅ 支持多实例共享
  ✅ 支持备份和恢复
```

### 8.2 PostgresStore 初始化

```python
# 首次使用必须调用 setup()
store = PostgresStore(connection_string="...")
store.setup()  # 初始化底层表结构

# 依赖安装
# pip install langgraph-checkpoint-postgres
```

### 8.3 Store 与 Checkpointer 的区别

```
┌─────────────────────────────────────────────────────────┐
│              Store vs Checkpointer 对比                    │
│                                                         │
│  Store（长期记忆）                      Checkpointer（短期记忆）│
│                                                         │
│  跨会话持久化                            单会话内持久化                │
│  存储提炼的关键信息                        存储完整对话历史              │
│  通过 runtime.store 访问                通过 Checkpointer 自动管理    │
│  手动读写                                自动保存/加载                │
│  命名空间 + 键结构                       线程 ID + 检查点结构           │
│                                                         │
│  类比：人的知识库                         类比：人的工作记忆            │
└─────────────────────────────────────────────────────────┘
```

### 8.4 工具中访问 Store 的方式

```python
# 在工具中声明 ToolRuntime 参数
@tool
def my_tool(arg: str, runtime: ToolRuntime[AppContext]) -> str:
    # 通过 runtime.store 访问长期记忆
    runtime.store.put(("namespace",), "key", {"data": "value"})
    result = runtime.store.get(("namespace",), "key")
    ...
```

---

## 九、常见错误与陷阱

### 陷阱 1：InMemoryStore 用于生产环境

**现象**：服务重启后所有记忆丢失。

```python
# ❌ 错误：生产环境使用 InMemoryStore
store = InMemoryStore()
# 重启后 → 所有用户偏好、画像数据丢失！

# ✅ 正确：生产环境使用 PostgresStore
store = PostgresStore(connection_string="postgresql://...")
store.setup()
```

### 陷阱 2：忘记调用 setup()

**现象**：PostgresStore 初始化失败，无法读写。

```python
# ❌ 错误：未调用 setup()
store = PostgresStore(connection_string="...")
# 直接使用时报错：表不存在

# ✅ 正确：首次使用调用 setup()
store = PostgresStore(connection_string="...")
store.setup()  # 初始化表结构
```

### 陷阱 3：硬编码用户标识符

**现象**：所有用户读写同一份数据，无法隔离。

```python
# ❌ 错误：硬编码 user_id
@tool
def bad_tool(key: str, runtime: ToolRuntime) -> str:
    result = runtime.store.get(("users", "hardcoded_user_id", "prefs"), key)
    # 所有用户都读写同一份数据！

# ✅ 正确：从上下文动态获取
@tool
def good_tool(key: str, runtime: ToolRuntime[AppContext]) -> str:
    user_id = runtime.context.user_id  # 从上下文获取
    result = runtime.store.get(("users", user_id, "prefs"), key)
    # 每个用户读写自己的数据
```

### 陷阱 4：未启用语义搜索却使用 query 参数

**现象**：搜索返回空结果或不准确。

```python
# ❌ 错误：未配置 IndexConfig 却使用语义搜索
store = InMemoryStore()  # 没有配置嵌入模型
results = store.search(
    ("users", user_id, "prefs"),
    query="我喜欢喝的咖啡"  # 语义搜索，但无法工作
)

# ✅ 正确：配置 IndexConfig 后使用
store = InMemoryStore(
    index=IndexConfig(
        embed=embeddings.embed_query,
        dims=1536,
    )
)
results = store.search(
    ("users", user_id, "prefs"),
    query="我喜欢喝的咖啡"  # 语义搜索正常工作
)
```

---

## 十、总结：长期记忆的本质

| 概念 | 比喻 | 作用 |
|------|------|------|
| **Long-term Memory** | 人的知识库 | 跨会话持久化存储 |
| **Store** | 文件柜 | 存储和读取 JSON 文档 |
| **namespace** | 文件夹路径 | 隔离不同用户/应用的数据 |
| **key** | 文件名 | 精确定位某条记忆 |
| **value** | 文件内容 | 存储的实际数据（JSON） |
| **InMemoryStore** | 临时便签 | 开发环境使用，重启丢失 |
| **PostgresStore** | 保险柜 | 生产环境使用，持久保存 |
| **semantic search** | 联想搜索 | 基于语义相似度查找 |
| **context** | 身份证 | 动态传入用户标识符 |

**长期记忆的本质**是**让 Agent 突破单次会话的限制**。它通过 Store 机制将关键信息（用户画像、偏好、历史上下文）持久化存储，使得 Agent 能够在任意时间、任意对话中召回这些信息，从而提供连贯、个性化的用户体验。

```
一句话总结：

长期记忆的核心是"通过 Store 以 namespace + key 的层级结构
持久化存储 JSON 文档，Agent 通过 runtime.store 跨会话读写"，
让 Agent 记住用户的历史交互、偏好和上下文，实现个性化体验。
```


