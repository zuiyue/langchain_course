# LangChain MCP (Model Context Protocol) 核心概念教程



## 一、什么是 MCP？

### 1.1 从"碎片化集成"到"统一协议"

想象你要让 AI 助手连接各种外部系统：

```
没有 MCP 的时代（碎片化集成）:
  ❌ 连接数据库 → 写一个专用工具
  ❌ 调用 API    → 写一个 HTTP 工具
  ❌ 读取文件    → 写一个文件系统工具
  ❌ 发送邮件    → 写一个邮件工具
  ❌ 查询日历    → 写一个日历工具
  
  问题：
  - 每个服务都要写专用代码
  - 认证方式各不相同
  - 错误处理重复编写
  - 维护成本极高

有 MCP 的时代（统一协议）:
  ✅ 任何支持 MCP 的服务 → 一键接入
  ✅ 统一的工具、资源、提示词接口
  ✅ 标准的认证和错误处理
  ✅ 维护成本极低
```

**核心定义**：Model Context Protocol (MCP) 是一个**开放协议**，旨在**标准化**应用程序如何向大语言模型（LLM）提供**工具（Tools）**和**上下文（Context）**。通过 MCP，LangChain 代理可以统一、无缝地调用定义在独立 MCP 服务器上的外部功能，**无需为每个第三方服务编写专用集成代码**。

### 1.2 MCP 类比理解

```
MCP 就像是一个"万能插座"：

以前的电器（工具）：
  - 冰箱插头是两脚的
  - 洗衣机插头是三脚的
  - 空调插头是特殊的
  
  每个电器需要不同的插座和接线方式。

有了万能插座（MCP）后：
  - 所有电器统一用一种插头
  - 插上就能用，不用关心内部接线
  - 新增电器直接插上新插座
```

---

## 二、为什么需要 MCP？

### 2.1 核心问题：集成标准碎片化

```
现状挑战：
  📊 成千上万的 SaaS 服务和 API
  🔌 每个服务都有自己的 SDK 和认证方式
  🔧 开发者需要为每个服务写专用集成代码
  🔄 服务更新时，集成代码也要跟着更新

MCP 解决方案：
  📊 定义统一的协议标准
  🔌 提供标准化的连接方式
  🔧 一次集成，服务可插拔
  🔄 服务更新，无需修改客户端代码
```

### 2.2 MCP 的价值

| 价值 | 说明 |
|------|------|
| **统一接口** | 所有外部服务都通过同一种协议接入 |
| **即插即用** | 新增服务只需启动 MCP 服务器 |
| **降低维护** | 服务端更新不影响客户端 |
| **生态共享** | 任何支持 MCP 的服务都能被 LangChain 使用 |

---

## 三、MCP 的工作原理

### 3.1 架构全景

```
┌─────────────────────────────────────────────────────────┐
│                    MCP 架构全景                           │
│                                                         │
│  客户端（LangChain）                    MCP 服务器         │
│  ┌──────────────────┐                  ┌──────────────┐  │
│  │ MultiServerMCP   │                  │ MCP Server   │  │
│  │ Client           │◄── 协议 ────────►│              │  │
│  │                  │   (HTTP/stdio)   │ 暴露三类实体：│  │
│  │ 转换层:          │                  │              │  │
│  │ MCP → LangChain  │                  │ • Tools      │  │
│  │ 原生对象          │                  │ • Resources  │  │
│  └──────────────────┘                  │ • Prompts    │  │
│                                        └──────────────┘  │
│         │                                ▲               │
│         ▼                                │               │
│  ┌──────────────────┐                  ┌──────────────┐  │
│  │    Agent /       │                  │  更多 MCP     │  │
│  │    LangGraph     │                  │  服务器       │  │
│  └──────────────────┘                  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 3.2 工作流程

```
1. 客户端连接服务器
   ──► 指定传输协议（HTTP 或 stdio）
   ──► 建立通信通道

2. 服务器暴露能力
   ──► Tools（可执行函数）
   ──► Resources（数据/文件）
   ──► Prompts（提示词模板）

3. 客户端获取并转换
   ──► langchain-mcp-adapters 作为转换层
   ──► MCP 数据 → LangChain 原生对象
       • MCP Tool    → LangChain Tool
       • MCP Resource → Blob 文档
       • MCP Prompt   → Message 对象

4. Agent 消费
   ──► 将转换后的工具传入 create_agent
   ──► Agent 可以像使用普通工具一样使用 MCP 工具
```

---

## 四、MCP 服务器的三类实体

### 4.1 Tools（工具）—— 动作执行

**定义**：可执行的函数，Agent 可以调用它们来执行操作。

```
MCP Server 暴露的 Tools:
  ├── get_weather(city)     → 查询天气
  ├── send_email(to, body)  → 发送邮件
  ├── query_db(sql)         → 查询数据库
  └── create_ticket(desc)   → 创建工单

LangChain Agent 可以像调用普通工具一样调用它们。
```

### 4.2 Resources（资源）—— 数据读取

**定义**：可读取的数据或文件，Agent 可以访问这些信息作为上下文。

```
MCP Server 暴露的 Resources:
  ├── file:///docs/manual.pdf    → 产品手册
  ├── db://users/123             → 用户信息
  └── api://config/settings      → 系统配置

Agent 可以读取这些资源，作为回答问题的上下文。
```

### 4.3 Prompts（提示词）—— 模板复用

**定义**：预定义的提示词模板，Agent 可以使用它们来生成标准化的提示。

```
MCP Server 暴露的 Prompts:
  ├── code_review     → 代码审查提示词模板
  ├── summarize_doc   → 文档摘要提示词模板
  └── translate_text  → 翻译提示词模板

Agent 可以传入参数调用这些模板，生成完整的提示词。
```

### 4.4 三类实体对比

| 实体 | 性质 | 用途 | 类比 |
|------|------|------|------|
| **Tools** | 可执行 | 执行动作（写、改、删、查） | 手脚 |
| **Resources** | 只读 | 提供数据上下文 | 眼睛和耳朵 |
| **Prompts** | 模板 | 生成标准化提示词 | 剧本 |

---

## 五、传输协议：HTTP vs stdio

### 5.1 两种传输方式

MCP 支持两种传输协议，各有适用场景：

```
┌─────────────────────────────────────────────────────────┐
│                  两种传输协议对比                          │
│                                                         │
│  HTTP 传输                          stdio 传输            │
│                                                         │
│  适用：远程服务                     适用：本地服务         │
│  认证：Headers / Auth               认证：无（本地进程）   │
│  部署：独立服务器                   部署：子进程启动       │
│  性能：网络延迟                     性能：进程通信         │
│                                                         │
│  示例：                             示例：                │
│  - 云端 API 服务                    - 本地 Python 脚本    │
│  - 第三方 SaaS                     - 本地 CLI 工具       │
│  - 微服务架构                      - 轻量级集成          │
└─────────────────────────────────────────────────────────┘
```

### 5.2 HTTP 传输

用于连接远程 MCP 服务器，支持 Header 认证和追踪。

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "weather_server": {
        "transport": "http",
        "url": "https://weather-mcp.example.com/mcp",
        "headers": {
            "Authorization": "Bearer sk-...",
            "X-Request-ID": "req-123",
        },
    }
})

tools = await client.get_tools()
```

### 5.3 stdio 传输

通过标准输入/输出运行本地子进程，适合轻量级工具。

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "local_tools": {
        "transport": "stdio",
        "command": "python",
        "args": ["-m", "my_mcp_server"],  # 本地启动命令
    }
})

tools = await client.get_tools()
```

---

## 六、如何在 LangChain 中集成 MCP

### 6.1 四步集成流程

```
┌─────────────────────────────────────────────────────────┐
│              MCP 四步集成流程                              │
│                                                         │
│  1. 安装依赖                                             │
│     pip install langchain-mcp-adapters                   │
│                                                         │
│  2. 实例化客户端                                          │
│     client = MultiServerMCPClient({...})                 │
│                                                         │
│  3. 获取工具                                             │
│     tools = await client.get_tools()                     │
│                                                         │
│  4. 传入 Agent                                           │
│     agent = create_agent(llm, tools)                     │
└─────────────────────────────────────────────────────────┘
```

### 6.2 完整示例

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

# 1 & 2: 配置并实例化客户端
client = MultiServerMCPClient({
    "weather_server": {
        "transport": "http",
        "url": "https://weather-mcp.example.com/mcp",
    },
    "local_tools": {
        "transport": "stdio",
        "command": "python",
        "args": ["-m", "my_local_server"],
    },
})

# 3: 获取工具
tools = await client.get_tools()

# 4: 传入 Agent
agent = create_agent(
    model="gpt-4",
    tools=tools,
    system_prompt="你是助手，可以使用外部工具完成任务。"
)

# 调用 Agent
result = await agent.ainvoke({
    "messages": [{"role": "user", "content": "北京天气怎么样？"}]
})
```

---

## 七、高级功能

### 7.1 加载资源和提示词

除了工具，还可以加载服务器暴露的资源和提示词。

```python
# 加载资源
resources = await client.get_resources("weather_server")
for resource in resources:
    print(f"资源: {resource.uri}")
    # 读取内容
    content = await resource.read()
    print(f"内容: {content}")

# 加载提示词模板
prompt = await client.get_prompt(
    "weather_server",
    "weather_report",
    arguments={"city": "北京", "date": "2024-04-24"}
)
print(f"生成的提示词: {prompt}")
```

### 7.2 状态管理

默认情况下，`MultiServerMCPClient` 是**无状态**的——每次工具调用都会新建并清理 `ClientSession`。若服务器依赖会话状态，**必须**显式创建持久会话。

```python
# 无状态（默认）：每次调用新建并清理 Session
tools = await client.get_tools()
# 适合无状态的工具调用

# 有状态：创建持久会话
async with client.session("weather_server") as session:
    tools = load_mcp_tools(session)
    
    # 在会话内多次调用，状态保持
    result1 = await tools[0].ainvoke({"city": "北京"})
    result2 = await tools[0].ainvoke({"city": "上海"})
    # 服务器可以看到完整的会话历史
```

**⚠️ 重要**：若服务器依赖会话状态，必须显式使用 `client.session()`，否则状态会丢失。

### 7.3 Tool Interceptors（工具拦截器）

提供中间件能力，用于注入运行时上下文、修改请求/响应、实现重试/限流，以及通过 `Command` 控制图流程。

```python
from langchain_mcp_adapters.interceptors import ToolInterceptor

class ContextInjector(ToolInterceptor):
    """注入用户上下文的拦截器。"""
    
    def before_call(self, request):
        # 读取运行时上下文
        user_id = request.runtime.context.user_id
        
        # 注入参数
        return request.override(args={
            **request.args,
            "user_id": user_id,
        })

class ResponseValidator(ToolInterceptor):
    """验证工具响应的拦截器。"""
    
    def after_call(self, response):
        # 验证结果
        if not response.content:
            raise ValueError("工具返回空结果")
        return response

# 注册拦截器
client = MultiServerMCPClient(
    {...},
    tool_interceptors=[ContextInjector(), ResponseValidator()],
)
```

**拦截器的"洋葱模型"**：列表中第一个拦截器为最外层（最先执行 `before`，最后执行 `after`）。

```
注册顺序: [A, B, C]

执行流程:
  A.before ──► B.before ──► C.before ──► 实际调用 ──► C.after ──► B.after ──► A.after
```

### 7.4 Callbacks（回调）

处理进度更新、服务器日志及交互式输入请求（Elicitation）。

```python
from langchain_mcp_adapters.callbacks import Callbacks

def on_progress(progress):
    print(f"进度: {progress}")

def on_log(log):
    print(f"服务器日志: {log}")

def on_elicit(request):
    """处理服务器动态索要输入。"""
    # 必须返回 accept、decline 或 cancel 之一
    return {
        "action": "accept",
        "content": {"user_input": "这是用户的回复"},
    }

callbacks = Callbacks(
    on_progress=on_progress,
    on_log=on_log,
    on_elicit=on_elicit,
)

client = MultiServerMCPClient({...}, callbacks=callbacks)
```

**Elicitation 响应规范**：回调必须严格返回以下之一：

| 响应 | 说明 |
|------|------|
| `accept`（带 `content`） | 接受并提供内容 |
| `decline` | 拒绝请求 |
| `cancel` | 终止执行 |

---

## 八、使用场景与最佳实践

### 8.1 典型场景

| 场景 | 传输方式 | 说明 |
|------|----------|------|
| **本地轻量工具** | stdio | 直接运行本地 Python 脚本 |
| **生产远程服务** | HTTP | 结合 Headers/Auth 实现鉴权 |
| **多服务集成** | 混合 | 同时连接多个 HTTP 和 stdio 服务器 |
| **上下文注入** | 拦截器 | 动态注入用户 ID、权限、偏好 |
| **多模态响应** | 原生 | 解析文本、图像等混合内容 |
| **流程编排** | Command | 工具执行后路由到其他 Agent |

### 8.2 最佳实践

#### 实践 1：根据场景选择传输方式

```python
# 本地/轻量工具 → stdio
local_config = {
    "local_tools": {
        "transport": "stdio",
        "command": "python",
        "args": ["-m", "my_local_server"],
    }
}

# 生产/远程服务 → HTTP
remote_config = {
    "weather_server": {
        "transport": "http",
        "url": "https://weather-mcp.example.com/mcp",
        "headers": {"Authorization": "Bearer sk-..."},
    }
}
```

#### 实践 2：利用拦截器注入上下文

```python
class UserInfoInjector(ToolInterceptor):
    """注入用户信息的拦截器。"""
    
    def before_call(self, request):
        # 读取 Runtime Context
        user_id = request.runtime.context.user_id
        role = request.runtime.context.role
        
        # 注入工具参数
        return request.override(args={
            **request.args,
            "user_id": user_id,
            "role": role,
        })
```

#### 实践 3：处理多模态响应

MCP 返回的内容可能包含文本、图像等混合数据，通过 `ToolMessage.content_blocks` 统一解析。

```python
result = await tool.ainvoke({...})

# 解析混合内容
for block in result.content_blocks:
    if block["type"] == "text":
        print(f"文本: {block['text']}")
    elif block["type"] == "image":
        print(f"图像: {block['data'][:50]}...")
```

#### 实践 4：使用 Command 编排流程

在拦截器中返回 `Command` 对象，可在工具执行后直接更新状态或路由至其他 Agent 节点。

```python
from langgraph.types import Command

class RoutingInterceptor(ToolInterceptor):
    """根据工具结果路由到不同 Agent。"""
    
    def after_call(self, response):
        if response.content.get("needs_summary"):
            # 路由到摘要 Agent
            return Command(
                update={"result": response.content},
                goto="summary_agent",
            )
        return response
```

### 8.3 请求不可变原则

修改参数务必使用 `request.override()` 生成新对象，**禁止直接原地修改** `request.args`。

```python
# ✅ 正确：使用 override 生成新对象
def before_call(self, request):
    return request.override(args={
        **request.args,
        "new_param": "value",
    })

# ❌ 错误：原地修改
def before_call(self, request):
    request.args["new_param"] = "value"  # 违反不可变原则！
    return request
```

---

## 九、重要技术细节

### 9.1 默认无状态机制

`MultiServerMCPClient` 默认**每次工具调用都会新建并清理** `ClientSession`。

```
默认行为（无状态）:
  调用 1: 新建 Session → 执行 → 清理
  调用 2: 新建 Session → 执行 → 清理
  调用 3: 新建 Session → 执行 → 清理
  
  服务器看不到完整的会话历史！

显式会话（有状态）:
  async with client.session("server") as session:
      调用 1: 使用同一 Session
      调用 2: 使用同一 Session
      调用 3: 使用同一 Session
      
      服务器可以看到完整的会话历史！
```

### 9.2 结构化内容隐藏

MCP 返回的 `structuredContent` 默认仅保存在 `ToolMessage.artifact` 中（**对 LLM 不可见**）。若需模型感知，必须通过拦截器将其追加至 `content` 字段。

```python
class StructuredContentExposer(ToolInterceptor):
    """将结构化内容暴露给 LLM。"""
    
    def after_call(self, response):
        if response.artifact:
            # 将结构化内容追加到 content，让 LLM 可见
            response.content += f"\n\n结构化数据: {json.dumps(response.artifact)}"
        return response
```

### 9.3 拦截器洋葱模型

列表中第一个拦截器为最外层（最先执行 `before`，最后执行 `after`）。

```
注册: [A, B, C]

执行顺序:
  进入: A.before → B.before → C.before → 实际调用
  退出:                         C.after → B.after → A.after

类比洋葱:
  外层 A 包裹 中层 B 包裹 内层 C 包裹 核心调用
```

---

## 十、常见错误与陷阱

### 陷阱 1：忽略会话状态

**现象**：服务器依赖会话状态，但客户端每次调用都新建 Session，导致状态丢失。

```python
# ❌ 错误：无状态调用，服务器看不到历史
tools = await client.get_tools()
await tools[0].ainvoke({"query": "第一条"})
await tools[0].ainvoke({"query": "第二条"})  # 服务器不知道第一条

# ✅ 正确：使用持久会话
async with client.session("server_name") as session:
    tools = load_mcp_tools(session)
    await tools[0].ainvoke({"query": "第一条"})
    await tools[0].ainvoke({"query": "第二条"})  # 服务器可以看到完整历史
```

### 陷阱 2：结构化内容对 LLM 不可见

**现象**：MCP 返回的结构化数据存储在 `artifact` 中，LLM 看不到。

```python
# 问题：structuredContent 默认存在 artifact 中
# LLM 只能看到 content 字段

# 解决：通过拦截器将 artifact 追加到 content
class ExposeArtifact(ToolInterceptor):
    def after_call(self, response):
        if response.artifact:
            response.content += f"\n{json.dumps(response.artifact)}"
        return response
```

### 陷阱 3：原地修改请求对象

**现象**：直接修改 `request.args` 导致不可预期的行为。

```python
# ❌ 错误：原地修改
def before_call(self, request):
    request.args["user_id"] = "123"  # 违反不可变原则
    return request

# ✅ 正确：使用 override
def before_call(self, request):
    return request.override(args={
        **request.args,
        "user_id": "123",
    })
```

### 陷阱 4：Elicitation 回调返回值不规范

**现象**：回调返回非标准值，导致服务器无法处理。

```python
# ❌ 错误：返回非标准值
def on_elicit(self, request):
    return {"user_input": "some value"}  # 缺少 action 字段

# ✅ 正确：严格返回 accept/decline/cancel
def on_elicit(self, request):
    return {
        "action": "accept",
        "content": {"user_input": "some value"},
    }
```

---

## 十一、总结：MCP 的本质

| 概念 | 比喻 | 作用 |
|------|------|------|
| **MCP 协议** | 万能插座 | 标准化外部服务接入 |
| **Tools** | 手脚 | 执行外部动作 |
| **Resources** | 眼睛和耳朵 | 提供数据上下文 |
| **Prompts** | 剧本 | 标准化提示词模板 |
| **HTTP 传输** | 远程电话线 | 连接远程服务 |
| **stdio 传输** | 本地对讲机 | 连接本地进程 |
| **MultiServerMCPClient** | 调度中心 | 管理多个服务器连接 |
| **拦截器** | 海关检查 | 修改请求/响应、注入上下文 |
| **回调** | 状态报告 | 处理进度、日志、交互输入 |
| **Command** | 交通指挥 | 控制流程路由 |

**MCP 的本质**是**标准化的外部服务接入协议**。它定义了统一的接口，让 LangChain Agent 可以像使用本地工具一样，无缝调用任何支持 MCP 的外部服务——无需为每个服务编写专用集成代码，实现真正的"即插即用"。

```
一句话总结：

MCP 是一个开放协议，核心是"标准化应用程序如何向 LLM 提供
工具和上下文"。它让外部服务接入变得即插即用，
解决集成标准碎片化的问题。
```
