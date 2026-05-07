# LangChain Custom Middleware 核心概念教程


---

## 一、为什么需要自定义中间件？

### 1.1 从"内置不够用"说起

LangChain 提供了很多内置中间件，但实际开发中总会遇到**独特的业务需求**：

```
内置中间件解决的问题：
  ✅ 对话摘要 → SummarizationMiddleware
  ✅ 人工审批 → HumanInTheLoopMiddleware
  ✅ 工具重试 → ToolRetryMiddleware
  ✅ 敏感信息检测 → PIIDetectionMiddleware

内置中间件没解决的问题：
  ❓ "我想在每次模型调用前记录详细的调试日志"
  ❓ "我需要根据用户权限动态过滤可用工具"
  ❓ "我想在工具失败时执行特定的回退逻辑"
  ❓ "我想缓存某些工具的结果，避免重复调用"
  ❓ "我想在模型输出不符合要求时自动重试"

解决方案：自定义中间件！
```

**核心定义**：Custom Middleware 允许开发者通过在 Agent 执行流程的特定节点插入**钩子（hooks）**来拦截和控制执行过程。主要用于实现日志记录、状态验证、动态提示、重试逻辑、缓存和工具过滤等横切关注点，**无需修改核心 Agent 逻辑**。

### 1.2 自定义中间件的本质

```
自定义中间件的本质：
  ✅ 非侵入式：不需要修改 Agent 的核心代码
  ✅ 插件化：可以随意添加、移除、替换
  ✅ 精确控制：在特定的执行节点介入
  ✅ 可组合：多个中间件可以串联使用
  ✅ 高度定制：满足任何独特的业务需求
```

---

## 二、创建自定义中间件的两种方式

### 2.1 装饰器方式（Decorator）—— 轻量快捷

**适用场景**：只需实现单个钩子，或快速原型开发。

```python
from langchain.agents.middleware import before_model

@before_model
def my_custom_middleware(state):
    """在模型推理前执行。"""
    print(f"收到用户输入: {state['messages'][-1].content}")
    return state

# 注册到 Agent
agent = create_agent(
    model="gpt-4",
    tools=[search],
    middleware=[my_custom_middleware],
)
```

**特点**：
- ✅ 轻量快捷，几行代码搞定
- ✅ 适合单一职责的中间件
- ❌ 只能实现一个钩子

### 2.2 类方式（Class-based）—— 强大灵活

**适用场景**：需要组合多个钩子、同时提供同步/异步实现、或带有复杂初始化配置。

```python
from langchain.agents.middleware import AgentMiddleware

class MyCustomMiddleware(AgentMiddleware):
    """自定义中间件类。"""
    
    def before_model(self, state):
        """模型推理前执行。"""
        print("Before model...")
        return state
    
    def after_model(self, state):
        """模型推理后执行。"""
        print("After model...")
        return state
    
    def wrap_tool_call(self, state, tool_call, runtime):
        """工具调用包裹。"""
        try:
            return tool_call.invoke()
        except Exception as e:
            return f"工具失败: {str(e)}"

# 注册到 Agent
agent = create_agent(
    model="gpt-4",
    tools=[search],
    middleware=[MyCustomMiddleware()],
)
```

**特点**：
- ✅ 可以组合多个钩子
- ✅ 支持复杂的初始化配置（`__init__` 方法）
- ✅ 可以同时提供同步和异步实现
- ❌ 代码稍显冗长

---

## 三、钩子（Hooks）的类型和用法

### 3.1 两类核心钩子

```
┌─────────────────────────────────────────────────────────┐
│                    钩子的两种风格                          │
│                                                         │
│  Node-style（节点式）:                                   │
│     在特定点按顺序执行                                     │
│     适用于：日志、验证、状态更新                            │
│                                                         │
│  Wrap-style（包装式）:                                   │
│     包裹在调用周围，控制执行流                              │
│     适用于：重试、降级、缓存                               │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Node-style Hooks —— 节点式钩子

在 Agent 执行的特定点按**顺序**执行，适用于日志、验证和状态更新。

#### `before_agent` / `after_agent`

Agent **开始/结束时**各运行一次。

```python
from langchain.agents.middleware import before_agent, after_agent

@before_agent
def log_session_start(state):
    """记录会话开始。"""
    print(f"=== 会话开始 ===")
    print(f"用户: {state.get('user_id', 'unknown')}")
    return state

@after_agent
def log_session_end(state):
    """记录会话结束。"""
    print(f"=== 会话结束 ===")
    print(f"总消息数: {len(state['messages'])}")
    return state
```

**执行时机**：

```
会话生命周期：
  before_agent ──► [Agent 循环] ──► after_agent
```

#### `before_model`

每次调用模型**前**运行。适合做动态提示词注入、输入验证、日志记录。

```python
from langchain.agents.middleware import before_model
from langchain_core.messages import SystemMessage

@before_model
def inject_user_context(state):
    """根据用户身份注入上下文。"""
    user_role = state.get("user_role", "default")
    
    if user_role == "admin":
        context = SystemMessage("你是管理员助手，可以访问所有数据。")
    else:
        context = SystemMessage("你是普通用户助手。")
    
    # 在消息列表最前面插入系统提示
    return {
        "messages": [context] + state["messages"]
    }
```

**执行时机**：

```
Agent 循环：
  before_model ──► 模型调用 ──► 工具执行 ──► 回到 before_model
```

#### `after_model`

每次模型返回响应**后**运行。适合做输出验证、清理、后处理。

```python
from langchain.agents.middleware import after_model

@after_model
def validate_and_clean(state):
    """验证并清理模型输出。"""
    last_message = state["messages"][-1]
    content = last_message.content
    
    # 清理敏感词
    sensitive_words = ["password", "secret"]
    for word in sensitive_words:
        content = content.replace(word, "[FILTERED]")
    
    state["messages"][-1].content = content
    return state
```

**执行时机**：

```
Agent 循环：
  模型调用 ──► after_model ──► 工具执行 ──► 回到模型调用
```

### 3.3 Wrap-style Hooks —— 包装式钩子

包裹在调用周围，**控制执行流**，适用于重试、降级、缓存。

#### `wrap_model_call`

包裹每次模型调用。可以决定是否执行、执行几次（如重试）或直接短路。

```python
from langchain.agents.middleware import wrap_model_call
import time

@wrap_model_call
def retry_on_failure(state, request, handler):
    """模型调用失败时自动重试。"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # 执行实际的模型调用
            return handler(request)
        except Exception as e:
            if attempt == max_retries - 1:
                raise  # 最后一次失败，抛出异常
            
            # 等待后重试（指数退避）
            wait_time = 2 ** attempt
            print(f"模型调用失败，{wait_time}秒后重试...")
            time.sleep(wait_time)
```

**工作原理**：

```
wrap_model_call 的执行流：
  中间件代码前 ──► handler(request) ──► 中间件代码后
                       │
                       ▼
                  实际模型调用
```

**另一个示例：缓存**

```python
from langchain.agents.middleware import wrap_model_call

_cache = {}

@wrap_model_call
def cache_model_calls(state, request, handler):
    """缓存模型调用结果。"""
    # 生成缓存键
    cache_key = str(request.messages[-1].content)
    
    if cache_key in _cache:
        print("缓存命中！")
        return _cache[cache_key]  # 直接返回缓存
    
    # 执行实际调用
    response = handler(request)
    
    # 存入缓存
    _cache[cache_key] = response
    return response
```

#### `wrap_tool_call`

包裹每次工具调用。适合做错误处理、重试、日志记录、权限检查。

```python
from langchain.agents.middleware import wrap_tool_call

@wrap_tool_call
def handle_tool_errors(state, tool_call, runtime):
    """处理工具调用异常。"""
    try:
        result = tool_call.invoke()
        return result
    except TimeoutError:
        return "工具调用超时，请重试或换其他方式。"
    except Exception as e:
        return f"工具调用失败：{str(e)}"
```

**另一个示例：工具权限检查**

```python
@wrap_tool_call
def check_tool_permissions(state, tool_call, runtime):
    """检查用户是否有权使用该工具。"""
    user_role = state.get("user_role", "default")
    tool_name = tool_call.name
    
    # 定义权限矩阵
    restricted_tools = {
        "delete_record": "admin",
        "send_email": "admin",
    }
    
    if tool_name in restricted_tools:
        required_role = restricted_tools[tool_name]
        if user_role != required_role:
            return f"权限不足：{tool_name} 需要 {required_role} 角色。"
    
    # 权限通过，执行工具
    return tool_call.invoke()
```

### 3.4 便利钩子：`@dynamic_prompt`

用于在运行时动态生成或修改系统提示。

```python
from langchain.agents.middleware import dynamic_prompt
from langchain_core.messages import SystemMessage

@dynamic_prompt
def generate_dynamic_prompt(state, request):
    """根据运行时状态生成系统提示。"""
    user_role = state.get("user_role", "default")
    user_language = state.get("language", "zh")
    
    base_prompt = "你是一个有用的助手。"
    
    if user_role == "admin":
        base_prompt += "你是管理员助手，可以执行删除、修改等操作。"
    else:
        base_prompt += "你是普通用户助手，只能查询信息。"
    
    if user_language == "en":
        base_prompt += " Please respond in English."
    else:
        base_prompt += " 请用中文回复。"
    
    return SystemMessage(content=base_prompt)
```

**⚠️ 重要规则**：

- `ModelRequest.system_message` 始终是 `SystemMessage` 对象
- 修改提示词时必须通过 `.content_blocks` 列表追加，**不可直接覆盖字符串**
- 这样可保留结构并支持 Anthropic 的 `cache_control` 缓存指令

```python
# ✅ 正确：通过 content_blocks 追加
@dynamic_prompt
def good_prompt(state, request):
    request.system_message.content_blocks.append({
        "type": "text",
        "text": "额外的上下文信息..."
    })
    return request.system_message

# ❌ 错误：直接覆盖字符串
@dynamic_prompt
def bad_prompt(state, request):
    request.system_message = "你是助手"  # 破坏结构！
    return request.system_message
```

---

## 四、执行顺序：洋葱模型

### 4.1 核心概念

假设注册顺序为 `[middleware1, middleware2, middleware3]`，执行顺序如下：

```
┌─────────────────────────────────────────────────────────┐
│                    洋葱模型执行顺序                        │
│                                                         │
│  1. before_agent（正序）: 1 → 2 → 3                     │
│                                                         │
│  2. Agent 循环开始：                                      │
│     before_model（正序）: 1 → 2 → 3                      │
│                                                         │
│     wrap_model_call（嵌套）:                              │
│        1 包裹 2，2 包裹 3，3 包裹实际模型调用              │
│                                                         │
│     wrap_tool_call（嵌套）:                               │
│        1 包裹 2，2 包裹 3，3 包裹实际工具调用              │
│                                                         │
│     after_model（逆序）: 3 → 2 → 1                      │
│                                                         │
│  3. after_agent（逆序）: 3 → 2 → 1                      │
└─────────────────────────────────────────────────────────┘
```

### 4.2 详细执行流程

```
完整执行流程：

用户输入
  │
  ▼
┌─────────────────┐
│ before_agent 1  │  ← 正序
└────────┬────────┘
         ▼
┌─────────────────┐
│ before_agent 2  │
└────────┬────────┘
         ▼
┌─────────────────┐
│ before_agent 3  │
└────────┬────────┘
         ▼
    ┌────────┐
    │ Agent  │ ← 循环开始
    │ 循环   │
    └───┬────┘
        │
        ▼
┌─────────────────┐
│ before_model 1  │  ← 正序
└────────┬────────┘
         ▼
┌─────────────────┐
│ before_model 2  │
└────────┬────────┘
         ▼
┌─────────────────┐
│ before_model 3  │
└────────┬────────┘
         ▼
┌─────────────────────────────────┐
│ wrap_model_call 1（外层）        │
│   ┌───────────────────────────┐ │
│   │ wrap_model_call 2（中层）  │ │
│   │   ┌─────────────────────┐ │ │
│   │   │ wrap_model_call 3    │ │ │
│   │   │   ┌───────────────┐ │ │ │
│   │   │   │  实际模型调用  │ │ │ │
│   │   │   └───────────────┘ │ │ │
│   │   └─────────────────────┘ │ │
│   └───────────────────────────┘ │
└─────────────────────────────────┘
         ▼
┌─────────────────┐
│ after_model 3   │  ← 逆序
└────────┬────────┘
         ▼
┌─────────────────┐
│ after_model 2   │
└────────┬────────┘
         ▼
┌─────────────────┐
│ after_model 1   │
└────────┬────────┘
         ▼
    (工具执行或回到模型调用)
        │
        ▼
┌─────────────────┐
│ after_agent 3   │  ← 逆序
└────────┬────────┘
         ▼
┌─────────────────┐
│ after_agent 2   │
└────────┬────────┘
         ▼
┌─────────────────┐
│ after_agent 1   │
└────────┬────────┘
         ▼
      输出响应
```

### 4.3 Wrap 嵌套的执行顺序

```python
# wrap_model_call 的嵌套执行
@wrap_model_call
def middleware_1(state, request, handler):
    print("1 before")
    response = handler(request)
    print("1 after")
    return response

@wrap_model_call
def middleware_2(state, request, handler):
    print("2 before")
    response = handler(request)
    print("2 after")
    return response

@wrap_model_call
def middleware_3(state, request, handler):
    print("3 before")
    response = handler(request)
    print("3 after")
    return response

# 输出顺序：
# 1 before
#   2 before
#     3 before
#       实际模型调用
#     3 after
#   2 after
# 1 after
```

**类比理解**：

```
洋葱模型：
  外层（middleware_1）先接触
  中层（middleware_2）其次
  内层（middleware_3）最后
  
  进入时：外 → 中 → 内 → 核心
  退出时：内 → 中 → 外
```

---

## 五、状态管理机制

### 5.1 自定义状态（State Schema）

可以继承 `AgentState` 并使用 `NotRequired` 添加字段，跨钩子共享数据。

```python
from typing import TypedDict, NotRequired
from langchain.agents import AgentState

class CustomState(AgentState):
    # 添加自定义字段
    user_context: NotRequired[dict]
    retry_count: NotRequired[int]
    cache_hits: NotRequired[int]

# 在中间件中访问和更新
@before_model
def track_user_context(state):
    state["user_context"] = {"session_start": "now"}
    return state

@after_model
def increment_cache_hits(state):
    hits = state.get("cache_hits", 0)
    return {"cache_hits": hits + 1}
```

### 5.2 状态更新机制差异

| 钩子类型 | 更新方式 | 说明 |
|----------|----------|------|
| **Node-style** | 直接返回 `dict` | 通过图的 reducer 合并 |
| **Wrap-style（模型调用）** | 返回 `ExtendedModelResponse` 包含 `Command` | 需要特殊包装 |
| **Wrap-style（工具调用）** | 直接返回 `Command` | 直接返回即可 |

**多中间件组合规则**：

- **消息（messages）**：默认**累加**
- **非 reducer 字段**：遵循**外层优先（Outer wins）**原则
- **重试逻辑**：能安全丢弃旧调用产生的 Command

---

## 六、高级特性

### 6.1 提前退出（Agent Jumps）

返回包含 `"jump_to"` 的字典，可以让 Agent 跳转到指定节点。

```python
from langchain.agents.middleware import before_model, hook_config

@hook_config(can_jump_to=["end", "tools"])
@before_model
def emergency_stop(state):
    """紧急停止：检测到特定关键词时终止执行。"""
    last_msg = state["messages"][-1].content
    
    if "STOP" in last_msg:
        return {
            "jump_to": "end",  # 跳转到结束
            "messages": [{"role": "assistant", "content": "已停止执行。"}]
        }
    
    return state

@hook_config(can_jump_to=["tools"])
@after_model
def force_tool_execution(state):
    """强制工具执行：检测到特定条件时跳转到工具。"""
    if should_use_tool(state):
        return {"jump_to": "tools"}
    return state
```

**⚠️ 必须使用 `@hook_config(can_jump_to=[...])` 显式声明允许的跳转目标**。

**可用的跳转目标**：

| 目标 | 说明 |
|------|------|
| `"end"` | 终止 Agent 执行 |
| `"tools"` | 跳转到工具执行节点 |
| `"model"` | 跳转到模型调用节点 |

### 6.2 动态覆盖：切换模型和过滤工具

使用 `request.override()` 可在运行时动态切换模型或过滤工具。

```python
from langchain.agents.middleware import before_model

@before_model
def dynamic_model_switch(state, request):
    """根据任务复杂度切换模型。"""
    task_complexity = estimate_complexity(state["messages"][-1].content)
    
    if task_complexity == "high":
        # 切换到强模型
        request.override(model="gpt-4")
    else:
        # 使用轻量模型
        request.override(model="gpt-4-mini")
    
    return request

@before_model
def filter_tools_by_permission(state, request):
    """根据用户权限过滤工具。"""
    user_role = state.get("user_role", "default")
    
    if user_role != "admin":
        # 移除管理员专用工具
        request.override(
            tools=[t for t in request.tools if t.name not in ["delete_record", "admin_tool"]]
        )
    
    return request
```

---

## 七、完整示例：综合自定义中间件

### 7.1 日志记录中间件

```python
from langchain.agents.middleware import before_agent, after_agent, before_model, after_model
import logging
import time

logging.basicConfig(level=logging.INFO)

@before_agent
def log_session_start(state):
    """记录会话开始。"""
    logging.info("=== 新会话开始 ===")
    state["_session_start"] = time.time()
    return state

@after_agent
def log_session_end(state):
    """记录会话结束。"""
    duration = time.time() - state.get("_session_start", time.time())
    logging.info(f"=== 会话结束，持续时间: {duration:.1f}秒 ===")
    return state

@before_model
def log_model_input(state):
    """记录模型输入。"""
    last_msg = state["messages"][-1]
    logging.info(f"模型输入: {last_msg.content[:100]}...")
    return state

@after_model
def log_model_output(state):
    """记录模型输出。"""
    last_msg = state["messages"][-1]
    logging.info(f"模型输出: {last_msg.content[:100]}...")
    return state
```

### 7.2 工具重试中间件

```python
from langchain.agents.middleware import wrap_tool_call
import time

@wrap_tool_call
def tool_retry(state, tool_call, runtime):
    """工具调用失败时自动重试。"""
    max_retries = 3
    initial_delay = 1
    
    for attempt in range(max_retries):
        try:
            return tool_call.invoke()
        except Exception as e:
            if attempt == max_retries - 1:
                # 最后一次失败
                return f"工具 {tool_call.name} 调用失败：{str(e)}"
            
            # 等待后重试（指数退避 + 抖动）
            delay = initial_delay * (2 ** attempt)
            import random
            jitter = random.uniform(0, 1)
            time.sleep(delay + jitter)
```

### 7.3 动态提示词中间件

```python
from langchain.agents.middleware import dynamic_prompt
from langchain_core.messages import SystemMessage

@dynamic_prompt
def time_aware_prompt(state, request):
    """根据时间生成不同的提示词。"""
    import datetime
    now = datetime.datetime.now()
    hour = now.hour
    
    time_context = ""
    if 6 <= hour < 12:
        time_context = "现在是早上。"
    elif 12 <= hour < 18:
        time_context = "现在是下午。"
    else:
        time_context = "现在是晚上。"
    
    # 追加到系统提示
    request.system_message.content_blocks.append({
        "type": "text",
        "text": time_context
    })
    
    return request.system_message
```

---

## 八、最佳实践与设计哲学

### 8.1 单一职责

每个中间件只专注做一件事。

```python
# ✅ 好的中间件：职责单一
@before_model
def log_input(state):
    """只负责记录日志。"""
    ...

@after_model
def validate_output(state):
    """只负责验证输出。"""
    ...

# ❌ 坏的中间件：职责过多
@before_model
def do_everything(state):
    """又记录日志、又验证、又修改..."""
    ...
```

### 8.2 优雅处理异常

避免中间件错误导致整个 Agent 崩溃。

```python
# ✅ 优雅处理
@after_model
def safe_validate(state):
    """安全验证，异常时不崩溃。"""
    try:
        # 验证逻辑
        return state
    except Exception as e:
        # 记录错误，但不崩溃
        import logging
        logging.error(f"验证失败: {e}")
        return state  # 返回原状态，继续执行

# ❌ 危险处理
@after_model
def unsafe_validate(state):
    """异常会导致整个 Agent 崩溃。"""
    validate(state)  # 如果抛出异常，整个流程崩溃
    return state
```

### 8.3 按需选钩

| 需求 | 推荐钩子 | 原因 |
|------|----------|------|
| 日志记录 | `before_model`, `after_model` | Node-style，顺序执行 |
| 输入验证 | `before_model` | 在模型调用前验证 |
| 输出清理 | `after_model` | 在模型调用后处理 |
| 重试逻辑 | `wrap_model_call`, `wrap_tool_call` | 需要控制执行次数 |
| 缓存 | `wrap_model_call` | 可以短路返回 |
| 动态提示词 | `@dynamic_prompt` | 专为提示词设计 |

### 8.4 明确定义状态

清晰文档化任何自定义的 State 属性，避免命名冲突。

```python
class CustomState(AgentState):
    """
    自定义状态，包含以下字段：
    - user_context: dict - 用户上下文信息
    - retry_count: int - 重试次数计数器
    - cache_hits: int - 缓存命中次数
    """
    user_context: NotRequired[dict]
    retry_count: NotRequired[int]
    cache_hits: NotRequired[int]
```

### 8.5 独立测试

在集成前对中间件进行单元测试。

```python
def test_log_input_middleware():
    state = {"messages": [HumanMessage(content="测试")]}
    result = log_input(state)
    assert result == state  # 确保不改变状态

def test_retry_middleware():
    # 模拟工具调用失败
    mock_tool = Mock(side_effect=[TimeoutError, "success"])
    # 验证重试逻辑
    ...
```

### 8.6 注意注册顺序

关键中间件（如权限校验、限流）应放在列表最前面。

```python
# ✅ 推荐：关键中间件在前
middleware = [
    PermissionCheckMiddleware(),     # 权限检查（最先）
    RateLimitMiddleware(),           # 限流
    LoggingMiddleware(),             # 日志
    SummarizationMiddleware(),       # 摘要
]

# ❌ 不推荐：关键中间件在后
middleware = [
    SummarizationMiddleware(),
    LoggingMiddleware(),
    PermissionCheckMiddleware(),     # 权限检查应该最先执行
]
```

### 8.7 优先使用内置中间件

检查官方是否已提供现成实现。

```python
# ✅ 推荐：优先使用内置
from langchain.agents.middleware import (
    SummarizationMiddleware,
    HumanInTheLoopMiddleware,
    ToolRetryMiddleware,
)

# ❌ 不推荐：重复造轮子
@before_model
def my_summarization(state):
    # 自己实现摘要压缩...
```

---

## 九、总结：自定义中间件的本质

| 概念 | 比喻 | 作用 |
|------|------|------|
| **Node-style Hook** | 检查站 | 在特定点顺序执行 |
| **Wrap-style Hook** | 包装器 | 包裹调用，控制执行流 |
| **before_model** | 事前准备 | 模型调用前的拦截 |
| **after_model** | 事后处理 | 模型调用后的拦截 |
| **wrap_model_call** | 模型包装器 | 重试、缓存、降级 |
| **wrap_tool_call** | 工具包装器 | 错误处理、权限检查 |
| **dynamic_prompt** | 动态剧本 | 运行时生成提示词 |
| **洋葱模型** | 洋葱层 | 执行顺序的可视化 |

**自定义中间件的本质**是**高度定制化的拦截器**。它通过在 Agent 核心循环的关键节点暴露钩子，让你能够满足任何独特的业务需求——从日志记录、权限检查、缓存、重试到动态模型切换——一切皆可通过中间件实现，且无需修改核心代码。

---
