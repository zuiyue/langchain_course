# LangChain Structured Output 核心概念教程


---

## 一、什么是结构化输出？

### 1.1 从自由文本到结构化数据

想象你在做一个客户管理系统：

```
传统 LLM 输出（自由文本）:
  输入："提取这条信息的联系人：张三，邮箱 zhangsan@example.com，电话 13800138000"
  输出："好的，我找到了以下信息：
         联系人是张三，他的邮箱是 zhangsan@example.com，
         电话号码是 13800138000。"
  
  问题：文本格式不固定，程序无法直接提取字段！

结构化输出:
  输入："提取这条信息的联系人：张三，邮箱 zhangsan@example.com，电话 13800138000"
  输出: {
      "name": "张三",
      "email": "zhangsan@example.com",
      "phone": "13800138000"
  }
  
  好处：格式固定，程序可以直接使用！
```

**核心定义**：结构化输出（Structured Output）允许 AI 智能体以**特定、可预测的格式**返回数据，替代传统的自然语言响应。返回的数据直接为 **JSON 对象、Pydantic 模型或 Python 数据类**，应用程序可直接使用而无需额外解析。

### 1.2 为什么需要结构化输出？

```
场景对比：

❌ 不用结构化输出：
  用户输入："请分析这条评论的情感"
  AI 输出："我认为这条评论的情感是积极的，因为用户提到了'很好'、'满意'等词汇..."
  
  程序处理：需要再用 NLP 或正则来解析这段文本 ──► 不可靠！

✅ 使用结构化输出：
  用户输入："请分析这条评论的情感"
  AI 输出: {
      "sentiment": "positive",
      "confidence": 0.92,
      "keywords": ["很好", "满意"]
  }
  
  程序处理：直接读取 result["sentiment"] ──► 完美！
```

### 1.3 在 Agent 中的集成方式

通过 `create_agent` 的 `response_format` 参数控制，LangChain 会自动处理结构化输出：

```python
from pydantic import BaseModel
from langchain.agents import create_agent

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    response_format=ContactInfo,  # 定义期望的输出格式
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "提取：张三，邮箱 zhangsan@example.com"}]
})

# 结构化响应在 structured_response 键中
print(result["structured_response"])
# ContactInfo(name='张三', email='zhangsan@example.com', phone='')
```

**工作流程**：

```
1. 开发者定义期望的输出 Schema
        │
        ▼
2. Agent 调用模型，附带 Schema 信息
        │
        ▼
3. 模型生成符合 Schema 的数据
        │
        ▼
4. LangChain 捕获并验证数据
        │
        ▼
5. 放入 Agent 状态的 'structured_response' 键中返回
```

---

## 二、结构化输出的 Schema 格式

LangChain 支持多种 Schema 格式，各有特点：

### 2.1 Pydantic 模型（推荐）—— 最强类型安全

```python
from pydantic import BaseModel, Field

class ContactInfo(BaseModel):
    """联系人的信息。"""
    name: str = Field(description="姓名")
    email: str = Field(description="邮箱地址")
    phone: str = Field(description="电话号码")
    age: int = Field(description="年龄", ge=0, le=150)  # ge=最小, le=最大
    is_vip: bool = Field(description="是否为 VIP 客户")

# 返回值是 Pydantic 实例，带字段校验
contact = result["structured_response"]
print(contact.name)       # "张三"
print(contact.email)      # "zhangsan@example.com"
print(contact.is_vip)     # True
```

**优势**：
- ✅ 字段级别校验（如范围、格式）
- ✅ IDE 自动补全和类型提示
- ✅ 返回 Pydantic 实例，可用 `.` 访问字段

### 2.2 Dataclass —— 轻量级选择

```python
from dataclasses import dataclass

@dataclass
class ContactInfo:
    name: str
    email: str
    phone: str

# 返回值是字典
contact = result["structured_response"]
print(contact["name"])  # "张三"
```

**特点**：无需安装 Pydantic，但返回的是字典而非对象。

### 2.3 TypedDict —— 类型化字典

```python
from typing import TypedDict

class ContactInfo(TypedDict):
    name: str
    email: str
    phone: str

# 返回值是字典
contact = result["structured_response"]
print(contact["name"])  # "张三"
```

### 2.4 JSON Schema —— 最灵活

```python
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "姓名"},
        "email": {"type": "string", "description": "邮箱"},
        "age": {"type": "integer", "minimum": 0, "maximum": 150}
    },
    "required": ["name", "email"]
}

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    response_format=schema,
)
```

**适用场景**：需要动态生成 Schema 或与外部系统集成时。

### 2.5 Union 类型 —— 多格式选择

```python
from typing import Union
from pydantic import BaseModel

class Person(BaseModel):
    """人员信息。"""
    name: str
    role: str  # "person"

class Organization(BaseModel):
    """组织机构。"""
    name: str
    type: str  # "org"
    member_count: int

# 模型根据上下文自动选择最匹配的 Schema
agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    response_format=Union[Person, Organization],
)

# 输入提到个人 → 返回 Person
# 输入提到公司 → 返回 Organization
```

---

## 三、结构化输出的策略

### 3.1 三种策略

| 策略 | 原理 | 可靠性 | 适用场景 |
|------|------|--------|----------|
| **ProviderStrategy** | 利用大模型提供商 API 的原生功能 | 最高 | 支持原生的模型（OpenAI、Anthropic、Gemini 等） |
| **ToolStrategy** | 通过工具调用模拟结构化输出 | 高 | 不支持原生结构化输出的模型 |
| **自动选择** | LangChain 根据模型能力动态选择 | 最佳体验 | 推荐，无需手动指定 |

### 3.2 ProviderStrategy —— 原生支持

部分提供商通过 API **原生支持**结构化输出，这是**最可靠的方式**，由提供商在服务端强制校验 Schema。

**支持的提供商**：

| 提供商 | 模型 | 支持情况 |
|--------|------|----------|
| OpenAI | GPT-4o, GPT-4.1 | ✅ 原生 |
| Anthropic | Claude Sonnet, Opus | ✅ 原生 |
| Google | Gemini | ✅ 原生 |
| xAI | Grok | ✅ 原生 |

```python
from langchain.agents.middleware import ProviderStrategy

# 显式使用（通常不需要）
agent = create_agent(
    model="openai:gpt-4o",
    response_format=ProviderStrategy(ContactInfo),
)
```

**严格模式（Strict Mode）**：

部分提供商（如 OpenAI、xAI）支持强制模型严格遵循 Schema，不允许多余字段：

```python
from langchain.agents.middleware import ProviderStrategy

# 启用严格模式（需 langchain>=1.2）
agent = create_agent(
    model="openai:gpt-4o",
    response_format=ProviderStrategy(ContactInfo, strict=True),
)
```

### 3.3 ToolStrategy —— 通用回退

通过工具调用实现结构化输出，适用于**所有支持工具调用的模型**。

```python
from langchain.agents.middleware import ToolStrategy

agent = create_agent(
    model="some-model-without-native-structured-output",
    response_format=ToolStrategy(ContactInfo),
)
```

**原理**：

```
LangChain 创建一个"虚拟工具"：
  工具名: "extract_contact_info"
  工具参数: ContactInfo 的 JSON Schema
  工具描述: "以结构化格式输出联系人信息"

模型"调用"这个工具 ──► 返回符合 Schema 的数据
```

**自定义工具消息**：

通过 `tool_message_content` 参数，可自定义结构化输出生成后在对话历史中显示的 `ToolMessage` 内容：

```python
agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    response_format=ToolStrategy(
        ContactInfo,
        tool_message_content="已提取联系人信息。"  # 替代默认的工具消息
    ),
)
```

### 3.4 自动选择（推荐）—— 最佳体验

**推荐做法**：直接传递 Schema 类型，无需显式包装策略，LangChain 会自动优化路由。

```python
# ✅ 推荐：直接传递类型，LangChain 自动选择最优策略
agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    response_format=ContactInfo,  # 自动选择 ProviderStrategy 或 ToolStrategy
)
```

**自动选择逻辑**：

```
LangChain (>=1.1) 会动态读取模型的 profile 数据：

1. 检查模型是否支持原生结构化输出？
   ├── 是 ──► 使用 ProviderStrategy
   └── 否
       └── 检查模型是否支持工具调用？
           ├── 是 ──► 使用 ToolStrategy
           └── 否 ──► 抛出错误
```

**手动覆盖**：也可以通过 `init_chat_model(profile=...)` 手动覆盖配置。

---

## 四、智能错误处理

模型生成结构化输出时可能出错（如格式不符合 Schema）。LangChain 提供**自动重试机制**。

### 4.1 默认行为

```python
# 默认 handle_errors=True，自动重试
agent = create_agent(
    model="openai:gpt-4o",
    response_format=ContactInfo,
)
```

### 4.2 自定义错误处理

```python
# 方式 1：始终使用自定义错误提示重试
agent = create_agent(
    model="openai:gpt-4o",
    response_format=ContactInfo,
    handle_errors="请确保输出是有效的 JSON，包含 name, email, phone 字段。",
)

# 方式 2：仅捕获特定异常并重试
agent = create_agent(
    model="openai:gpt-4o",
    response_format=ContactInfo,
    handle_errors=ValueError,  # 或 (ValueError, TypeError)
)

# 方式 3：自定义函数根据异常类型返回特定错误提示
def custom_error_handler(exception: Exception) -> str:
    if isinstance(exception, ValueError):
        return "字段值不符合要求，请重新生成。"
    elif isinstance(exception, KeyError):
        return "缺少必需字段，请确保包含所有必填字段。"
    return "输出格式错误，请重试。"

agent = create_agent(
    model="openai:gpt-4o",
    response_format=ContactInfo,
    handle_errors=custom_error_handler,
)

# 方式 4：禁用重试，异常直接向上抛出
agent = create_agent(
    model="openai:gpt-4o",
    response_format=ContactInfo,
    handle_errors=False,
)
```

### 4.3 常见错误场景

#### 场景 1：多输出错误

模型错误地同时调用了多个结构化输出工具。系统会拦截并提示"仅期望一个响应"，引导模型修正重试。

```
错误流程：
  模型同时调用 extract_contact 和 extract_org_info  ──► 错误！
  ──► LangChain 拦截："仅期望一个响应"
  ──► 模型修正：只调用一个工具
  ──► 成功
```

#### 场景 2：Schema 校验错误

输出值不符合约束（如评分超出 `1-5` 范围）。系统会返回具体的 Pydantic/JSON Schema 验证错误详情，提示模型重新生成合法值。

```
错误流程：
  模型输出: {"rating": 10}  ← 超出范围！
  ──► Pydantic 校验失败："rating 必须在 1-5 之间"
  ──► 错误详情返回给模型
  ──► 模型修正: {"rating": 5}
  ──► 成功
```

---

## 五、使用场景与最佳实践

### 5.1 典型场景

#### 场景 1：信息提取

```python
class ContactInfo(BaseModel):
    """联系人信息。"""
    name: str = Field(description="姓名")
    email: str = Field(description="邮箱地址")
    phone: str = Field(description="电话号码")

agent = create_agent(
    model="openai:gpt-4o",
    response_format=ContactInfo,
    system_prompt="你是一个信息提取助手，从文本中提取联系人信息。"
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "我的联系人叫张三，邮箱是 zhangsan@example.com，电话 13800138000"}]
})

contact = result["structured_response"]
print(contact.name)   # "张三"
print(contact.email)  # "zhangsan@example.com"
```

#### 场景 2：情感分析

```python
class SentimentAnalysis(BaseModel):
    """情感分析结果。"""
    sentiment: str = Field(description="情感倾向", enum=["positive", "negative", "neutral"])
    confidence: float = Field(description="置信度，0-1 之间", ge=0, le=1)
    keywords: list[str] = Field(description="关键词列表")

agent = create_agent(
    model="openai:gpt-4o",
    response_format=SentimentAnalysis,
    system_prompt="分析用户评论的情感倾向。"
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "这个产品真的很好用，非常满意！"}]
})

analysis = result["structured_response"]
print(analysis.sentiment)   # "positive"
print(analysis.confidence)  # 0.95
print(analysis.keywords)    # ["很好用", "非常满意"]
```

#### 场景 3：任务提取

```python
class TodoItem(BaseModel):
    """待办事项。"""
    task: str = Field(description="任务描述")
    assignee: str = Field(description="负责人")
    deadline: str = Field(description="截止日期，YYYY-MM-DD 格式")
    priority: str = Field(description="优先级", enum=["low", "medium", "high"])

agent = create_agent(
    model="openai:gpt-4o",
    response_format=TodoItem,
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "请小明在周五前完成代码审查，这是高优先级任务"}]
})

todo = result["structured_response"]
print(todo.task)        # "代码审查"
print(todo.assignee)    # "小明"
print(todo.deadline)    # "2024-XX-XX（周五）"
print(todo.priority)    # "high"
```

#### 场景 4：客户投诉分类

```python
class Complaint(BaseModel):
    """客户投诉分类。"""
    category: str = Field(description="投诉类别", enum=["product", "service", "billing", "technical"])
    severity: str = Field(description="严重程度", enum=["low", "medium", "high", "critical"])
    summary: str = Field(description="问题摘要")

agent = create_agent(
    model="openai:gpt-4o",
    response_format=Complaint,
)
```

### 5.2 最佳实践

#### 实践 1：优先使用原生结构化输出

当模型支持时，优先使用提供商原生的结构化输出功能，因其具备高可靠性与严格的服务端校验。

```python
# ✅ 推荐：使用支持原生结构化输出的模型
agent = create_agent(
    model="openai:gpt-4o",  # 支持 ProviderStrategy
    response_format=ContactInfo,
)
```

#### 实践 2：Schema 设计要精准

```python
# ✅ 好的 Schema：字段描述清晰，有约束
class ContactInfo(BaseModel):
    """联系人信息。"""
    name: str = Field(description="姓名，如'张三'、'李四'")
    email: str = Field(description="邮箱地址，必须符合 email 格式")
    phone: str = Field(description="11 位手机号码")
    age: int = Field(description="年龄", ge=0, le=150)

# ❌ 差的 Schema：描述模糊，无约束
class BadContact(BaseModel):
    name: str  # 没有描述
    info: str  # "info" 太模糊
```

#### 实践 3：处理可选字段

```python
from typing import Optional

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None  # 可选字段
    notes: Optional[str] = None

# 模型可以不返回 phone 和 notes
```

#### 实践 4：注意兼容性限制

若同时指定了普通工具，模型必须支持"工具调用"与"结构化输出"**并发使用**。

```python
# Agent 同时使用工具和结构化输出
agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_web],             # 普通工具
    response_format=ContactInfo,    # 结构化输出
)

# 确保模型支持这两种能力同时使用
```

---

## 六、总结：结构化输出的本质

| 概念 | 比喻     | 作用 |
|------|--------|------|
| **Schema** | 表格模板   | 定义期望的输出格式 |
| **Pydantic** | 带校验的表格 | 字段级别验证 |
| **ProviderStrategy** | 原厂质检   | 服务商端强制校验 |
| **ToolStrategy** | 模拟表格   | 通过工具调用实现 |
| **自动选择** | 智能路由   | 自动选最优策略 |
| **handle_errors** | 纠错机制   | 自动重试修正 |
| **structured_response** | 成品     | 验证后的结构化数据 |

**结构化输出的本质**是**让 AI 的输出从"不可解析的自然语言"变成"可直接使用的结构化数据"**。它通过 Schema 定义、自动校验和智能重试，确保模型返回的数据格式稳定、可靠，应用程序可以直接消费。

---
