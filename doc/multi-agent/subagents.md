

## 核心概念

### Subagents 是什么？

Subagents 是一种**架构模式**，由一个中心主智能体（Supervisor）通过调用工具的方式协调多个子智能体。

```
用户 → 主智能体（Supervisor） → 调用子智能体工具 → 子智能体执行 → 返回结果 → 主智能体整合 → 用户
```

### 为什么需要 Subagents？

| 价值 | 说明 |
|------|------|
| **防止上下文膨胀** | 每个子智能体在独立的干净上下文窗口中运行 |
| **领域解耦** | 每个专家专注自己的领域，不加载无关知识 |
| **集中控制** | 主智能体掌握全局，子智能体专注执行 |
| **团队并行开发** | 不同团队可独立开发子智能体 |

### 两种工具暴露模式

#### 模式一：Tool per Agent（每智能体一工具）

```python
@tool
def call_research_agent(query: str) -> str:
    """调用市场研究专家..."""
    result = research_agent.invoke({"messages": [...]})
    return result["messages"][-1].content

@tool
def call_finance_agent(query: str) -> str:
    """调用财务分析专家..."""
    ...

# 主智能体工具列表
supervisor = create_agent(tools=[call_research_agent, call_finance_agent])
```

**适用场景**: <10 个静态子智能体

#### 模式二：Single Dispatch（单一调度工具）

```python
SUBAGENTS = {
    "research": research_agent,
    "writer": writer_agent,
    ...
}

@tool
def task(agent_name: str, query: str) -> str:
    """根据名称动态路由到子智能体"""
    agent = SUBAGENTS[agent_name]
    result = agent.invoke({"messages": [...]})
    return result["messages"][-1].content

# 主智能体只占用 1-2 个工具位置
supervisor = create_agent(tools=[task, list_agents])
```

**适用场景**: >10 个或动态增减的子智能体

## 最佳实践

1. **子智能体名称和描述需清晰**: 直接影响主智能体的路由准确率
2. **输出约束**: 在子智能体系统提示中明确要求输出格式，确保主智能体能获取关键信息
3. **上下文隔离**: 子智能体默认无状态，每次调用使用全新上下文
4. **状态持久化**: 如需跨调用保持记忆，使用 `checkpointer=MemorySaver()` 和固定 `thread_id`
5. **少量用 Tool per Agent，大量用 Single Dispatch**: 避免主智能体工具列表过长
