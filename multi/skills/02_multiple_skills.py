

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL


# ============================================
# 步骤 1：定义技能库（支持动态管理）
# ============================================

# 初始技能库
SKILLS = {
    "python": {
        "name": "Python",
        "description": "Python 编程专家",
        "enabled": True,
        "prompt": """你是 Python 专家。
请帮助用户解决 Python 相关问题，提供代码示例和最佳实践。""",
    },
    "javascript": {
        "name": "JavaScript",
        "description": "JavaScript/前端开发专家",
        "enabled": True,
        "prompt": """你是 JavaScript 和前端开发专家。
请帮助用户解决 JavaScript、React、Vue 等相关问题。""",
    },
    "sql": {
        "name": "SQL",
        "description": "SQL 数据库专家",
        "enabled": True,
        "prompt": """你是 SQL 数据库专家。
请帮助用户编写 SQL 查询、设计数据库和优化性能。""",
    },
    "devops": {
        "name": "DevOps",
        "description": "DevOps/运维专家（已禁用）",
        "enabled": False,  # 默认禁用
        "prompt": """你是 DevOps 专家。
请帮助用户解决 CI/CD、Docker、Kubernetes 等问题。""",
    },
}


# ============================================
# 步骤 2：创建技能管理工具
# ============================================

@tool
def load_skill(skill_name: str) -> str:
    """加载指定技能的专用提示词。

    Available skills (enabled only):
    - python: Python 编程专家
    - javascript: JavaScript/前端开发专家
    - sql: SQL 数据库专家

    Args:
        skill_name: 技能名称
    """
    print(f"\n   加载技能: {skill_name}")

    if skill_name not in SKILLS:
        available = [name for name, s in SKILLS.items() if s["enabled"]]
        return f"错误：未知的技能 '{skill_name}'。可用的技能有：{', '.join(available)}"

    skill = SKILLS[skill_name]

    if not skill["enabled"]:
        return f"技能 '{skill_name}' 当前已禁用。"

    print(f"   技能: {skill['name']}")
    return skill["prompt"]


@tool
def list_skills() -> str:
    """列出所有可用的技能。"""
    print("\n   列出可用技能:")
    skills_list = []
    for name, info in SKILLS.items():
        status = "✅" if info["enabled"] else "❌ (已禁用)"
        skills_list.append(f"  {status} {name}: {info['description']}")
    result = "\n".join(skills_list)
    print(result)
    return result


@tool
def enable_skill(skill_name: str) -> str:
    """启用指定技能。

    Args:
        skill_name: 要启用的技能名称
    """
    print(f"\n   启用技能: {skill_name}")

    if skill_name not in SKILLS:
        return f"错误：未知的技能 '{skill_name}'"

    SKILLS[skill_name]["enabled"] = True
    return f"技能 '{skill_name}' 已启用。"


@tool
def disable_skill(skill_name: str) -> str:
    """禁用指定技能。

    Args:
        skill_name: 要禁用的技能名称
    """
    print(f"\n    禁用技能: {skill_name}")

    if skill_name not in SKILLS:
        return f"错误：未知的技能 '{skill_name}'"

    SKILLS[skill_name]["enabled"] = False
    return f"技能 '{skill_name}' 已禁用。"


# ============================================
# 步骤 3：创建主 Agent
# ============================================

agent = create_agent(
    model=init_chat_model(
        model="deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.7,
    ),
    tools=[load_skill, list_skills, enable_skill, disable_skill],
    system_prompt="""你是一个智能助手。

            你可以通过调用 load_skill 工具来获取特定领域的专业知识。
            **重要**：收到技能提示词后，你必须立即按照该提示词的要求回答用户的问题。
            
            可用工具：
            - list_skills: 查看所有技能（包括已禁用）
            - load_skill: 加载指定技能
            - enable_skill: 启用某个技能
            - disable_skill: 禁用某个技能
            
            工作流程：
            1. 理解用户需求
            2. 调用 load_skill 获取对应技能的提示词
            3. 按照技能提示词的要求回答用户问题
            
            如果用户要求启用或禁用技能，使用相应工具。
            """,
            )


# ============================================
# 测试运行
# ============================================

def main():

    # 测试场景
    test_cases = [
        {
            "name": "场景 1: 查看可用技能",
            "query": "列出你所有的技能，包括已禁用的",
        },
        {
            "name": "场景 2: 加载 Python 技能",
            "query": "帮我写一个 Python 的装饰器示例",
        },
    ]

    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f" {test_case['name']}")
        print(f"{'='*60}")
        print(f"用户: {test_case['query']}")

        # 调用 Agent
        result = agent.invoke({
            "messages": [{"role": "user", "content": test_case["query"]}]
        })

        # 打印回复
        last_message = result["messages"][-1]
        if hasattr(last_message, 'content'):
            print(f"\n Agent 回复:")
            print(f"{'-'*60}")
            print(last_message.content[:500])
            print(f"{'-'*60}")


if __name__ == "__main__":
    main()
