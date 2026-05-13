

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
# 步骤 1：定义技能库（提示词模块）
# ============================================

# 技能定义：每个技能包含名称、描述和专用提示词
SKILLS = {
                "python_programming": {
                    "name": "Python 编程",
                    "description": "Python 语言专家，擅长编写 Python 代码、调试和优化",
                    "prompt": """你现在是 Python 编程专家。
                                你的职责是帮助用户解决 Python 相关问题。
                                
                                要求：
                                1. 提供可运行的代码示例
                                2. 解释代码的关键点
                                3. 指出最佳实践和常见陷阱
                                4. 如果涉及性能优化，给出具体的优化建议
                                
                                请直接回答用户的问题，提供详细的代码示例。""",
                },
                "sql_database": {
                    "name": "SQL 数据库",
                    "description": "SQL 查询专家，擅长编写复杂 SQL 查询和数据库设计",
                    "prompt": """你现在是 SQL 数据库专家。
                                你的职责是帮助用户编写 SQL 查询和优化数据库设计。
                                
                                要求：
                                1. 提供标准的 SQL 查询语句
                                2. 解释查询的逻辑和性能考虑
                                3. 给出数据库设计建议
                                4. 如果涉及多表查询，说明 JOIN 策略
                                
                                请直接回答用户的问题，提供具体的 SQL 示例。""",
                },
                "technical_writing": {
                    "name": "技术写作",
                    "description": "技术文档写作专家，擅长撰写 API 文档、用户手册和技术博客",
                    "prompt": """你现在是技术写作专家。
                                你的职责是帮助用户撰写高质量的技术文档。
                                
                                要求：
                                1. 结构清晰，层次分明
                                2. 使用简洁明了的语言
                                3. 包含必要的代码示例和截图说明
                                4. 遵循技术文档的最佳实践
                                
                                请直接回答用户的需求，提供具体的文档内容。""",
                },
}


# ============================================
# 步骤 2：创建技能加载工具
# ============================================

@tool
def load_skill(skill_name: str) -> str:
    """加载指定技能的专用提示词。

    当你需要某个特定领域的专业知识时调用此工具。

    Available skills:
    - python_programming: Python 编程专家
    - sql_database: SQL 数据库专家
    - technical_writing: 技术文档写作专家

    Args:
        skill_name: 技能名称（如 "python_programming"）
    """
    print(f"\n  🔧 加载技能: {skill_name}")

    if skill_name not in SKILLS:
        available = list(SKILLS.keys())
        return f"错误：未知的技能 '{skill_name}'。可用的技能有：{', '.join(available)}"

    skill = SKILLS[skill_name]
    print(f"  📝 技能: {skill['name']}")
    print(f"  💡 描述: {skill['description']}")

    # 返回技能的专用提示词
    return skill["prompt"]


@tool
def list_skills() -> str:
    """列出所有可用的技能。

    当你需要了解系统提供哪些专业技能时调用此工具。
    """
    print("\n  📋 列出可用技能:")
    skills_list = []
    for name, info in SKILLS.items():
        skills_list.append(f"  - {name}: {info['description']}")
    result = "\n".join(skills_list)
    print(result)
    return result


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
    tools=[load_skill, list_skills],
    system_prompt="""你是一个智能助手。
            
            你可以通过调用 load_skill 工具来获取特定领域的专业知识。
            调用 load_skill 后，你会收到该技能的专用提示词。
            **重要**：收到技能提示词后，你必须立即按照该提示词的要求回答用户的问题。
            
            可用的技能：
            - python_programming: Python 编程
            - sql_database: SQL 数据库
            - technical_writing: 技术写作
            
            如果不确定用户需要什么技能，可以先调用 list_skills 查看所有可用技能。
            
            工作流程：
            1. 理解用户需求
            2. 调用 load_skill 获取对应技能的提示词
            3. 按照技能提示词的要求回答用户问题
            """,
            )


# ============================================
# 测试运行
# ============================================

def main():


    # 测试场景
    test_queries = [
        {
            "name": "场景 1: 查看可用技能",
            "query": "你有哪些专业技能？",
        },
        {
            "name": "场景 2: 调用 Python 技能",
            "query": "帮我用 Python 写一个快速排序算法",
        },
        {
            "name": "场景 3: 调用 SQL 技能",
            "query": "如何查询每个部门薪资最高的员工？",
        },
    ]

    for test_case in test_queries:
        print(f"\n{'='*60}")
        print(f"{test_case['name']}")
        print(f"{'='*60}")
        print(f" 用户: {test_case['query']}")

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
