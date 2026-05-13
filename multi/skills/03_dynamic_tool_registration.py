
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
# 步骤 1：定义技能专属工具
# ============================================

# SQL 技能专属工具
@tool
def execute_sql(sql_query: str) -> str:
    """执行 SQL 查询并返回结果。

     仅在加载 SQL 技能后可用。

    Args:
        sql_query: SQL 查询语句
    """
    print(f"\n   执行 SQL: {sql_query}")
    # 模拟执行结果（实际应该连接数据库）
    return f"SQL 执行成功！\n查询：{sql_query}\n结果：[模拟数据：100 条记录]"


@tool
def optimize_sql(sql_query: str) -> str:
    """优化 SQL 查询性能。

     仅在加载 SQL 技能后可用。

    Args:
        sql_query: 要优化的 SQL 查询
    """
    print(f"\n  🔍 优化 SQL: {sql_query}")
    return f"优化建议：\n1. 添加索引\n2. 避免 SELECT *\n3. 使用 EXPLAIN 分析"


# Python 技能专属工具
@tool
def run_python_code(code: str) -> str:
    """执行 Python 代码并返回输出。

     仅在加载 Python 技能后可用。

    Args:
        code: Python 代码字符串
    """
    print(f"\n  执行 Python 代码")
    # 模拟执行（实际应该用 subprocess 或 exec）
    return f"代码执行成功！\n输出：[模拟结果]"


# ============================================
# 步骤 2：定义技能库
# ============================================

SKILLS = {
    "sql": {
        "name": "SQL 数据库",
        "description": "SQL 查询专家",
        "prompt": """你是 SQL 数据库专家。
你可以使用 execute_sql 工具执行 SQL 查询，使用 optimize_sql 工具优化查询。

请直接帮助用户编写 SQL，并使用工具执行验证。""",
        "tools": [execute_sql, optimize_sql],  # 技能专属工具
    },
    "python": {
        "name": "Python 编程",
        "description": "Python 编程专家",
        "prompt": """你是 Python 编程专家。
你可以使用 run_python_code 工具执行 Python 代码。

请直接帮助用户编写 Python 代码，并使用工具验证。""",
        "tools": [run_python_code],
    },
}


# ============================================
# 步骤 3：创建技能加载工具
# ============================================

@tool
def load_skill(skill_name: str) -> str:
    """加载指定技能及其专属工具。

    Available skills:
    - sql: SQL 数据库专家（附带 execute_sql, optimize_sql 工具）
    - python: Python 编程专家（附带 run_python_code 工具）

    Args:
        skill_name: 技能名称
    """
    print(f"\n  🔧 加载技能: {skill_name}")

    if skill_name not in SKILLS:
        available = list(SKILLS.keys())
        return f"错误：未知的技能 '{skill_name}'。可用的技能有：{', '.join(available)}"

    skill = SKILLS[skill_name]
    tool_names = [t.name for t in skill["tools"]]
    print(f"  📝 技能: {skill['name']}")
    print(f"  🔌 专属工具: {', '.join(tool_names)}")

    # 返回提示词 + 工具列表信息
    result = f"已加载技能：{skill['name']}\n"
    result += f"可用工具：{', '.join(tool_names)}\n\n"
    result += skill["prompt"]
    return result


@tool
def list_skills() -> str:
    """列出所有可用技能及其专属工具。"""
    print("\n  📋 列出可用技能:")
    skills_list = []
    for name, info in SKILLS.items():
        tool_names = [t.name for t in info["tools"]]
        skills_list.append(f"  - {name}: {info['description']}")
        skills_list.append(f"    专属工具: {', '.join(tool_names)}")
    result = "\n".join(skills_list)
    print(result)
    return result


# ============================================
# 步骤 4：创建主 Agent（包含所有工具）
# ============================================

# 收集所有工具
all_tools = [load_skill, list_skills]
for skill in SKILLS.values():
    all_tools.extend(skill["tools"])

agent = create_agent(
    model=init_chat_model(
        model="deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.7,
    ),
    tools=all_tools,
    system_prompt="""你是一个智能助手。

        你可以通过调用 load_skill 工具来获取特定领域的专业知识。
        **重要**：收到技能提示词后，你必须立即按照该提示词的要求回答用户的问题。
        
        工作流程：
        1. 理解用户需求
        2. 调用 load_skill 获取对应技能的提示词和可用工具
        3. 使用技能专属工具执行操作
        4. 按照技能提示词的要求回答用户问题
        """,
        )


# ============================================
# 测试运行
# ============================================

def main():

    # 测试场景
    test_cases = [
        {
            "name": "场景 1: 加载 SQL 技能",
            "query": "帮我查询所有用户的姓名和邮箱",
        },
    ]

    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f" {test_case['name']}")
        print(f"{'='*60}")
        print(f" 用户: {test_case['query']}")

        # 调用 Agent
        result = agent.invoke({
            "messages": [{"role": "user", "content": test_case["query"]}]
        })

        # 打印回复
        last_message = result["messages"][-1]
        if hasattr(last_message, 'content'):
            print(f" Agent 回复:")
            print(f"{'-'*60}")
            print(last_message.content[:500])
            print(f"{'-'*60}")


if __name__ == "__main__":
    main()
