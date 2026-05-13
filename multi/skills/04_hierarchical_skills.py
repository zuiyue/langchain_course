

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
# 步骤 1：定义分层技能库（树状结构）
# ============================================

SKILLS = {
    # 根节点：数据科学
    "data_science": {
        "name": "数据科学",
        "description": "数据科学领域（包含多个子技能）",
        "children": ["pandas", "matplotlib", "scikit_learn"],
    },
    # 子节点：Pandas
    "pandas": {
        "name": "Pandas 数据处理",
        "description": "Pandas 数据处理和清洗专家",
        "parent": "data_science",
        "prompt": """你是 Pandas 数据处理专家。
擅长 DataFrame 操作、数据清洗、合并、分组聚合等。

请直接帮助用户使用 Pandas 解决问题，提供代码示例。""",
    },
    # 子节点：Matplotlib
    "matplotlib": {
        "name": "Matplotlib 可视化",
        "description": "Matplotlib 数据可视化专家",
        "parent": "data_science",
        "prompt": """你是 Matplotlib 数据可视化专家。
擅长绘制折线图、柱状图、散点图、热力图等。

请直接帮助用户创建图表，提供完整的代码示例。""",
    },
    # 子节点：Scikit-learn
    "scikit_learn": {
        "name": "Scikit-learn 机器学习",
        "description": "Scikit-learn 机器学习专家",
        "parent": "data_science",
        "prompt": """你是 Scikit-learn 机器学习专家。
擅长分类、回归、聚类、特征工程等。

请直接帮助用户构建模型，提供完整的训练流程。""",
    },
    # 另一个根节点：编程
    "programming": {
        "name": "编程",
        "description": "编程领域（包含多个子技能）",
        "children": ["python", "javascript", "go"],
    },
    "python": {
        "name": "Python",
        "description": "Python 编程专家",
        "parent": "programming",
        "prompt": """你是 Python 编程专家。
请帮助用户解决 Python 相关问题，提供代码示例。""",
    },
    "javascript": {
        "name": "JavaScript",
        "description": "JavaScript 编程专家",
        "parent": "programming",
        "prompt": """你是 JavaScript 编程专家。
请帮助用户解决 JavaScript 相关问题，提供代码示例。""",
    },
    "go": {
        "name": "Go",
        "description": "Go 编程专家",
        "parent": "programming",
        "prompt": """你是 Go 编程专家。
请帮助用户解决 Go 相关问题，提供代码示例。""",
    },
}


# ============================================
# 步骤 2：创建分层技能加载工具
# ============================================

@tool
def list_skills(category: str = None) -> str:
    """列出所有可用技能，可按类别筛选。

    Args:
        category: 可选，技能类别（如 "data_science" 或 "programming"）
    """
    print(f"\n   列出可用技能 (类别: {category or '全部'}):")

    if category and category in SKILLS:
        root = SKILLS[category]
        if "children" in root:
            print(f"  {root['name']} (根节点)")
            for child in root["children"]:
                child_skill = SKILLS[child]
                print(f"    └─ {child}: {child_skill['description']}")
            return f"类别 '{category}' 包含 {len(root['children'])} 个子技能"
        else:
            return f"  {category}: {root['description']}"
    else:
        # 列出所有根节点
        roots = [name for name, skill in SKILLS.items() if "parent" not in skill]
        skills_list = []
        for root in roots:
            info = SKILLS[root]
            if "children" in info:
                skills_list.append(f"   {root}: {info['description']}")
                skills_list.append(f"     子技能: {', '.join(info['children'])}")
            else:
                skills_list.append(f"   {root}: {info['description']}")
        result = "\n".join(skills_list)
        print(result)
        return result


@tool
def load_skill(skill_name: str) -> str:
    """加载指定子技能。

    Available sub-skills:
    - pandas: Pandas 数据处理
    - matplotlib: Matplotlib 可视化
    - scikit_learn: Scikit-learn 机器学习
    - python: Python 编程
    - javascript: JavaScript 编程
    - go: Go 编程

    Args:
        skill_name: 子技能名称
    """
    print(f"\n  🔧 加载子技能: {skill_name}")

    if skill_name not in SKILLS:
        return f"错误：未知的技能 '{skill_name}'"

    skill = SKILLS[skill_name]

    if "children" in skill:
        return f"'{skill_name}' 是一个根节点类别，包含子技能：{', '.join(skill['children'])}。请使用 load_skill 加载具体子技能。"

    parent_info = ""
    if "parent" in skill:
        parent_info = f"（所属类别：{skill['parent']}）"

    print(f"   技能: {skill['name']} {parent_info}")
    return skill["prompt"]


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
    tools=[list_skills, load_skill],
    system_prompt="""你是一个智能助手。
        
        你可以通过以下方式获取技能：
        1. 调用 list_skills 查看可用技能类别
        2. 调用 list_skills(category="xxx") 查看某类别下的子技能
        3. 调用 load_skill 加载具体子技能
        
        **重要**：收到技能提示词后，你必须立即按照该提示词的要求回答用户的问题。
        
        可用根类别：
        - data_science: 数据科学（子技能：pandas, matplotlib, scikit_learn）
        - programming: 编程（子技能：python, javascript, go）
        
        工作流程：
        1. 理解用户需求
        2. 按需加载对应的子技能
        3. 按照技能提示词回答用户问题
        """,
)


# ============================================
# 测试运行
# ============================================

def main():

    # 测试场景
    test_cases = [
        {
            "name": "场景 1: 查看数据科学子技能",
            "query": "数据科学类别下有哪些子技能？",
        },
        {
            "name": "场景 2: 加载 Pandas 子技能",
            "query": "帮我用 Pandas 读取 CSV 并做数据清洗",
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
            print(f"\n Agent 回复:")
            print(f"{'-'*60}")
            print(last_message.content[:500])
            print(f"{'-'*60}")



if __name__ == "__main__":
    main()
