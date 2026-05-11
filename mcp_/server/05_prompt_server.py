

from fastmcp import FastMCP

mcp = FastMCP("BasicDemo")


@mcp.prompt()
def code_review(language:str)->str:

    """代码审查提示词"""

    return f"你是一个专业的代码审查专家。请审核如何{language}代码"



@mcp.prompt()
def tool_review(toolname:str)->str:

    """工具审查提示词"""

    return f"你是一个专业的工具审查专家。请审核如何{toolname}的安全性"




if __name__ == "__main__":
    mcp.run(
        host="0.0.0.0",
        port=8080,
        transport="sse"
    )