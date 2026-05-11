
from fastmcp import FastMCP,Context


mcp = FastMCP("Math Server")


@mcp.tool()
async def confirm_operation(task:str,ctx:Context | None=None) -> str:
    """执行需要确认的操作，在执行前请求用户确认"""

    print(f"需要确认的风险操作：{task}")

    if ctx :
        result= await ctx.elicit(
            message=f"确认执行操作：{task}?",
            response_type=["是，继续","否，取消"]
        )

        print(f"用户相应：{result}")

        if hasattr(result,"data") and result.data and "是" in str(result.data):
            print("YYYYYYYY")
            return f"操作‘{task}’已经执行（用户确认）"
        else:
            print("NNNNNNNN")
            return f"操作‘{task}’已经取消（用户确认）"


    return f"操作{task}有风险,暂不执行！"


if __name__ == "__main__":


    mcp.run(
        transport="sse",
        host="0.0.0.0",
        port=8001,
    )

