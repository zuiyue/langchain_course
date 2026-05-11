
from fastmcp import FastMCP

mcp = FastMCP("Math Server")


@mcp.tool()
def add(a: float, b: float) -> float:
    """将两个数字相加"""
    result = a + b
    print(f"  [Math Server] 调用 add({a}, {b}) = {result}")
    return result


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """将两个数字相乘"""
    result = a * b
    print(f"  [Math Server] 调用 multiply({a}, {b}) = {result}")
    return result

@mcp.tool()
def get_weather(city:str)->str:
    """根据城市查看天气状况"""
    print(f"{city}")
    return f"{city}天气晴朗，22°"


if __name__ == "__main__":


    mcp.run(
        transport="sse",
        host="0.0.0.0",
        port=8001,
    )

