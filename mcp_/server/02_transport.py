

from fastmcp import FastMCP

mcp = FastMCP("BasicDemo")


@mcp.tool()
def say_hello(name:str)->str:
    """打招呼"""
    print("Hello, world!")
    return f"Hello, {name}"


if __name__ == "__main__":
    mcp.run(
        host="0.0.0.0",
        port=8080,
        transport="streamable-http"
    )