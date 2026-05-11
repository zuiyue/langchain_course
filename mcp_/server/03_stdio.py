

from fastmcp import FastMCP

mcp = FastMCP("BasicDemo")


@mcp.tool()
def say_hello(name:str)->str:
    """打招呼"""
    print("Hello, world!")
    return f"Hello, {name}"


if __name__ == "__main__":
    mcp.run(

        transport="stdio"
    )