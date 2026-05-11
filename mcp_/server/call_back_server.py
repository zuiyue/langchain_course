import asyncio

from fastmcp import FastMCP, Context

mcp = FastMCP("BasicDemo")


@mcp.tool()
async def process_data(task:str,count:int =10,ctx:Context | None=None)->str:
    """
    处理耗时任务，发送进度更新
    """
    if ctx:
        await ctx.info(f"开始任务：{task},共{count}步")
        await ctx.error(f"error*******************")
        await ctx.log(f"log*******************")
        await ctx.debug(f"debug*******************")


    total_steps=count

    for i in range(total_steps):
        await asyncio.sleep(0.5)
        progress=(i+1)/total_steps

        if ctx:
            await ctx.report_progress(progress=progress,total=1.0)

            await ctx.info(f"进度{progress:.0%} ({i+1}/{total_steps})")

        print(f"[server]:进度{progress:.0%} ({i+1}/{total_steps})")

    if ctx:
        await ctx.info(f"任务完成：{task}")

    print(f"[sverver]:任务完成：{task}")

    return f"任务{task}完成，共处理了{count}个步骤"



if __name__ == "__main__":
    mcp.run(
        host="0.0.0.0",
        port=8080,
        transport="sse"
    )