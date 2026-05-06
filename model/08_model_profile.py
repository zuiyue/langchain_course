import time
from typing import TypedDict, Annotated

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from pip._internal.cli.spinners import RateLimiter

from pydantic import BaseModel, Field

from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL



rate_limiter=InMemoryRateLimiter(
    requests_per_second=5,
    check_every_n_seconds=0.1,
    max_bucket_size=1


)

model = init_chat_model(
    model="deepseek:deepseek-chat",
    temperature=1.5,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    rate_limiter=rate_limiter
)

HumanMessage
prompts=[
    "你好"  ,
    "再见",
    "你好",
    "你好",
    "你好",
    "你好"
]


start=time.perf_counter()

for i,p in enumerate(prompts):
    t0=time.perf_counter()
    r=model.invoke(p)
    t1=time.perf_counter()

    print(f"{p}: {t1-t0}",{r.text})

# model = init_chat_model(
#     model="gpt-5.4-mini"
# )
#
# print(model.profile)