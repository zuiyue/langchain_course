from typing import Callable, NotRequired, Annotated

from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ExtendedModelResponse, AgentMiddleware
from langchain.chat_models import init_chat_model
from langgraph.types import Command

from my_tool.env_utils import ZHIPU_API_KEY, ZHIPU_BASE_URL

model = init_chat_model(
    model="glm-5.1",
    model_provider="openai",
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL
)

def combin(old:str, new:str) -> str:
    return f"{old} {new}"

class CustomState(AgentState):
    last_model_call_tokens:NotRequired[int]
    user_id:NotRequired[str]
    trace_layer:NotRequired[Annotated[str, combin]]


class CustomMiddleware_001(AgentMiddleware):
    state_schema = CustomState
    def wrap_model_call(self,request: ModelRequest, handle: Callable[[ModelRequest], ModelResponse]) -> ExtendedModelResponse:
        print("001 start")
        result = handle(request)
        print("001 end")
        tokens = sum(len(str(msg.content)) for msg in request.messages)

        return ExtendedModelResponse(
            model_response=result,
            command=Command(
                update={
                    "last_model_call_tokens": tokens,
                    "trace_layer":"001"
                }
            )
        )



class CustomMiddleware_002(AgentMiddleware):
    state_schema = CustomState
    def wrap_model_call(self,request: ModelRequest, handle: Callable[[ModelRequest], ModelResponse]) -> ExtendedModelResponse:
        print("002 start")
        result = handle(request)
        print("002 end")

        return ExtendedModelResponse(
            model_response=result,
            command=Command(
                update={

                    "trace_layer": "002"
                }
            )
        )



def get_weather(city: str) -> str:
    """根据城市名称获取天气情况"""

    print("get weather........")
    return  f"{city} 天气晴朗，22°"




agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[CustomMiddleware_001(),CustomMiddleware_002()],)

if __name__ == "__main__":
    result=agent.invoke({"messages":[
        {"role":"user","content":"写一首春天的诗句"}
    ],
        "model_call_count":0,
        "user_id":"user_123"
    })


    print(result.get("last_model_call_tokens"))
    print(result.get("trace_layer"))
    print(result["messages"][-1].content)