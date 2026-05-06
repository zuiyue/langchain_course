from typing import TypedDict, Annotated

from langchain.chat_models import init_chat_model

from pydantic import BaseModel, Field

from my_tool.env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
model = init_chat_model(
    model="deepseek:deepseek-chat",
    temperature=1.5,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)


class Movie(BaseModel):
    title: str=Field(description="Movie title")
    year: int=Field(description="Movie year")
    director: str=Field(description="Movie director")
    rating: float=Field(description="Movie rating")



class MovieDict(TypedDict):
    title: Annotated[str, "Movie title"]
    year: Annotated[int, "Movie year"]
    director: Annotated[str, "Movie director"]
    rating: Annotated[float, "Movie rating"]

json_schema={
    "title": "Movie",
    "description": "A movie with details",
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Movie title"},
        "year": {"type": "integer", "description": "Release year"},
        "director": {"type": "string", "description": "Director name"},
        "rating": {"type": "number", "description": "Rating out of 10"},
    },
    "required": ["title", "year", "director", "rating"],
}



# json_schema={
#     "title": "Movie",
#     "description": "A movie with details",
#     "type": "object",
#     "properties": {
#         "title": {"type": "string", "description": "Movie title"},
#         "year": {"type": "integer", "description": "Release year"},
#         "director": {"type": "string", "description": "Director name"},
#         "rating": {"type": "float", "description": "Rating out of 10"},
#     },
#     "required": ["title", "year", "director", "rating"],
# }


# result=model.invoke("提供电影阿凡达的详细信息")
# print(result)
#
# pydantic_model=model.with_structured_output(Movie)
#
# result=pydantic_model.invoke("提供电影阿凡达的详细信息")
# print(result)


# result=model.invoke("提供电影阿凡达的详细信息")
# print(result)
# print("*"*50)
# pydantic_model=model.with_structured_output(MovieDict)
#
# result=pydantic_model.invoke("提供电影阿凡达的详细信息")
# print(result)


# result=model.invoke("提供电影阿凡达的详细信息")
# print(result)
# print("*"*50)
# json_schema_model=model.with_structured_output(json_schema,method="json_schema")
# result=json_schema_model.invoke("提供电影阿凡达的详细信息")
# print(result)



pydantic_model=model.with_structured_output(Movie,include_raw=True)
result=pydantic_model.invoke("提供电影阿凡达的详细信息")
print(result)


pydantic_model=model.with_structured_output(Movie,include_raw=False)
result=pydantic_model.invoke("提供电影阿凡达的详细信息")
print(result)