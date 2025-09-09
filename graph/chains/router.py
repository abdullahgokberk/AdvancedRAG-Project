# Literal sayesinde bu ya bir vectorstore elemanı olacak ya da websearch yapılacak diyebiliyoruz.Herhangibirini kabul etmesi için ... koyduk.
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
# pydantic kullanım amacı bir değişkenin cinsini zorunlu kılmak. Pythonda bu durum normalde sorun yaratmıyordu bu sebeple.
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class RouteQuery(BaseModel):
    """
    Route a user query to the most relevant datasource
    """

    datasource : Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to a vectorstore or websearch",
    )

llm = ChatOpenAI(temperature=0)

#LLM'in çıktısını yapısal şekilde ele alan bu fonksiyona sınıfımızı veriyoruz.
structured_llm_router = llm.with_structured_output(RouteQuery, method="function_calling")

system_prompt = """You are an expert at routing a user question to a vectorstore or web search. \n 
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks. \n 
Use the vectorstore for questions on these topics. For all else, use web-search."""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}")
    ]
)

question_router = route_prompt | structured_llm_router

