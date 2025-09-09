from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class GradeAnswer(BaseModel):
    binary_score: str = Field(
        description="Answer is relevant to the question, 'yes' or 'no'"
    )

llm = ChatOpenAI(temperature=0)

structured_llm_grader = llm.with_structured_output(GradeAnswer)

system_prompt = """You are a grader assessing whether an answer addresses / resolves a question. \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "User question: \n\n  {question} \n\n LLM generation: {generation}")
    ]
)

answer_grader = answer_prompt | structured_llm_grader