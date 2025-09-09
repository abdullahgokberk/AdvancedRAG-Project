from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: bool = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

llm = ChatOpenAI(temperature=0)

structured_llm_grader = llm.with_structured_output(GradeDocuments, method="function_calling")

system_prompt = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n 
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Retrieved document: {document} User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

"""
Deneme yapmak istersek.
from dotenv import load_dotenv
from ingestion import retriever

load_dotenv()


if __name__ == "__main__":
    user_question = "What is prompt engineering?"
    #get_relevant_documents metodu ile verilen question'a uygun documentleri getiriyoruz.
    docs = retriever.get_relevant_documents(user_question)
    retrieved_document = docs[0].page_content
    print(retrieval_grader.invoke({
        "document": retrieved_document,
        "question": user_question
    }))
"""