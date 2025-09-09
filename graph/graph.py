from dotenv import load_dotenv

from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import question_router, RouteQuery
from graph.node_constants import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

load_dotenv()
#WEBSEARCH ve GENERATE returnları agent'in  hangi node'a gideceğini return eder.
def decide_to_generate(state: GraphState):
    print("----DECIDE TO GENERATE----")
    if state["web_search"]:
        print("WEB SEARCH")
        return WEBSEARCH
    else:
        return GENERATE

def grade_generation_grounded_in_docs_and_question(state: GraphState) -> str:
    print("CHECK HALLUCINATION")
    generation = state["generation"]
    question = state["question"]
    documents = state["documents"]

    hallucination_score = hallucination_grader.invoke({"generation": generation,"documents": documents})
# := ifadesi hem içinin dolu olup true olduğunu ifade ediyor.
# Bu satırlarda kullanılan hallucination_grade ve answer_grade değişkenleri aslında "walrus operatörü" (:=) sayesinde oluşturulmuş geçici değişkenlerdir.
    if hallucination_grade := hallucination_score.binary_score:
        print("NOT HALLUCINATION")
        answer_score = answer_grader.invoke({"generation": generation,"question": question})
        if answer_grade := answer_score.binary_score:
            print("READY TO ANSWER")
            return "useful"
        else:
            print("Generation not useful to question")
            return "not useful"
    else:
        print("HALLUCINATION FOUND")
        return "not supported"

def route_question(state: GraphState) -> str:
    print("----ROUTE QUESTION----")
    question = state["question"]
    source = question_router.invoke({"question": question})

    if source.datasource == "vectorstore":
        return RETRIEVE
    elif source.datasource == "websearch":
        return WEBSEARCH




workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

#Akışın koşullu başlangıç noktası
workflow.set_conditional_entry_point(
    route_question,
    {WEBSEARCH: WEBSEARCH,
     RETRIEVE: RETRIEVE},
)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)

# GRADE_DOCUMENTS adımı yapılınca verilen fonksiyondaki koşula göre sonraki node adımını seçer. Bu sebeple koşullu edge ekleme fonksiyonu kullandık.
# Fonksiyonun return degeri : Gitmek istediğim Node

workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {WEBSEARCH: WEBSEARCH,
    GENERATE: GENERATE},
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_docs_and_question,
    {"not supported": GENERATE,
     "not useful": WEBSEARCH,
     "useful": END,},
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)


app=workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")