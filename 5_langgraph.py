import operator
from typing import TypedDict, Annotated, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langsmith import traceable
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

# ---------- Setup ----------
load_dotenv()

# GROQ MODEL
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# ---------- Structured schema & model ----------
class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay")
    score: int = Field(description="Score out of 10", ge=0, le=10)

structured_model = model.with_structured_output(EvaluationSchema)

# ---------- Sample essay ----------
essay2 = """India and AI Time
Now world change very fast because new tech call Artificial Intel… something (AI).
India also want become big in this AI thing...
(essay shortened for brevity)
"""

# ---------- LangGraph state ----------
class UPSCState(TypedDict, total=False):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[List[int], operator.add]
    avg_score: float


# ---------- Nodes ----------
@traceable(name="evaluate_language_fn", tags=["dimension:language"])
def evaluate_language(state: UPSCState):
    prompt = (
        "Evaluate LANGUAGE QUALITY of this essay. "
        "Give feedback and score /10:\n\n" + state["essay"]
    )
    out = structured_model.invoke(prompt)

    return {
        "language_feedback": out.feedback,
        "individual_scores": [out.score]
    }


@traceable(name="evaluate_analysis_fn", tags=["dimension:analysis"])
def evaluate_analysis(state: UPSCState):
    prompt = (
        "Evaluate DEPTH OF ANALYSIS of this essay. "
        "Give feedback and score /10:\n\n" + state["essay"]
    )
    out = structured_model.invoke(prompt)

    return {
        "analysis_feedback": out.feedback,
        "individual_scores": [out.score]
    }


@traceable(name="evaluate_thought_fn", tags=["dimension:clarity"])
def evaluate_thought(state: UPSCState):
    prompt = (
        "Evaluate CLARITY OF THOUGHT of this essay. "
        "Give feedback and score /10:\n\n" + state["essay"]
    )
    out = structured_model.invoke(prompt)

    return {
        "clarity_feedback": out.feedback,
        "individual_scores": [out.score]
    }


@traceable(name="final_evaluation_fn", tags=["aggregate"])
def final_evaluation(state: UPSCState):
    prompt = f"""
Create OVERALL FEEDBACK based on:

Language: {state.get('language_feedback','')}
Analysis: {state.get('analysis_feedback','')}
Clarity: {state.get('clarity_feedback','')}
"""
    overall = model.invoke(prompt).content

    scores = state.get("individual_scores", []) or []
    avg = (sum(scores) / len(scores)) if scores else 0.0

    return {
        "overall_feedback": overall,
        "avg_score": avg
    }


# ---------- Build Graph ----------
graph = StateGraph(UPSCState)

graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_thought", evaluate_thought)
graph.add_node("final_evaluation", final_evaluation)

# Fan-out → join
graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_thought")

graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_thought", "final_evaluation")

graph.add_edge("final_evaluation", END)

workflow = graph.compile()


# ---------- Run ----------
if __name__ == "__main__":
    result = workflow.invoke(
        {"essay": essay2},
        config={
            "run_name": "evaluate_upsc_essay_groq",
            "tags": ["essay", "langgraph", "groq"],
            "metadata": {
                "model": "llama-3.3-70b-versatile",
                "dimensions": ["language", "analysis", "clarity"],
            },
        },
    )

    print("\n=== Evaluation Results ===")
    print("Language feedback:\n", result.get("language_feedback", ""), "\n")
    print("Analysis feedback:\n", result.get("analysis_feedback", ""), "\n")
    print("Clarity feedback:\n", result.get("clarity_feedback", ""), "\n")
    print("Overall feedback:\n", result.get("overall_feedback", ""), "\n")
    print("Individual scores:", result.get("individual_scores", []))
    print("Average score:", result.get("avg_score", 0.0))
