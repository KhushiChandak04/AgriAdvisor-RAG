"""LangGraph workflow for complex and multi-crop RAG questions."""
import re
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from . import config, llm
from .multicrop import detect_all_crops, separate_query_per_crop, split_query_parts
from .prompts import SYSTEM_PROMPT, build_user_prompt

class AgentState(TypedDict, total=False):
    question: str
    query: str
    query_parts: List[str]
    history: str
    crops: List[str]
    queries: Dict[str, List[str]]
    top_k: int
    passages: List[Dict[str, Any]]
    missing_crops: List[str]
    note: str
    answer: str


def _build_graph(retriever):
    @traceable(name="agent_detect_crops", run_type="chain", project_name=config.LANGSMITH_PROJECT)
    def detect(state: AgentState) -> Dict[str, Any]:
        found = detect_all_crops(state["question"])
        crops = found or state.get("crops", [])
        return {"crops": crops}

    @traceable(name="agent_create_crop_subqueries", run_type="chain", project_name=config.LANGSMITH_PROJECT)
    def sub_query(state: AgentState) -> Dict[str, Any]:
        parts = state.get("query_parts") or [state.get("query", state["question"])]
        queries = {crop: [] for crop in state["crops"]}
        for part in parts:
            for crop, crop_query in separate_query_per_crop(part, state["crops"]).items():
                queries[crop].append(crop_query)
        return {"queries": queries}

    @traceable(name="agent_retrieve_evidence", run_type="chain", project_name=config.LANGSMITH_PROJECT)
    def retrieve(state: AgentState) -> Dict[str, Any]:
        crops = state["crops"]
        query_parts = state.get("query_parts") or [state.get("query", state["question"])]
        per_query = max(
            2,
            state["top_k"] // max(1, len(crops) * len(query_parts)) + 1,
        )
        passages = []
        for crop in crops:
            for query in state["queries"][crop]:
                for passage in retriever.search(query, k=per_query, crop=crop):
                    passages.append({**passage, "crop": crop, "_retrieval_query": query})
        return {"passages": passages}

    @traceable(name="agent_rerank_evidence", run_type="chain", project_name=config.LANGSMITH_PROJECT)
    def rerank(state: AgentState) -> Dict[str, Any]:
        query_parts = state.get("query_parts") or [state.get("query", "")]
        ranked = []
        for passage in state.get("passages", []):
            query = passage.get("_retrieval_query", state.get("query", ""))
            terms = set(re.findall(r"[a-z]{3,}", query.lower()))
            text_terms = set(re.findall(r"[a-z]{3,}", passage["text"].lower()))
            overlap = len(terms & text_terms) / max(1, len(terms))
            passage_score = float(passage.get("score", 0.0)) + 0.08 * overlap
            ranked.append((passage_score, passage))
        ranked.sort(key=lambda item: item[0], reverse=True)

        selected = []
        selected_ids = set()
        for part_index, _ in enumerate(query_parts):
            for crop in state["crops"]:
                candidate = next(
                    (
                        (score, passage) for score, passage in ranked
                        if passage.get("_retrieval_query") == state["queries"][crop][part_index]
                        and passage.get("crop") == crop
                        and id(passage) not in selected_ids
                    ),
                    None,
                )
                if candidate:
                    selected.append(candidate)
                    selected_ids.add(id(candidate[1]))
        selected.extend(item for item in ranked if id(item[1]) not in selected_ids)
        result = []
        for _, passage in selected[:state["top_k"]]:
            result.append({key: value for key, value in passage.items() if key != "_retrieval_query"})
        return {"passages": result}

    @traceable(name="agent_validate_evidence", run_type="chain", project_name=config.LANGSMITH_PROJECT)
    def validate(state: AgentState) -> Dict[str, Any]:
        allowed = set(state["crops"])
        query_parts = state.get("query_parts") or [state.get("query", state["question"])]
        passages = [
            passage for passage in state.get("passages", [])
            if passage.get("crop") in allowed
            and isinstance(passage.get("text"), str)
            and passage["text"].strip()
            and passage.get("source")
            and passage.get("page") is not None
            and isinstance(passage.get("score"), (int, float))
        ]
        best = {
            crop: max((p.get("score", 0.0) for p in passages if p["crop"] == crop), default=0.0)
            for crop in state["crops"]
        }
        passages = [
            p for p in passages
            if p["score"] >= config.MIN_SCORE and p["score"] >= 0.6 * best[p["crop"]]
        ]
        missing_crops = [crop for crop in state["crops"] if best[crop] < config.MIN_SCORE]
        note = ""
        if len(state["crops"]) > 1:
            note = (
                f"This is a comparison of: {', '.join(state['crops'])}. Write ONE bullet per aspect "
                "using only passages for that crop. Never apologise or refuse the whole comparison. "
                "If an aspect lacks evidence for a crop, do not guess or volunteer the omission."
            )
        if len(query_parts) > 1:
            if note:
                note += " "
            note += (
                "The farmer asked multiple separate questions/aspects. Address each one in its own "
                "bullet; do not answer only the final aspect. Use only evidence in the passages and "
                "do not invent details to fill gaps."
            )
        return {"passages": passages, "missing_crops": missing_crops, "note": note}

    @traceable(name="agent_generate_answer", run_type="chain", project_name=config.LANGSMITH_PROJECT)
    def answer(state: AgentState) -> Dict[str, str]:
        if not state.get("passages"):
            return {"answer": ""}
        query_parts = state.get("query_parts") or [state.get("query", state["question"])]
        prompt_question = "\n".join(f"- {part}" for part in query_parts)
        if len(query_parts) == 1:
            prompt_question = query_parts[0]
        if prompt_question != state["question"]:
            prompt_question += f"\n(Farmer's original wording: {state['question']})"
        response = llm.chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(
                prompt_question, state["passages"], state.get("history", ""), state.get("note", "")
            )},
        ])
        return {"answer": response}

    graph = StateGraph(AgentState)
    graph.add_node("detect", detect)
    graph.add_node("sub_query", sub_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("validate", validate)
    graph.add_node("answer", answer)
    graph.add_edge(START, "detect")
    graph.add_edge("detect", "sub_query")
    graph.add_edge("sub_query", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "validate")
    graph.add_edge("validate", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


@traceable(
    name="complex_query_agent",
    run_type="chain",
    project_name=config.LANGSMITH_PROJECT,
    process_inputs=lambda inputs: {key: value for key, value in inputs.items() if key != "retriever"},
)
def run_rag_agent(question: str, query: str, retriever, crops: List[str],
                  history: str, top_k: int,
                  query_parts: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run the graph without placing the retriever/model object in traced state."""
    graph = _build_graph(retriever)
    return graph.invoke({
        "question": question,
        "query": query,
        "query_parts": query_parts or split_query_parts(query),
        "history": history,
        "crops": crops,
        "top_k": top_k,
    })
