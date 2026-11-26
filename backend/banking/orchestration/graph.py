from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import classify_intent, escalate_or_finish, generate_answer, policy_check, retrieve_context, validate_grounding
from .router import after_classification, after_policy, after_retrieval
from .state import ChatState


builder = StateGraph(ChatState)
builder.add_node("policy_check", policy_check)
builder.add_node("classify_intent", classify_intent)
builder.add_node("retrieve_context", retrieve_context)
builder.add_node("generate_answer", generate_answer)
builder.add_node("validate_grounding", validate_grounding)
builder.add_node("escalate_or_finish", escalate_or_finish)

builder.add_edge(START, "policy_check")
builder.add_conditional_edges("policy_check", after_policy, {"finish": "escalate_or_finish", "classify": "classify_intent"})
builder.add_conditional_edges("classify_intent", after_classification, {"finish": "escalate_or_finish", "retrieve": "retrieve_context"})
builder.add_conditional_edges("retrieve_context", after_retrieval, {"finish": "escalate_or_finish", "generate": "generate_answer"})
builder.add_edge("generate_answer", "validate_grounding")
builder.add_edge("validate_grounding", "escalate_or_finish")
builder.add_edge("escalate_or_finish", END)

graph = builder.compile()
