"""
LangGraph nodes for RAG workflow using a Groq-safe ReAct-style agent
"""

from typing import List, Optional

from src.state.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent


class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None  # lazy-init agent

    # -------------------------
    # Graph node: retrieve docs
    # -------------------------
    def retrieve_docs(self, state: RAGState) -> RAGState:
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs,
            answer=state.answer,
        )

    # -------------------------
    # Build Groq-safe tools
    # -------------------------
    def _build_tools(self) -> List[Tool]:
        """Build retriever tool only (Groq-compatible)"""

        def retriever_tool_fn(query: str) -> str:
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No relevant documents found."

            merged = []
            for i, d in enumerate(docs[:6], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")

            return "\n\n".join(merged)

        retriever_tool = Tool(
            name="retriever",
            description=(
                "Use this tool to search the indexed documents. "
                "Input should be a natural language query. "
                "Output will be relevant document passages."
            ),
            func=retriever_tool_fn,
        )

        return [retriever_tool]

    # -------------------------
    # Build agent
    # -------------------------
    def _build_agent(self):
        tools = self._build_tools()

        system_prompt = (
            "You are a helpful RAG assistant.\n"
            "Use the 'retriever' tool to answer questions from the indexed documents.\n"
            "If the answer is not present in the documents, say so clearly.\n"
            "Do NOT hallucinate.\n"
            "Return ONLY the final answer."
        )

        self._agent = create_react_agent(
            self.llm,
            tools=tools,
            prompt=system_prompt,
        )

    # -------------------------
    # Graph node: generate answer
    # -------------------------
    def generate_answer(self, state: RAGState) -> RAGState:
        if self._agent is None:
            self._build_agent()

        result = self._agent.invoke(
            {"messages": [HumanMessage(content=state.question)]}
        )

        messages = result.get("messages", [])
        answer: Optional[str] = None

        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg, "content", None)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "I could not find an answer in the provided documents.",
        )
