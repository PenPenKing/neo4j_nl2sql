import json
import logging
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np

logger = logging.getLogger(__name__)

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from config import settings
from core.executor import Neo4jExecutor
from core.retriever import SchemaRetriever


DATA_SCHEMA = _root / "data" / "schema.json"
DATA_FEW_SHOT = _root / "data" / "few_shot.jsonl"
DATA_SCHEMA_TERMS = _root / "data" / "schema_terms.txt"
DATA_SCHEMA_VECTOR_DOCS = _root / "data" / "schema_vector_docs.jsonl"


@lru_cache(maxsize=256)
def _cached_embed_query(query: str) -> np.ndarray:
    """查询向量 LRU 缓存，避免重复计算同一问题的向量。"""
    emb = OllamaEmbeddings(
        model=settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
    )
    vec = emb.embed_query(query)
    return np.asarray(vec, dtype=np.float64)


class SchemaVectorStore:
    """
    文档向量预计算 + 查询向量缓存
    启动时一次性加载并计算所有文档向量，之后检索只查缓存
    """

    def __init__(self):
        self._schema_docs: List[Dict[str, Any]] = []
        self._schema_vectors: Optional[np.ndarray] = None
        self._few_shot_docs: List[Dict[str, Any]] = []
        self._few_shot_vectors: Optional[np.ndarray] = None
        self._initialized = False

    def _load_schema_vector_docs(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not DATA_SCHEMA_VECTOR_DOCS.exists():
            return rows
        with open(DATA_SCHEMA_VECTOR_DOCS, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def _load_few_shot_records(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not DATA_FEW_SHOT.exists():
            return rows
        with open(DATA_FEW_SHOT, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def initialize(self) -> None:
        """启动时调用，一次性预计算所有文档向量。"""
        if self._initialized:
            return

        logger.info("SchemaVectorStore 开始初始化...")

        # 加载文档
        self._schema_docs = self._load_schema_vector_docs()
        self._few_shot_docs = self._load_few_shot_records()

        emb = OllamaEmbeddings(
            model=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
        )

        # schema docs 向量预计算
        if self._schema_docs:
            schema_texts = [str(d.get("text") or "") for d in self._schema_docs]
            vecs = emb.embed_documents(schema_texts)
            self._schema_vectors = np.asarray(vecs, dtype=np.float64)
            logger.info(f"schema 向量预计算完成: {self._schema_vectors.shape}")

        # few-shot 向量预计算
        if self._few_shot_docs:
            few_texts = [
                str(r.get("question") or r.get("text") or r.get("q") or "")
                for r in self._few_shot_docs
            ]
            vecs = emb.embed_documents(few_texts)
            self._few_shot_vectors = np.asarray(vecs, dtype=np.float64)
            logger.info(f"few-shot 向量预计算完成: {self._few_shot_vectors.shape}")

        self._initialized = True
        logger.info("SchemaVectorStore 初始化完成")

    def _cosine_topk(
        self,
        query_vec: np.ndarray,
        doc_vectors: np.ndarray,
        records: List[Dict[str, Any]],
        k: int,
    ) -> List[Dict[str, Any]]:
        """给定查询向量和已预计算的文档向量矩阵，返回 top-k 结果。"""
        if doc_vectors is None or len(records) == 0:
            return []
        if len(records) <= k:
            return [{**r, "_similarity": 1.0} for r in records[:k]]

        norms_d = np.linalg.norm(doc_vectors, axis=1)
        norm_q = np.linalg.norm(query_vec)
        sims = (doc_vectors @ query_vec) / (norms_d * norm_q + 1e-9)
        top_indices = np.argsort(-sims)[:k]

        return [
            {**records[int(i)], "_similarity": float(sims[int(i)])}
            for i in top_indices
        ]

    def search_schema(
        self, query: str, k: int = 3
    ) -> List[Dict[str, Any]]:
        """向量检索 schema docs，返回 top-k + similarity。"""
        if not self._initialized:
            self.initialize()
        qv = _cached_embed_query(query)
        return self._cosine_topk(qv, self._schema_vectors, self._schema_docs, k)

    def search_few_shot(
        self, query: str, k: int = 2
    ) -> List[Dict[str, Any]]:
        """向量检索 few-shot，返回 top-k + similarity。"""
        if not self._initialized:
            self.initialize()
        qv = _cached_embed_query(query)
        return self._cosine_topk(qv, self._few_shot_vectors, self._few_shot_docs, k)


# 全局单例，agent 复用
_vector_store: Optional[SchemaVectorStore] = None


def get_vector_store() -> SchemaVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = SchemaVectorStore()
    return _vector_store


def _load_schema_text() -> str:
    if not DATA_SCHEMA.exists():
        return "（未找到 schema.json，请先运行 preprocess 导出。）"
    try:
        with open(DATA_SCHEMA, encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, ensure_ascii=False, indent=2)[:120000]
    except Exception as e:
        return f"（读取 schema.json 失败: {e}）"




def _schema_hit_keyword_line(doc: Dict[str, Any]) -> str:
    """从 schema 向量文档条目中提取用于展示的「关键词」一行。"""
    kind = doc.get("kind")
    if kind == "node":
        return f"节点:{doc.get('label', '')}"
    if kind == "relationship":
        return f"关系:{doc.get('type', '')}"
    return str(doc.get("id", ""))


def _select_few_shots_by_embedding(
    query: str, k: int = 2
) -> List[Dict[str, Any]]:
    """通过向量缓存检索 few-shot 示例。"""
    return get_vector_store().search_few_shot(query, k)


def _select_schema_docs_by_embedding(
    query: str, k: int = 3
) -> List[Dict[str, Any]]:
    """通过向量缓存检索 schema docs。"""
    return get_vector_store().search_schema(query, k)


def _format_few_shots(examples: List[Dict[str, Any]]) -> str:
    if not examples:
        return "（暂无 few-shot 示例，请在 data/few_shot.jsonl 中添加。）"
    parts = []
    for i, ex in enumerate(examples, 1):
        q = ex.get("question") or ex.get("text") or ""
        cy = ex.get("cypher") or ex.get("cql") or ""
        sim = ex.get("_similarity")
        sim_s = f" 相似度={sim:.4f}" if sim is not None else ""
        parts.append(f"示例{i}{sim_s} 用户问题：{q}\n示例{i} Cypher：\n{cy}\n")
    return "\n".join(parts)


def extract_cypher(llm_text: str) -> str:
    m = re.search(r"```(?:cypher)?\s*([\s\S]*?)```", llm_text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return llm_text.strip()


class GraphState(TypedDict, total=False):
    user_query: str
    schema_context: str
    matched_keywords: List[str]
    vector_schema_keywords: List[str]
    vector_schema_hits: List[Dict[str, Any]]
    few_shot_examples: List[Dict[str, Any]]
    messages: List[BaseMessage]
    last_cypher: str
    last_llm_text: str
    execution: Optional[Dict[str, Any]]
    failure_count: int


def _build_system_prompt(state: GraphState) -> str:
    mks = state.get("matched_keywords") or []
    kw = "、".join(mks) if mks else "（无）"
    vsk = state.get("vector_schema_keywords") or []
    vsk_line = "、".join(vsk) if vsk else "（无）"
    fs = _format_few_shots(state.get("few_shot_examples") or [])
    schema = state.get("schema_context") or ""
    return f"""你是 Neo4j Cypher 专家。根据用户自然语言问题，只输出一条可执行的 Cypher 查询。
要求：
1. 仅使用 schema 中出现的节点标签、关系类型与属性名；字符串字面量与图谱中已有取值保持一致（含中文）。
2. 不要编造标签或关系；需要时用 CONTAINS、= 等匹配属性。
3. 结果用 ```cypher 代码块包裹，除此之外不要输出多余说明（除非用户要求解释）。
4. 优先使用 MATCH、WHERE、RETURN，注意 LIMIT 避免过大结果集。

【图谱 Schema（JSON）】
{schema}

【用户问题中精准匹配到的词典词（AC）】
{kw}

【向量检索与问题语义相近的图谱类型/关系（节选）】
{vsk_line}

【相似问题与参考 Cypher（few-shot，向量检索）】
{fs}
"""


def node_retrieve(state: GraphState) -> GraphState:
    q = state.get("user_query") or ""
    retriever = SchemaRetriever(str(DATA_SCHEMA_TERMS))
    keywords = retriever.get_matched_schema(q)
    schema_hits = _select_schema_docs_by_embedding(q, k=3)
    vector_schema_keywords = [_schema_hit_keyword_line(h) for h in schema_hits]
    few = _select_few_shots_by_embedding(q, k=2)
    schema_text = _load_schema_text()
    sys_content = _build_system_prompt(
        {
            "user_query": q,
            "schema_context": schema_text,
            "matched_keywords": keywords,
            "vector_schema_keywords": vector_schema_keywords,
            "few_shot_examples": few,
        }
    )
    messages: List[BaseMessage] = [
        SystemMessage(content=sys_content),
        HumanMessage(content=q),
    ]
    return {
        "schema_context": schema_text,
        "matched_keywords": keywords,
        "vector_schema_keywords": vector_schema_keywords,
        "vector_schema_hits": schema_hits,
        "few_shot_examples": few,
        "messages": messages,
        "execution": None,
        "failure_count": 0,
        "last_cypher": "",
        "last_llm_text": "",
    }


def node_generate(state: GraphState) -> GraphState:
    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )
    messages = list(state.get("messages") or [])
    ex = state.get("execution")
    if ex and ex.get("status") == "error":
        err = ex.get("error") or ""
        tb = (ex.get("full_traceback") or "")[:2500]
        messages.append(
            HumanMessage(
                content=(
                    "上一次生成的 Cypher 执行失败，请根据错误修正，仍只输出一条 ```cypher 代码块。\n"
                    f"错误摘要：{err}\n\n堆栈节选：\n{tb}"
                )
            )
        )
    resp = llm.invoke(messages)
    text = resp.content if hasattr(resp, "content") else str(resp)
    cypher = extract_cypher(text)
    new_messages = messages + [AIMessage(content=text)]
    return {
        "messages": new_messages,
        "last_cypher": cypher,
        "last_llm_text": text,
    }


def node_execute(state: GraphState) -> GraphState:
    executor = Neo4jExecutor()
    try:
        res = executor.run_query(state.get("last_cypher") or "")
    finally:
        executor.close()
    fc = state.get("failure_count", 0)
    if res.get("status") != "success":
        fc = fc + 1
    return {
        "execution": res,
        "failure_count": fc,
    }


def route_after_execute(state: GraphState) -> str:
    ex = state.get("execution") or {}
    if ex.get("status") == "success":
        return "done"
    if state.get("failure_count", 0) >= settings.max_retries:
        return "done"
    return "retry"


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("retrieve", node_retrieve)
    g.add_node("generate", node_generate)
    g.add_node("execute", node_execute)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "execute")
    g.add_conditional_edges(
        "execute",
        route_after_execute,
        {"retry": "generate", "done": END},
    )
    return g.compile()


def _few_shots_for_output(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """终端/API 用的 few-shot 结构（不含大段 text）。"""
    rows: List[Dict[str, Any]] = []
    for ex in examples:
        rows.append(
            {
                "question": ex.get("question") or ex.get("text") or ex.get("q") or "",
                "cypher": ex.get("cypher") or ex.get("cql") or "",
                "similarity": ex.get("_similarity"),
            }
        )
    return rows


class CypherAgent:
    """LangGraph 驱动的 Neo4j 问答：检索上下文 → 生成 Cypher → 执行 → 失败则反思重试。"""

    def __init__(self):
        self.graph = build_graph()
        # 预热向量存储（启动时一次性计算所有文档向量）
        get_vector_store().initialize()

    def run(self, user_query: str) -> Dict[str, Any]:
        out: GraphState = self.graph.invoke({"user_query": user_query})
        ex = out.get("execution") or {}
        exact_kw = out.get("matched_keywords") or []
        vec_kw = out.get("vector_schema_keywords") or []
        fs_raw = out.get("few_shot_examples") or []
        return {
            "user_query": user_query,
            "exact_match_keywords": exact_kw,
            "vector_schema_keywords": vec_kw,
            "vector_schema_hits": out.get("vector_schema_hits") or [],
            "few_shot_retrieved": _few_shots_for_output(fs_raw),
            "generated_cypher": out.get("last_cypher") or "",
            "matched_keywords": exact_kw,
            "few_shot_examples": fs_raw,
            "last_cypher": out.get("last_cypher") or "",
            "last_llm_text": out.get("last_llm_text") or "",
            "execution": ex,
            "failure_count": out.get("failure_count", 0),
        }


def run_agent(user_query: str) -> Dict[str, Any]:
    return CypherAgent().run(user_query)


def print_agent_report(r: Dict[str, Any]) -> None:
    """格式化打印一次 Agent 运行的四类检索信息。"""
    print("1. 精准匹配关键词（AC / schema_terms）:")
    print("  ", r.get("exact_match_keywords") or r.get("matched_keywords") or [])
    print("2. 向量检索关键词（图谱 schema_vector_docs，节点/关系名）:")
    print("  ", r.get("vector_schema_keywords") or [])
    print("3. 向量检索得到的示范语句（few_shot.jsonl，含相似度）:")
    for i, ex in enumerate(r.get("few_shot_retrieved") or [], 1):
        sim = ex.get("similarity")
        sim_s = f" similarity={sim:.4f}" if sim is not None else ""
        print(f"  --- 示例 {i}{sim_s} ---")
        print("  问:", (ex.get("question") or "").strip())
        print("  Cypher:\n", (ex.get("cypher") or "").strip())
    if not (r.get("few_shot_retrieved") or []):
        print("  （无，请补充 data/few_shot.jsonl）")
    print("4. Agent 执行生成的检索语句（Cypher）:")
    print("  ", (r.get("generated_cypher") or r.get("last_cypher") or "").strip())
    ex = r.get("execution") or {}
    print("执行状态:", ex.get("status"))
    print("结果条数:", len(ex.get("data") or []))


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]).strip() or "科学载荷管理器工作电流异常相关故障有哪些？"
    r = run_agent(q)
    print_agent_report(r)
