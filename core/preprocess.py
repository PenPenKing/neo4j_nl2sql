import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from neo4j import GraphDatabase

from config import settings


def _cypher_ident(label: str) -> str:
    return "`" + label.replace("`", "``") + "`"


def _fetch_labels(session):
    rows = session.run(
        "CALL db.labels() YIELD label RETURN label ORDER BY label"
    )
    return [r["label"] for r in rows]


def _fetch_relationship_types(session):
    rows = session.run(
        "CALL db.relationshipTypes() YIELD relationshipType "
        "RETURN relationshipType ORDER BY relationshipType"
    )
    return [r["relationshipType"] for r in rows]


def _node_props_from_schema_procedures(session):
    """Neo4j 4.4+：db.schema.nodeTypeProperties"""
    rows = session.run(
        """
        CALL db.schema.nodeTypeProperties()
        YIELD nodeLabels, propertyName, propertyTypes, mandatory
        RETURN nodeLabels, propertyName, propertyTypes, mandatory
        """
    )
    by_label: dict[str, set[str]] = {}
    for r in rows:
        labels = r["nodeLabels"] or []
        pname = r["propertyName"]
        if not pname:
            continue
        for lbl in labels:
            by_label.setdefault(lbl, set()).add(pname)
    return {k: sorted(v) for k, v in by_label.items()}


def _rel_props_from_schema_procedures(session):
    """Neo4j 4.4+：db.schema.relTypeProperties"""
    rows = session.run(
        """
        CALL db.schema.relTypeProperties()
        YIELD relType, propertyName, propertyTypes, mandatory
        RETURN relType, propertyName, propertyTypes, mandatory
        """
    )
    by_type: dict[str, set[str]] = {}
    for r in rows:
        rt = r["relType"]
        pname = r["propertyName"]
        if not rt or not pname:
            continue
        by_type.setdefault(rt, set()).add(pname)
    return {k: sorted(v) for k, v in by_type.items()}


def _sample_node_properties(session, label: str) -> List[str]:
    q = (
        f"MATCH (n:{_cypher_ident(label)}) "
        "WITH n LIMIT 1000 UNWIND keys(n) AS k RETURN DISTINCT k ORDER BY k"
    )
    return [r["k"] for r in session.run(q)]


def _sample_rel_properties(session, rel_type: str) -> List[str]:
    q = (
        f"MATCH ()-[r:{_cypher_ident(rel_type)}]->() "
        "WITH r LIMIT 1000 UNWIND keys(r) AS k RETURN DISTINCT k ORDER BY k"
    )
    return [r["k"] for r in session.run(q)]


def _count_nodes_by_label(session, label: str) -> int:
    q = f"MATCH (n:{_cypher_ident(label)}) RETURN count(n) AS c"
    rec = session.run(q).single()
    return int(rec["c"]) if rec else 0


def _count_rels_by_type(session, rel_type: str) -> int:
    q = f"MATCH ()-[r:{_cypher_ident(rel_type)}]->() RETURN count(r) AS c"
    rec = session.run(q).single()
    return int(rec["c"]) if rec else 0


def _count_all_nodes(session) -> int:
    rec = session.run("MATCH (n) RETURN count(n) AS c").single()
    return int(rec["c"]) if rec else 0


def _count_all_relationships(session) -> int:
    rec = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()
    return int(rec["c"]) if rec else 0


def _serialize_sample_value(v: Any) -> Any:
    """将 Neo4j 返回值转为可 JSON 序列化的形式，保留中文等 Unicode 字符串。"""
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    return str(v)


def _distinct_values_for_node_prop(
    session, label: str, prop: str, limit: int
) -> List[Any]:
    """按标签 + 属性名拉取去重后的属性值（含中文、英文等），用于 schema 展示。"""
    rows = session.run(
        """
        MATCH (n)
        WHERE $lbl IN labels(n) AND $prop IN keys(n)
        WITH n[$prop] AS v
        WHERE v IS NOT NULL
        RETURN DISTINCT v AS v
        LIMIT $lim
        """,
        lbl=label,
        prop=prop,
        lim=int(limit),
    )
    out: List[Any] = []
    seen: set = set()
    for r in rows:
        raw = r["v"]
        serialized = _serialize_sample_value(raw)
        key = repr(serialized)
        if key in seen:
            continue
        seen.add(key)
        out.append(serialized)
    return out


def _collect_schema_terms(payload: Dict[str, Any]) -> List[str]:
    """从导出 payload 收集用于精准匹配的词表（去重、排序）。"""
    terms: set[str] = set()
    for n in payload.get("nodes") or []:
        lbl = (n.get("label") or "").strip()
        if lbl:
            terms.add(lbl)
        for p in n.get("properties") or []:
            p = (p or "").strip()
            if p:
                terms.add(p)
        for prop, vals in (n.get("property_value_samples") or {}).items():
            pk = (prop or "").strip()
            if pk:
                terms.add(pk)
            for v in vals or []:
                s = str(v).strip()
                if s:
                    terms.add(s)
    for r in payload.get("relationships") or []:
        rt = (r.get("type") or "").strip()
        if rt:
            terms.add(rt)
        for p in r.get("properties") or []:
            p = (p or "").strip()
            if p:
                terms.add(p)
        for prop, vals in (r.get("property_value_samples") or {}).items():
            pk = (prop or "").strip()
            if pk:
                terms.add(pk)
            for v in vals or []:
                s = str(v).strip()
                if s:
                    terms.add(s)
    return sorted(terms)


def _write_schema_terms(terms: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(terms) + ("\n" if terms else ""), encoding="utf-8")


def _build_node_vector_text(node: Dict[str, Any]) -> str:
    """单节点类型：供向量嵌入的自然语言描述（中英混排保留）。"""
    lbl = node.get("label") or ""
    ref = _cypher_ident(lbl)
    lines: List[str] = [
        f"【图数据库节点类型】标签为 {ref}（Cypher 中写作 :{ref}）。",
        f"该标签下约有 {node.get('count', 0)} 个节点实体。",
    ]
    props = node.get("properties") or []
    if props:
        lines.append("属性键包括：" + "、".join(props) + "。")
    samples = node.get("property_value_samples") or {}
    for pk, vals in samples.items():
        if not vals:
            continue
        parts = [str(v) for v in vals]
        lines.append(f"属性「{pk}」的取值示例（节选）：" + "；".join(parts) + "。")
    return "\n".join(lines)


def _build_rel_vector_text(rel: Dict[str, Any]) -> str:
    """单关系类型：供向量嵌入的自然语言描述。"""
    rt = rel.get("type") or ""
    ref = _cypher_ident(rt)
    lines: List[str] = [
        f"【图数据库关系类型】名称为 {ref}（Cypher 中写作 -[:{ref}]->）。",
        f"该类型关系约有 {rel.get('count', 0)} 条。",
    ]
    props = rel.get("properties") or []
    if props:
        lines.append("关系上的属性键包括：" + "、".join(props) + "。")
    samples = rel.get("property_value_samples") or {}
    for pk, vals in samples.items():
        if not vals:
            continue
        parts = [str(v) for v in vals]
        lines.append(f"属性「{pk}」的取值示例（节选）：" + "；".join(parts) + "。")
    return "\n".join(lines)


def _write_schema_vector_docs_jsonl(payload: Dict[str, Any], path: Path) -> None:
    """
    每行一个 JSON：单节点类型或单关系类型，供「一条文档一向量」检索。
    字段 text 为嵌入主字段；其余字段便于过滤与回填 Cypher。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for n in payload.get("nodes") or []:
            lbl = n.get("label") or ""
            doc = {
                "id": f"node:{lbl}",
                "kind": "node",
                "label": lbl,
                "count": n.get("count", 0),
                "properties": n.get("properties") or [],
                "property_value_samples": n.get("property_value_samples") or {},
                "text": _build_node_vector_text(n),
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        for r in payload.get("relationships") or []:
            rt = r.get("type") or ""
            doc = {
                "id": f"rel:{rt}",
                "kind": "relationship",
                "type": rt,
                "count": r.get("count", 0),
                "properties": r.get("properties") or [],
                "property_value_samples": r.get("property_value_samples") or {},
                "text": _build_rel_vector_text(r),
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def _distinct_values_for_rel_prop(
    session, rel_type: str, prop: str, limit: int
) -> List[Any]:
    rows = session.run(
        """
        MATCH ()-[r]->()
        WHERE type(r) = $rt AND $prop IN keys(r)
        WITH r[$prop] AS v
        WHERE v IS NOT NULL
        RETURN DISTINCT v AS v
        LIMIT $lim
        """,
        rt=rel_type,
        prop=prop,
        lim=int(limit),
    )
    out: List[Any] = []
    seen: set = set()
    for r in rows:
        raw = r["v"]
        serialized = _serialize_sample_value(raw)
        key = repr(serialized)
        if key in seen:
            continue
        seen.add(key)
        out.append(serialized)
    return out


def preprocess(
    out_path: Optional[Union[str, Path]] = None,
    *,
    value_sample_limit: int = 50,
    terms_path: Optional[Union[str, Path]] = None,
    vector_docs_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    连接 Neo4j，导出图谱本体（节点标签、关系类型、属性键及各类型数量）到 data/schema.json。
    优先使用 db.schema.* 过程；不可用时对各类型采样 keys() 作为回退。
    meta 中含库内节点总数、关系总数；各标签 count 按该标签统计（多标签节点会在多个标签下各计一次）。
    对每个属性额外导出 property_value_samples：去重后的属性值样本（含中文、英文等），便于 NL2Cypher / RAG。

    同步生成：
    - schema_terms.txt：去重排序后的词表，供 AC / 词典精准匹配；
    - schema_vector_docs.jsonl：每行一个节点类型或关系类型文档（含 text 字段），供单实体/单关系向量检索。
    """
    out = Path(out_path) if out_path is not None else _root / "data" / "schema.json"
    terms_out = Path(terms_path) if terms_path is not None else _root / "data" / "schema_terms.txt"
    vec_out = (
        Path(vector_docs_path)
        if vector_docs_path is not None
        else _root / "data" / "schema_vector_docs.jsonl"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    driver = None
    try:
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        driver.verify_connectivity()

        with driver.session() as session:
            labels = _fetch_labels(session)
            rel_types = _fetch_relationship_types(session)
            used_procedures = True
            try:
                node_props = _node_props_from_schema_procedures(session)
                rel_props = _rel_props_from_schema_procedures(session)
            except Exception:
                used_procedures = False
                node_props = {}
                rel_props = {}
                for lbl in labels:
                    node_props[lbl] = _sample_node_properties(session, lbl)
                for rt in rel_types:
                    rel_props[rt] = _sample_rel_properties(session, rt)

            for lbl in labels:
                node_props.setdefault(lbl, [])
            for rt in rel_types:
                rel_props.setdefault(rt, [])

            node_counts = {lbl: _count_nodes_by_label(session, lbl) for lbl in labels}
            rel_counts = {rt: _count_rels_by_type(session, rt) for rt in rel_types}

            nodes_out: List[Dict[str, Any]] = []
            for lbl in labels:
                props = sorted(set(node_props.get(lbl, [])))
                samples: Dict[str, List[Any]] = {}
                for p in props:
                    vals = _distinct_values_for_node_prop(
                        session, lbl, p, value_sample_limit
                    )
                    if vals:
                        samples[p] = vals
                nodes_out.append(
                    {
                        "label": lbl,
                        "count": node_counts.get(lbl, 0),
                        "properties": props,
                        "property_value_samples": samples,
                    }
                )

            rels_out: List[Dict[str, Any]] = []
            for rt in rel_types:
                props = sorted(set(rel_props.get(rt, [])))
                samples = {}
                for p in props:
                    vals = _distinct_values_for_rel_prop(
                        session, rt, p, value_sample_limit
                    )
                    if vals:
                        samples[p] = vals
                rels_out.append(
                    {
                        "type": rt,
                        "count": rel_counts.get(rt, 0),
                        "properties": props,
                        "property_value_samples": samples,
                    }
                )

            payload = {
                "meta": {
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                    "neo4j_uri": settings.neo4j_uri,
                    "source": "db.schema.*"
                    if used_procedures
                    else "sample_keys_fallback",
                    "total_nodes_in_db": _count_all_nodes(session),
                    "total_relationships_in_db": _count_all_relationships(session),
                    "value_sample_limit": value_sample_limit,
                },
                "nodes": nodes_out,
                "relationships": rels_out,
            }
    except Exception as e:
        raise RuntimeError(
            "Neo4j schema 导出失败。请确认：1) 已安装驱动：pip install neo4j；"
            "2) Neo4j 服务已启动；3) config / .env 中 URI、用户名、密码正确。"
        ) from e
    finally:
        if driver is not None:
            driver.close()

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    out.write_text(text, encoding="utf-8")

    term_list = _collect_schema_terms(payload)
    _write_schema_terms(term_list, terms_out)
    _write_schema_vector_docs_jsonl(payload, vec_out)

    return payload


if __name__ == "__main__":
    preprocess()
    data = _root / "data"
    print(f"已写入: {data / 'schema.json'}")
    print(f"已写入: {data / 'schema_terms.txt'}")
    print(f"已写入: {data / 'schema_vector_docs.jsonl'}")
