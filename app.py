"""
FastAPI 入口：自然语言 → Cypher → Neo4j 执行。

启动（在项目根目录）:
  uvicorn app:app --reload --host 0.0.0.0 --port 8000
或:
  python app.py
"""

from __future__ import annotations

import datetime
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加项目根目录到 Python 路径
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

STATIC_DIR = _root / "static"

from fastapi import FastAPI, HTTPException, status
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

_agent: Any | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global _agent
    try:
        # 延迟导入，避免启动时加载错误
        from core.processor import CypherAgent
        _agent = CypherAgent()
        logger.info("CypherAgent 初始化成功")
    except Exception as e:
        logger.error(f"CypherAgent 初始化失败: {e}")
        _agent = None
    yield
    _agent = None
    logger.info("应用关闭")


app = FastAPI(
    title="Neo4j NL2Cypher Agent",
    description="精准匹配 + 向量召回 + LangGraph Agent 生成并执行 Cypher",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 配置（生产环境请限制域名）
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origin_regex=r"https?://.*",  # 可调整
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="自然语言问题", example="查询与工作电流相关的故障模式")


class FewShotItem(BaseModel):
    question: str = ""
    cypher: str = ""
    similarity: Optional[float] = None


class QueryResponse(BaseModel):
    user_query: str
    exact_match_keywords: List[str] = []
    vector_schema_keywords: List[str] = []
    vector_schema_hits: List[Dict[str, Any]] = []
    few_shot_retrieved: List[FewShotItem] = []
    generated_cypher: str = ""
    last_llm_text: str = ""
    execution_status: Optional[str] = None
    execution_data: List[Dict[str, Any]] = []
    execution_error: Optional[str] = None
    failure_count: int = 0


def _to_response(result: Dict[str, Any]) -> QueryResponse:
    """转换内部结果为响应模型"""
    ex = result.get("execution") or {}
    few = result.get("few_shot_retrieved") or []
    
    few_models = [
        FewShotItem(
            question=str(x.get("question") or ""),
            cypher=str(x.get("cypher") or ""),
            similarity=x.get("similarity"),
        )
        for x in few
    ]
    
    return QueryResponse(
        user_query=str(result.get("user_query") or ""),
        exact_match_keywords=list(result.get("exact_match_keywords") or []),
        vector_schema_keywords=list(result.get("vector_schema_keywords") or []),
        vector_schema_hits=list(result.get("vector_schema_hits") or []),
        few_shot_retrieved=few_models,
        generated_cypher=str(result.get("generated_cypher") or ""),
        last_llm_text=str(result.get("last_llm_text") or ""),
        execution_status=ex.get("status"),
        execution_data=list(ex.get("data") or []),
        execution_error=ex.get("error"),
        failure_count=int(result.get("failure_count") or 0),
    )


@app.get("/")
async def serve_ui():
    """Web 前端：在浏览器中打开 http://127.0.0.1:8000/ 输入问题并查看结果"""
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        return {
            "message": "未找到 static/index.html",
            "docs": "/docs",
            "health": "/health",
        }
    return FileResponse(index)


app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)


@app.get("/health", status_code=status.HTTP_200_OK)
async def health():
    """健康检查"""
    agent_status = "loaded" if _agent is not None else "not_loaded"
    return {
        "status": "ok",
        "agent": agent_status,
        "timestamp": datetime.datetime.now().isoformat()
    }


@app.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query(req: QueryRequest):
    """执行自然语言查询"""
    if _agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CypherAgent 未初始化，请检查服务启动日志"
        )
    
    try:
        logger.info(f"收到查询请求: {req.question[:50]}...")
        result = _agent.run(req.question.strip())
        resp = _to_response(result)
        
        # 记录执行结果
        status_code = resp.execution_status
        logger.info(f"查询执行状态: {status_code}, 返回 {len(resp.execution_data)} 条数据")
        
        return jsonable_encoder(resp.model_dump())
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询处理失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"查询处理失败: {str(e)}"
        ) from e


@app.post("/query/raw", status_code=status.HTTP_200_OK)
async def query_raw(req: QueryRequest):
    """返回原始结果（用于调试）"""
    if _agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent 未初始化"
        )
    
    try:
        logger.info(f"收到原始查询请求: {req.question[:50]}...")
        result = _agent.run(req.question.strip())
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"原始查询处理失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"原始查询处理失败: {str(e)}"
        ) from e


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 开发时使用 reload
        log_level="info"
    )