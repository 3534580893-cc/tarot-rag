"""
塔罗牌 RAG — FastAPI 服务
运行：uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(
    title="🔮 塔罗牌 RAG API",
    description="基于 DeepSeek + ChromaDB 的塔罗牌智能问答服务",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 请求/响应模型 ───────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    category: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


class SearchResponse(BaseModel):
    results: list[dict]
    total: int


class StatsResponse(BaseModel):
    total_chunks: int
    total_files: int
    categories: dict


class AddDocRequest(BaseModel):
    name: str
    category: str
    content: str


class UpdateDocRequest(BaseModel):
    id: str
    content: str


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: str  # 'like' | 'dislike'


# ── 静态文件（前端 SPA）────────────────────────

ADMIN_DIR = Path(__file__).parent.parent / "admin"
if ADMIN_DIR.exists():
    app.mount("/admin", StaticFiles(directory=str(ADMIN_DIR), html=True), name="admin")


# ── 接口 ────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "🔮 塔罗牌 RAG API 运行中", "docs": "/docs", "admin": "/admin"}


@app.get("/config")
def get_config():
    """前端读取运行时配置"""
    return {
        "llm_model": config.LLM_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_provider": config.EMBEDDING_PROVIDER,
        "top_k": config.TOP_K,
    }


@app.post("/query", response_model=QueryResponse)
def rag_query(req: QueryRequest):
    """RAG 智能问答"""
    try:
        from src.rag_engine import query
        result = query(
            question=req.question,
            top_k=req.top_k,
            category_filter=req.category,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., description="搜索关键词"),
    top_k: int = Query(8, ge=1, le=30),
    category: str | None = Query(None),
):
    """知识库向量检索"""
    try:
        from src.rag_engine import search_knowledge
        docs = search_knowledge(query=q, top_k=top_k, category=category)
        return {"results": docs, "total": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
def stats():
    """知识库统计"""
    try:
        from src.ingest import get_stats
        return get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rebuild")
def rebuild():
    """全量重建索引"""
    try:
        from src.ingest import rebuild_all
        result = rebuild_all(verbose=False)
        return {"status": "ok", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update")
def update():
    """增量更新索引"""
    try:
        from src.ingest import update_changed
        result = update_changed(verbose=False)
        return {"status": "ok", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/knowledge-files")
def knowledge_files():
    """获取知识文件目录树"""
    result = {}
    if config.KNOWLEDGE_DIR.exists():
        for cat_dir in sorted(config.KNOWLEDGE_DIR.iterdir()):
            if cat_dir.is_dir():
                files = [str(f.relative_to(config.KNOWLEDGE_DIR)) for f in sorted(cat_dir.rglob("*.md"))]
                if files:
                    result[cat_dir.name] = files
    return result


@app.post("/add-document")
def add_document_api(req: AddDocRequest):
    """添加自定义知识条目"""
    try:
        from src.ingest import add_document
        result = add_document(name=req.name, category=req.category, content=req.content)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/custom-documents")
def custom_documents():
    """获取 custom/ 目录下的所有文档"""
    custom_dir = config.KNOWLEDGE_DIR / "custom"
    result = []
    if custom_dir.exists():
        for f in sorted(custom_dir.glob("*.md")):
            result.append({"id": f.stem, "content": f.read_text(encoding="utf-8")})
    return result


@app.post("/update-document")
def update_document(req: UpdateDocRequest):
    """保存并重新入库自定义文档"""
    try:
        custom_dir = config.KNOWLEDGE_DIR / "custom"
        f = custom_dir / f"{req.id}.md"
        if not f.exists():
            raise HTTPException(status_code=404, detail="文件不存在")
        f.write_text(req.content, encoding="utf-8")
        from src.ingest import ingest_file, _get_chroma_client, _get_collection
        client = _get_chroma_client()
        col = _get_collection(client)
        ingest_file(f, col, verbose=False)
        return {"status": "ok", "id": req.id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete-document/{source_id}")
def delete_document_api(source_id: str):
    """删除自定义文档"""
    try:
        from src.ingest import delete_document
        result = delete_document(f"custom/{source_id}")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
def save_feedback(req: FeedbackRequest):
    """保存用户对问答的反馈"""
    feedback_file = config.BASE_DIR / "feedback.json"
    try:
        data = json.loads(feedback_file.read_text(encoding="utf-8")) if feedback_file.exists() else []
        data.append({
            "question": req.question,
            "answer": req.answer,
            "rating": req.rating,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        feedback_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 启动 ────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
    )
