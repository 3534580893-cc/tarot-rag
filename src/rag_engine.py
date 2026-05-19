"""
塔罗牌 RAG — 查询核心
负责：问题向量化 → 检索相关文档 → 构造 Prompt → 调用 LLM 生成回答
"""
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.ingest import _embed_texts, _get_collection


# ── 检索 ────────────────────────────────────────

def search_knowledge(
    query: str,
    top_k: int = None,
    category: str = None,
    score_threshold: float = None,
) -> list[dict]:
    """
    纯向量检索（不调用 LLM）
    返回: [{"document": str, "metadata": dict, "score": float}, ...]
    """
    top_k = top_k or config.TOP_K
    score_threshold = score_threshold or config.SCORE_THRESHOLD
    collection = _get_collection()

    # 向量化查询
    query_embedding = _embed_texts([query])[0]

    # 构造查询参数
    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if category:
        kwargs["where"] = {"category": category}

    results = collection.query(**kwargs)

    # 整理结果（ChromaDB 返回的 distance 是余弦距离，score = 1 - distance）
    docs = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        score = 1 - distance  # 余弦相似度
        if score < score_threshold:
            continue
        docs.append({
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": score,
        })

    return docs


# ── RAG 查询 ────────────────────────────────────

SYSTEM_PROMPT = """你是一位专业的塔罗牌解读师和知识顾问。你拥有丰富的塔罗牌知识，包括大阿卡纳、小阿卡纳、各种牌阵、象征体系和历史背景。

请根据提供的知识库内容来回答用户的问题。回答要求：
1. 基于知识库内容回答，不要编造不存在的信息
2. 回答风格专业但易懂，适当使用塔罗牌术语
3. 如果知识库中没有相关信息，请诚实告知并给出你的理解
4. 适当引用来源，帮助用户深入了解"""


def _build_context(docs: list[dict]) -> str:
    """将检索到的文档拼接为上下文"""
    if not docs:
        return "（知识库中未找到相关内容）"
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc["metadata"]
        name = meta.get("name", "未知")
        category = meta.get("category", "")
        parts.append(f"【来源 {i}】{name}（{category}）\n{doc['document']}")
    return "\n\n---\n\n".join(parts)


def _build_messages(
    question: str,
    context: str,
    chat_history: list[dict] = None,
) -> list[dict]:
    """构造 LLM 消息列表"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 加入历史对话（最近 6 轮）
    if chat_history:
        for msg in chat_history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    # 用户问题 + 检索上下文
    user_msg = f"""请根据以下知识库内容回答我的问题。

## 知识库内容：
{context}

## 我的问题：
{question}"""

    messages.append({"role": "user", "content": user_msg})
    return messages


def query(
    question: str,
    top_k: int = None,
    category_filter: str = None,
    chat_history: list[dict] = None,
) -> dict:
    """
    RAG 完整查询流程：检索 → 生成
    返回: {"answer": str, "sources": [{"name", "category", "score"}, ...]}
    """
    # 1. 检索相关文档
    docs = search_knowledge(
        query=question,
        top_k=top_k,
        category=category_filter,
    )

    # 2. 构造上下文和消息
    context = _build_context(docs)
    messages = _build_messages(question, context, chat_history)

    # 3. 调用 DeepSeek LLM
    client = OpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
    )
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
    )

    answer = resp.choices[0].message.content

    # 4. 整理来源信息
    sources = []
    for doc in docs:
        meta = doc["metadata"]
        sources.append({
            "name": meta.get("name", "未知"),
            "category": meta.get("category", ""),
            "score": doc["score"],
        })

    return {"answer": answer, "sources": sources}
