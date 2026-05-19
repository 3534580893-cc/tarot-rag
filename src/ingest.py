"""
塔罗牌 RAG — 知识入库 & 增量更新
负责：文件解析 → 文本分块 → Embedding → 写入 ChromaDB
"""
import hashlib
import re
import sys
from pathlib import Path

import chromadb
from openai import OpenAI

# 确保项目根目录可被 import
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ── ChromaDB 客户端 ─────────────────────────────

def _get_chroma_client():
    """获取 ChromaDB 持久化客户端"""
    return chromadb.PersistentClient(path=str(config.VECTORSTORE_DIR))


def _get_collection(client=None):
    """获取或创建 ChromaDB collection"""
    if client is None:
        client = _get_chroma_client()
    return client.get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


# ── Embedding ───────────────────────────────────

def _embed_texts(texts: list[str]) -> list[list[float]]:
    """调用 Embedding API 批量向量化（支持 DeepSeek 或 DashScope）"""
    if config.EMBEDDING_PROVIDER == "deepseek":
        # 使用 DeepSeek Embedding
        client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
        )
        resp = client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=texts,
        )
        return [item.embedding for item in resp.data]
    else:
        # 使用 DashScope Embedding（备选）
        import dashscope
        resp = dashscope.TextEmbedding.call(
            model=config.EMBEDDING_MODEL,
            input=texts,
            api_key=config.DASHSCOPE_API_KEY,
            dimension=config.EMBEDDING_DIMENSION,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Embedding API 调用失败: {resp.code} - {resp.message}")
        return [item["embedding"] for item in resp.output["embeddings"]]


# ── 文件解析 ────────────────────────────────────

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """解析 Markdown frontmatter（--- 块），返回 (metadata_dict, body)"""
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', text, re.DOTALL)
    if not match:
        return {}, text
    meta = {}
    for line in match.group(1).split('\n'):
        if ':' in line:
            key, val = line.split(':', 1)
            meta[key.strip()] = val.strip()
    body = text[match.end():]
    return meta, body


def _chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    """将文本按字符数分块"""
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


def _file_hash(filepath: Path) -> str:
    """计算文件 MD5 哈希"""
    return hashlib.md5(filepath.read_bytes()).hexdigest()


def _infer_category(filepath: Path) -> str:
    """从文件路径推断分类"""
    relative = filepath.relative_to(config.KNOWLEDGE_DIR)
    parts = relative.parts
    if len(parts) > 1:
        return parts[0]
    return "uncategorized"


# ── 核心入库函数 ────────────────────────────────

def ingest_file(filepath: Path, collection=None, verbose=True) -> int:
    """
    将单个 Markdown 文件入库到 ChromaDB
    返回写入的 chunk 数量
    """
    if collection is None:
        collection = _get_collection()

    text = filepath.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(text)
    if not body.strip():
        return 0

    category = meta.get("category", _infer_category(filepath))
    name = meta.get("name", meta.get("id", filepath.stem))
    source_file = str(filepath.relative_to(config.KNOWLEDGE_DIR))
    file_hash = _file_hash(filepath)

    # 先删除该文件旧的 chunks
    try:
        existing = collection.get(where={"source_file": source_file})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    # 分块
    chunks = _chunk_text(body)
    if not chunks:
        return 0

    # Embedding
    embeddings = _embed_texts(chunks)

    # 写入 ChromaDB
    ids = [f"{source_file}::chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "name": name,
            "category": category,
            "source_file": source_file,
            "file_hash": file_hash,
            "chunk_index": i,
            "total_chunks": len(chunks),
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    if verbose:
        print(f"  ✅ {source_file} → {len(chunks)} chunks")
    return len(chunks)


# ── 对外接口：app.py 调用 ──────────────────────

def get_stats() -> dict:
    """获取知识库统计信息"""
    collection = _get_collection()
    all_data = collection.get(include=["metadatas"])

    categories = {}
    files = set()
    for meta in all_data["metadatas"]:
        cat = meta.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
        files.add(meta.get("source_file", ""))

    return {
        "total_chunks": len(all_data["ids"]),
        "total_files": len(files),
        "categories": categories,
    }


def add_document(name: str, category: str, content: str) -> dict:
    """添加新知识条目（保存文件并入库）"""
    # 生成文件名
    safe_name = re.sub(r'[^\w\u4e00-\u9fff]', '_', name).strip('_')
    custom_dir = config.KNOWLEDGE_DIR / "custom"
    custom_dir.mkdir(parents=True, exist_ok=True)
    filepath = custom_dir / f"{safe_name}.md"

    # 写 Markdown 文件（带 frontmatter）
    md_content = f"""---
id: {safe_name}
name: {name}
category: {category}
---

{content}
"""
    filepath.write_text(md_content, encoding="utf-8")

    # 入库
    collection = _get_collection()
    chunks = ingest_file(filepath, collection, verbose=False)

    return {"file": str(filepath), "chunks": chunks}


def delete_document(source_id: str) -> dict:
    """删除知识条目（文件 + 向量）"""
    collection = _get_collection()

    # 删向量
    try:
        existing = collection.get(where={"source_file": source_id})
        deleted = len(existing["ids"])
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        deleted = 0

    # 删文件
    filepath = config.KNOWLEDGE_DIR / f"{source_id}.md"
    if filepath.exists():
        filepath.unlink()

    return {"chunks_deleted": deleted}


def update_changed(verbose=True) -> dict:
    """增量更新：检测变更文件并重新入库"""
    collection = _get_collection()

    # 获取已入库文件的 hash
    all_data = collection.get(include=["metadatas"])
    existing_hashes = {}
    for meta in all_data["metadatas"]:
        sf = meta.get("source_file", "")
        fh = meta.get("file_hash", "")
        if sf:
            existing_hashes[sf] = fh

    added = 0
    changed = 0
    total_chunks = 0

    # 遍历知识目录
    for md_file in sorted(config.KNOWLEDGE_DIR.rglob("*.md")):
        if md_file.name == "README.md":
            continue
        source_file = str(md_file.relative_to(config.KNOWLEDGE_DIR))
        current_hash = _file_hash(md_file)

        if source_file not in existing_hashes:
            # 新文件
            chunks = ingest_file(md_file, collection, verbose=verbose)
            total_chunks += chunks
            added += 1
        elif existing_hashes[source_file] != current_hash:
            # 已修改
            chunks = ingest_file(md_file, collection, verbose=verbose)
            total_chunks += chunks
            changed += 1

    return {"added": added, "changed": changed, "chunks": total_chunks}


def rebuild_all(verbose=True) -> dict:
    """全量重建：清空向量库并重新入库所有文件"""
    client = _get_chroma_client()

    # 删除旧 collection
    try:
        client.delete_collection(config.CHROMA_COLLECTION)
    except Exception:
        pass

    collection = _get_collection(client)

    files_count = 0
    total_chunks = 0

    for md_file in sorted(config.KNOWLEDGE_DIR.rglob("*.md")):
        if md_file.name == "README.md":
            continue
        chunks = ingest_file(md_file, collection, verbose=verbose)
        if chunks > 0:
            files_count += 1
            total_chunks += chunks

    return {"files": files_count, "chunks": total_chunks}


# ── CLI 入口 ────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="塔罗知识库入库工具")
    parser.add_argument("--rebuild", action="store_true", help="全量重建索引")
    parser.add_argument("--update", action="store_true", help="增量更新")
    parser.add_argument("--stats", action="store_true", help="查看统计")
    args = parser.parse_args()

    if args.rebuild:
        print("🔄 开始全量重建...")
        result = rebuild_all()
        print(f"✅ 完成！{result['files']} 个文件，{result['chunks']} 个 chunk")
    elif args.update:
        print("⚡ 开始增量更新...")
        result = update_changed()
        print(f"✅ 完成！新增 {result['added']}，修改 {result['changed']}，共 {result['chunks']} chunk")
    elif args.stats:
        stats = get_stats()
        print(f"总 Chunk 数: {stats['total_chunks']}")
        print(f"知识文件数: {stats['total_files']}")
        for k, v in stats['categories'].items():
            print(f"  {k}: {v}")
    else:
        parser.print_help()
