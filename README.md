# 🔮 Tarot — 塔罗牌 RAG 智能知识库

基于 **DeepSeek + ChromaDB + FastAPI + React** 构建的塔罗牌智能问答与知识管理系统。

---

## 项目动机

市面上的塔罗牌资料分散在书籍、博客、视频中，查找效率低、信息碎片化。**Taro** 将 78 张塔罗牌、15 种牌阵、历史流派、元素对应等知识整合为结构化知识库，通过 RAG（检索增强生成）技术，让用户用自然语言提问即可获得专业、有据可查的塔罗解读。

**核心价值：**
- **知识可溯源** — 每条回答都标注了引用来源和相关度评分
- **知识可维护** — Markdown 文件即知识源，编辑后增量入库，无需重新训练
- **多入口访问** — REST API / Web 管理台 / 命令行，适配不同场景

---

## 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                      用户入口                             │
│   Web Admin UI (/admin)  │    REST API (8000)    │  CLI  │
└────────────┬────────────────────┬────────────────┬───────┘
             │                    │                │
             ▼                    ▼                ▼
┌─────────────────────────────────────────────────────────┐
│                    RAG 查询引擎                           │
│   query() → search_knowledge() → _build_context()        │
│                     → LLM 生成                            │
└────────────┬─────────────────────────────────────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
┌──────────┐   ┌──────────────┐
│ ChromaDB │   │ DeepSeek API │
│ 向量检索  │   │ LLM+Embedding│
└──────────┘   └──────────────┘
       ▲
       │  ingest.py (入库)
       │
┌─────────────────────────────────────────────────────────┐
│                   知识层 (Markdown)                       │
│  major_arcana/  minor_arcana/  spreads/  history/        │
│  elements/  custom/                                      │
└─────────────────────────────────────────────────────────┘
```

**数据流：**
1. Markdown 文件 → `ingest.py` 分块 → DeepSeek Embedding → ChromaDB 向量存储
2. 用户提问 → Embedding 向量化 → ChromaDB 相似度检索 → 拼接上下文 → DeepSeek LLM 生成回答

---

## 快速开始

### 1. 环境准备

```bash
# Python 3.10+
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
# 创建 .env 文件
cat > .env << EOF
DEEPSEEK_API_KEY=sk-your-deepseek-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-flash
EMBEDDING_PROVIDER=deepseek
EMBEDDING_MODEL=deepseek-embeddings
EOF
```

### 3. 构建知识库索引（首次运行）

```bash
python3 src/ingest.py --rebuild
```

> 首次构建会向量化约 102 个文件，耗时约 2-5 分钟。

### 4. 启动服务

```bash
# 启动 FastAPI（端口 8000，含 Web 管理台）
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. 开始使用

- **Web 管理台**: 打开 `http://localhost:8000/admin`，在「智能问答」页面提问
- **API 文档**: 打开 `http://localhost:8000/docs` 查看 Swagger
- **命令行**: `python3 src/ingest.py --stats` 查看知识库统计

---

## API 接口

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/` | 服务状态 |
| POST | `/query` | **核心问答**（检索 + LLM 生成） |
| GET | `/search?q=xxx` | 纯向量搜索（不调用 LLM） |
| GET | `/stats` | 知识库统计 |
| POST | `/rebuild` | 全量重建索引 |
| POST | `/update` | 增量更新索引 |

**问答示例：**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "愚者牌逆位代表什么？", "top_k": 5}'
```

---

## 设计决策与踩坑记录

### 为什么选 ChromaDB 而不是 FAISS？

| 维度 | ChromaDB | FAISS |
|------|----------|-------|
| 部署复杂度 | `pip install` 即用，零配置 | 需编译 C++ 扩展，macOS 上容易出问题 |
| 持久化 | 内置持久化，重启不丢数据 | 需手动序列化/反序列化索引文件 |
| 元数据过滤 | 原生支持 `where` 条件过滤 | 需自己维护 ID 映射表 |
| 生态集成 | 与 LangChain/LlamaIndex 深度集成 | 偏底层，需更多胶水代码 |
| 适用场景 | 中小规模（<100万向量），快速原型 | 大规模（>百万级），极致性能 |

**结论：** 本项目知识库约 500-1000 个 chunk，ChromaDB 的简洁性和内置元数据过滤完全满足需求。FAISS 的极致性能在这里是过度设计。

### 为什么用 DeepSeek 做 Embedding？

1. **统一供应商** — LLM 和 Embedding 都用 DeepSeek，减少 API Key 管理复杂度
2. **中文友好** — DeepSeek Embedding 对中文语义理解优于很多开源模型
3. **性价比** — 价格远低于 OpenAI text-embedding-3，且 1024 维度足够表达塔罗牌语义
4. **备选方案** — 代码保留了 DashScope（阿里云）Embedding 切换能力，一行配置即可切换

### 为什么用 Markdown + Frontmatter 管理知识？

- **人机共读** — 既可以直接用编辑器修改，也能被程序解析
- **Git 友好** — 纯文本，diff 清晰，方便版本管理和协作
- **增量更新** — 通过文件 hash 检测变更，只重新入库修改过的文件
- **结构化元数据** — YAML frontmatter 携带分类、名称等信息，入库时自动提取

### 分块策略的选择

- **CHUNK_SIZE=600** — 塔罗牌单张牌义约 300-800 字，600 字的分块能保证每块包含完整语义
- **CHUNK_OVERLAP=100** — 适度的重叠避免关键信息在边界处被切断
- **SCORE_THRESHOLD=0.3** — 经过测试，低于 0.3 的结果基本不相关，过滤掉能减少 LLM 幻觉

---

## 项目结构

```
taro/
├── knowledge/                  # 知识源文件（Markdown，可直接编辑）
│   ├── major_arcana/           # 大阿卡纳 22 张
│   ├── minor_arcana/           # 小阿卡纳 56 张
│   ├── spreads/                # 15 种牌阵
│   ├── history/                # 历史起源、流派、学习指南
│   ├── elements/               # 元素、数字学、占星对应
│   └── custom/                 # 自定义知识（可通过 UI 管理）
├── vectorstore/                # ChromaDB 持久化向量库（自动生成）
├── src/
│   ├── ingest.py               # 知识入库（全量/增量/单文件）
│   ├── rag_engine.py           # RAG 查询引擎（检索 + LLM 生成）
│   ├── api/main.py             # FastAPI REST 服务
│   └── admin/index.html        # React Web 管理台
├── scripts/
│   ├── generate_knowledge.py   # 生成 78 张牌 Markdown 文件
│   └── gen_spreads.py          # 生成牌阵/历史/元素 Markdown 文件
├── config.py                   # 全局配置
├── requirements.txt
└── README.md
```

---

## 知识库内容清单

| 分类 | 数量 | 内容 |
|------|------|------|
| 大阿卡纳 | 22 张 | 正逆位含义、象征、数字学、占星对应 |
| 小阿卡纳 | 56 张 | 权杖/圣杯/宝剑/星币各 14 张 |
| 牌阵 | 15 种 | 单张、三牌、凯尔特十字、关系、七脉轮等 |
| 历史与方法 | 5 篇 | 起源、流派、占卜方法、学习路径 |
| 元素与对应 | 4 篇 | 四元素、数字学、占星、卡巴拉 |
| **合计** | **102 个文件** | |

