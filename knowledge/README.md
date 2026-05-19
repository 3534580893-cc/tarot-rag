# 🔮 塔罗牌 RAG 知识库

基于 **DeepSeek + ChromaDB + FastAPI + React** 构建的塔罗牌智能问答与知识管理系统。

---

## 📦 项目结构

```
tarot-rag/
├── knowledge/                  ← 所有知识源文件（直接可编辑的 Markdown）
│   ├── major_arcana/           ← 大阿卡纳 22张
│   ├── minor_arcana/           ← 小阿卡纳 56张（wands/cups/swords/pentacles）
│   ├── spreads/                ← 15种牌阵
│   ├── history/                ← 历史起源、流派、学习指南
│   ├── elements/               ← 元素、数字学、占星、卡巴拉对应
│   └── custom/                 ← 你自定义添加的知识（可通过 UI 管理）
├── vectorstore/                ← ChromaDB 持久化向量库（自动生成）
├── src/
│   ├── ingest.py               ← 知识入库核心（全量/增量/单文件）
│   ├── rag_engine.py           ← RAG 查询引擎（检索 + LLM 生成）
│   ├── api/main.py             ← FastAPI REST 服务
│   └── admin/index.html        ← React Web 管理台
├── scripts/
│   ├── generate_knowledge.py   ← 生成大小阿卡纳 Markdown 文件
│   └── gen_spreads.py          ← 生成牌阵/历史/元素 Markdown 文件
├── config.py                   ← 全局配置
├── .env.example                ← 环境变量模板
├── requirements.txt
└── start.sh                    ← 一键启动脚本
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key
```

```env
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
LLM_MODEL=deepseek-v4-flash
EMBEDDING_PROVIDER=deepseek
EMBEDDING_MODEL=deepseek-embeddings
```

### 3. 构建知识库索引（首次运行）

```bash
python3 src/ingest.py --rebuild
```

> 首次构建会调用 Embedding API 向量化所有 102 个文件，约需 3-8 分钟。

### 4. 启动服务

```bash
# 启动 FastAPI（端口 8000，含 Web 管理台）
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📡 REST API 文档

启动后访问 `http://localhost:8000/docs` 查看完整 Swagger 文档。

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/stats` | 知识库统计 |
| POST | `/query` | **核心问答**（检索 + LLM） |
| POST | `/search` | 纯向量搜索（不调用 LLM） |
| GET | `/cards/{name}` | 查询单张牌详情 |
| POST | `/knowledge/add` | 添加新知识条目 |
| PUT | `/knowledge/{id}` | 更新知识条目 |
| DELETE | `/knowledge/{id}` | 删除知识条目 |
| POST | `/index/rebuild` | 全量重建索引 |
| POST | `/index/update` | 增量更新索引 |

### 问答示例（curl）

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{
    "question": "凯尔特十字牌阵怎么解读？",
    "top_k": 5
  }'
```

---

## 🔧 知识库维护指南

### 方式 A：直接编辑 Markdown 文件（推荐）

1. 在 `knowledge/` 对应子目录找到并编辑 `.md` 文件
2. 运行增量更新：`python3 src/ingest.py --update`
3. 或通过管理界面的「索引维护」→「增量更新」

### 方式 B：Web 管理台

访问 `http://localhost:8000/admin`：
- **智能问答**：实时测试知识库效果
- **添加知识**：填写表单，自动保存文件并入库
- **知识管理**：编辑/删除 custom 目录下的自定义条目
- **知识检索**：验证某条内容是否正确入库
- **索引维护**：增量更新或全量重建

### 方式 C：REST API

```bash
# 添加新知识
curl -X POST http://localhost:8000/knowledge/add \
  -H "Content-Type: application/json" \
  -d '{"name":"我的笔记","category":"custom","content":"# 笔记内容..."}'

# 触发增量更新
curl -X POST http://localhost:8000/index/update
```

---

## 📁 Markdown 文件规范

每个知识文件需要包含 YAML frontmatter：

```markdown
---
id: 唯一标识（英文，不含空格）
name: 显示名称（中文）
category: major_arcana / minor_arcana / spread / history / elements / custom
---

# 标题

## 正文内容（支持完整 Markdown 语法）
...
```

---

## ⚙️ 配置说明

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `LLM_MODEL` | `deepseek-v4-flash` | DeepSeek 模型，可换 `deepseek-v4-pro` |
| `EMBEDDING_MODEL` | `deepseek-embeddings` | 向量化模型（1024维） |
| `CHUNK_SIZE` | `600` | 文本分块大小（字符数） |
| `CHUNK_OVERLAP` | `100` | 分块重叠字符数 |
| `TOP_K` | `5` | 检索返回的相关文档数 |
| `SCORE_THRESHOLD` | `0.3` | 相关性阈值（0-1） |

---

## 🌐 阿里云部署

```bash
# 1. 在 ECS 上克隆项目
git clone <repo> && cd tarot-rag

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env && vim .env

# 4. 构建索引
python3 src/ingest.py --rebuild

# 5. 使用 supervisor 或 systemd 管理进程
# FastAPI（后台）
nohup python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

# Web 管理台内置于 FastAPI，无需单独启动，访问 http://localhost:8000/admin 即可
```

---

## 📊 知识库内容清单

| 分类 | 数量 | 内容 |
|------|------|------|
| 大阿卡纳 | 22张 | 正逆位含义、象征、数字学、占星对应 |
| 小阿卡纳-权杖 | 14张 | Ace-10 + 侍从/骑士/王后/国王 |
| 小阿卡纳-圣杯 | 14张 | Ace-10 + 侍从/骑士/王后/国王 |
| 小阿卡纳-宝剑 | 14张 | Ace-10 + 侍从/骑士/王后/国王 |
| 小阿卡纳-星币 | 14张 | Ace-10 + 侍从/骑士/王后/国王 |
| 牌阵 | 15种 | 单张、三牌、凯尔特十字、关系、马蹄、七脉轮等 |
| 历史与方法 | 5篇 | 起源、流派、如何占卜、78张结构、学习路径 |
| 元素与对应 | 4篇 | 四元素、数字学、占星、卡巴拉 |
| **合计** | **102个文件** | |
