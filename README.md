# SuperMew

SuperMew 是一个基于 FastAPI + Vue 的本地 RAG 问答项目。

当前代码主线以“稳定、可演示、可调试”为目标，默认使用基线检索模式：

- chunk-level hybrid retrieval
- rerank
- auto-merge / page-merge
- 会话记忆
- 管理员文档上传、删除、批量删除

项目中还保留了面向 FinanceBench 的实验性链路，但默认不启用，不会接管主流程。

## 当前实际能力

- 用户注册、登录、JWT 鉴权
- 普通聊天接口和流式聊天接口
- 会话列表、会话消息查询、删除会话
- 管理员上传文档到知识库
- 管理员异步上传文档并查看任务进度
- 管理员删除单个文档、批量删除文档，并查看删除任务进度
- 支持文档格式：
  - PDF
  - Word：`.doc` / `.docx`
  - Excel：`.xls` / `.xlsx`
  - Text：`.txt`
  - Markdown：`.md`
  - CSV：`.csv`
- 文档解析后进行三级分块
- 叶子分块写入 Milvus，父级分块写入 PostgreSQL
- BM25 sparse + dense embedding 的混合检索
- RAG trace 返回基础检索信息，兼容现有 LangSmith 评估读取
- `/debug/retrieval` 调试接口

## 默认检索模式

项目默认模式是：

```env
RAG_RETRIEVAL_MODE=baseline
```

`baseline` 模式下：

- 使用稳定的 chunk-level 检索主路径
- 不使用 page-level index 接管主检索
- 不使用 query parser 强过滤
- 不用 experimental evidence pack 覆盖主 prompt
- 生成阶段基于最终 `final_retrieved_chunks / context_docs` 回答

可选实验模式：

```env
RAG_RETRIEVAL_MODE=finance_experimental
```

这个模式保留了面向 FinanceBench 的 page-level / evidence-pack 实验逻辑，但不建议直接作为主系统默认模式。

## 技术栈

- 后端：FastAPI、SQLAlchemy、LangChain、LangGraph
- 前端：Vue 3 CDN 单页
- 向量库：Milvus
- 数据库：PostgreSQL
- 缓存：Redis
- Embedding：`langchain-huggingface` + `sentence-transformers`
- 稀疏检索：本地 BM25 sparse 特征
- 可选 rerank：外部 rerank 服务

## 项目结构

```text
SuperMew/
├─ backend/                # FastAPI 后端、RAG、鉴权、数据访问
├─ frontend/               # Vue 前端
├─ data/                   # 上传文档、BM25 状态等本地数据
├─ datasets/               # 数据集
├─ langsmith_eval/         # 评估脚本
├─ log/                    # 改动日志
├─ docker-compose.yml      # PostgreSQL / Redis / Milvus 依赖
├─ pyproject.toml          # Python 依赖
└─ README.md
```

## 环境要求

- Python 3.12+
- Docker / Docker Compose
- 推荐使用 `uv`

如果你已经用 Conda 管理环境，也可以先激活自己的环境再安装依赖，例如：

```powershell
conda activate rag
```

## 快速开始

### 1. 安装依赖

推荐方式：

```bash
uv sync
```

如果你不用 `uv`：

```bash
pip install -e .
```

### 2. 准备环境变量

复制一份示例配置：

```powershell
Copy-Item .env.example .env
```

至少需要检查这些配置：

- `ARK_API_KEY`
- `MODEL`
- `BASE_URL`
- `DATABASE_URL`
- `REDIS_URL`
- `MILVUS_HOST`
- `MILVUS_PORT`
- `JWT_SECRET_KEY`
- `ADMIN_INVITE_CODE`

常用检索配置：

```env
MILVUS_SEARCH_EF=128
RAG_RETRIEVAL_MODE=baseline
FINANCE_RAG_CANDIDATE_K=50
FINANCE_RAG_FINAL_TOP_K=10
FINANCE_RAG_ENABLE_STEP_BACK=false
FINANCE_RAG_ENABLE_PAGE_MERGE=true
```

说明：

- `MILVUS_SEARCH_EF` 会自动与实际搜索 `k` 协调，避免 `ef <= k` 导致的 Milvus 检索报错。
- `RAG_RETRIEVAL_MODE=baseline` 是当前推荐默认值。

Table-Aware RAG 预留开关：

```env
# Table-Aware RAG feature flags
# First-stage default: disabled, no behavior change.
TABLE_AWARE_INGESTION=false
TABLE_AWARE_RETRIEVAL=off
TABLE_EVIDENCE_TOP_K=20
TABLE_EVIDENCE_FINAL_MAX=4
TABLE_FULL_FETCH_ENABLED=false
ENABLE_FINANCE_FORMULA_EXPANSION=false
```

说明：

- `TABLE_AWARE_INGESTION`
  - 是否在文档入库阶段解析和索引表格证据。
  - 第一阶段安全 commit 默认关闭，不改变现有行为。
- `TABLE_AWARE_RETRIEVAL`
  - `off`：完全关闭 table-aware retrieval
  - `auto`：仅当问题看起来像表格 / 数值 / 财务问题时启用
  - `force`：总是纳入表格证据候选，主要用于调试

当前第一阶段 commit 只新增配置开关，不解析表格，不修改 Milvus schema，也不改变 baseline 检索行为。

### 3. 启动依赖服务

```bash
docker compose up -d
```

会启动：

- PostgreSQL
- Redis
- Milvus standalone
- MinIO
- etcd
- Attu

默认端口：

- PostgreSQL：`5432`
- Redis：`6379`
- Milvus：`19530`
- Milvus health：`9091`
- MinIO API：`9000`
- MinIO Console：`9001`
- Attu：`8080`

### 4. 启动后端

```bash
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

或者：

```bash
python backend/app.py
```

### 5. 打开页面

- 前端页面：[http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- OpenAPI 文档：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## 文档处理与检索说明

### 文档入库

当前入库链路是：

1. 保存上传文件
2. 清理同名旧文档对应的向量和父块数据
3. 解析文档并生成页面内容
4. 执行三级分块
5. 对文本做清洗，移除 NUL 和危险控制字符
6. 父级分块写入 PostgreSQL
7. 页面级聚合内容写入 `document_pages`
8. 叶子分块写入 Milvus
9. 同步更新 BM25 sparse 状态

### 当前默认检索主路径

默认 `baseline` 模式会走：

1. chunk-level hybrid retrieval
2. rerank
3. auto-merge
4. page/chunk 邻域扩展
5. 将最终上下文送入生成模型

### 空知识库保护

当知识库为空时，系统会直接返回：

`知识库当前为空，尚未上传文档，无法基于文档检索回答。`

不会继续进入检索流程。

## 前端功能

当前前端是单页应用，提供这些实际功能：

- 注册 / 登录 / 退出登录
- 会话历史查看与删除
- 流式聊天
- 文档管理页面
- 多文件选择
- 批量上传
- 已上传文档列表刷新
- 单文档删除
- 批量删除
- 上传 / 删除任务进度展示
- 基础 RAG trace 展示

## 主要 API

### 鉴权

- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/me`

### 聊天

- `POST /chat`
- `POST /chat/stream`

### 会话

- `GET /sessions`
- `GET /sessions/{session_id}`
- `DELETE /sessions/{session_id}`

### 文档管理

管理员权限：

- `GET /documents`
- `POST /documents/upload`
- `POST /documents/upload/async`
- `GET /documents/upload/jobs`
- `GET /documents/upload/jobs/{job_id}`
- `DELETE /documents/{filename}`
- `DELETE /documents/delete/async/{filename}`
- `POST /documents/delete/async/batch`
- `GET /documents/delete/jobs/{job_id}`

### 调试

管理员权限：

- `POST /debug/retrieval`

这个接口只跑检索与上下文构造，不调用最终生成模型，适合检查：

- 当前命中了哪些 chunk
- `rag_trace` 返回了什么
- baseline / experimental 模式是否按预期生效

## FinanceBench 相关说明

项目保留了 FinanceBench 评估和实验代码，但当前推荐策略是：

- 主系统默认走 `baseline`
- FinanceBench 专项实验显式切到 `finance_experimental`
- 不要让实验链路影响通用 RAG 主流程

如果你只是要一个稳定的可演示项目，请保持：

```env
RAG_RETRIEVAL_MODE=baseline
```

## 可选脚本

如果你在实验模式下需要重建 page index，可使用：

```bash
python backend/scripts/rebuild_page_index.py
```

这个脚本主要服务于 `finance_experimental`，不是 baseline 模式的必需步骤。

## 当前已保留的重要修复

README 只列当前实际存在且仍在主线上生效的修复：

- NUL / 控制字符清洗
- 文档列表按实际入库结果刷新
- Milvus `ef` 自动调整
- 默认 `baseline` / 可选 `finance_experimental` 模式隔离
- RAG trace 基础字段返回
- LangSmith 评估兼容字段保留

## 当前不在 README 中展开的内容

下面这些内容在仓库中可能有实验代码或残留实现，但不作为当前主系统承诺能力写入 README：

- 视觉检索
- OCR / hi_res 表格系统
- 复杂的 FinanceBench page-level 主链路
- 任何未默认启用的实验 prompt 或实验检索路径

## 常见问题

### 1. 为什么上传后看不到文档？

先确认：

- 上传任务是否完成
- `/documents` 是否能正常返回
- 当前登录角色是否为 `admin`

### 2. 为什么 Milvus 会报 `ef should be larger than k`？

项目已内置自动修正逻辑，但仍建议在 `.env` 中保留：

```env
MILVUS_SEARCH_EF=128
```

### 3. 为什么财务实验结果不稳定？

目前复杂的 FinanceBench page-level 链路已经被隔离到：

```env
RAG_RETRIEVAL_MODE=finance_experimental
```

主系统默认不要启用它。
