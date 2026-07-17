# TraceMind：基于多模态 RAG 的可信产品客服 Agent

TraceMind 面向产品手册问答与复杂客服场景，核心目标不是做一个简单 FAQ 检索器，而是构建一条更接近真实客服工作的问答链路：先理解用户问题，再识别产品范围，再做基于证据的混合检索，最后生成尽量可追溯、可解释的答案。

当前仓库已经整理为一套本地可直接跑通的工程版本，支持：

- 产品类问题路由与手册识别
- 基于 Milvus 的向量检索 + BM25 混合检索
- 多轮场景下的澄清式回复
- FastAPI 服务与前端调试页
- 使用国内模型配置运行

## 当前状态

已核对当前知识库覆盖情况：

- 预期手册数：`40`
- 已入库手册 source 数：`40`
- 缺失：`0`
- 额外 source：`0`

也就是说，按当前 `processed_data` 与 `catalog` 计算，知识库覆盖没有遗漏。

## 项目结构

```text
TraceMind/
├─ README.md
├─ interface.py                  # 服务启动入口
├─ pipeline.py                   # 兼容导出入口
├─ tracemind/                    # 核心在线链路
│  ├─ api.py
│  ├─ pipeline.py
│  ├─ clarifier.py
│  ├─ query_classification.py
│  ├─ retriever.py
│  ├─ answer_general_query.py
│  ├─ answer_product_query.py
│  ├─ model_factory.py
│  ├─ config.py
│  └─ utils.py
├─ scripts/                      # 离线处理与建库脚本
│  ├─ build_kb.py                # 当前推荐的建库脚本
│  ├─ preprocess.py
│  ├─ chunk.py
│  ├─ generate_handbook_name.py
│  ├─ generate_catalog.py
│  └─ submit.py
├─ assets/
├─ catalog/
├─ data/
├─ processed_data/
├─ artifacts/
├─ configs/
├─ tools/
├─ milvus-backup-files/
└─ milvus-docker-compose.yml
```

## 环境准备

### 1. 安装依赖

推荐使用 `uv`，也可以继续使用现有 `.venv`。

```powershell
uv sync -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果已经有虚拟环境，也可以直接：

```powershell
.\.venv\Scripts\activate
pip install -e .
```

### 2. 配置 `.env`

先复制模板：

```powershell
Copy-Item .env.example .env
```

当前推荐至少配置这些字段：

```env
CHAT_BASE_URL=
CHAT_API_KEY=
CHAT_MODEL=

EMBEDDING_BASE_URL=
EMBEDDING_API_KEY=
EMBEDDING_MODEL=

MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
MILVUS_DB_NAME=default

USE_CONTEXTUAL_AUGMENTATION=1
USE_QUERY_CLS=1
```

当前工程已经适配国内模型方案。一个常见配置是：

```env
CHAT_BASE_URL="https://api.deepseek.com"
CHAT_API_KEY="你的 DeepSeek Key"
CHAT_MODEL="deepseek-v4-flash"

EMBEDDING_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_API_KEY="你的 DashScope Key"
EMBEDDING_MODEL="text-embedding-v3"
```

## 启动 Milvus

在项目根目录运行：

```powershell
docker compose -f milvus-docker-compose.yml up -d
```

检查容器：

```powershell
docker ps
```

正常情况下会看到：

- `milvus-standalone`
- `milvus-etcd`
- `milvus-minio`

## 当前推荐的建库流程

### 方案 A：直接使用仓库内现成的 `processed_data` 建库

这是当前最推荐、最稳定的方式。

```powershell
.\.venv\Scripts\python -m scripts.build_kb
```

说明：

- 该脚本直接消费 `processed_data/` 中已经整理好的手册内容
- 会把 Markdown 切块后写入 Milvus
- 当前版本优先保证“可跑通、可检索、可问答”
- 即使视觉描述链路不稳定，也能先把文字知识库建起来

如果只想调试某一本手册，可以：

```powershell
$env:MANUAL_FILTER="吹风机"
.\.venv\Scripts\python -m scripts.build_kb
Remove-Item Env:MANUAL_FILTER
```

### 方案 B：从更早的离线流程全量重建

如果你想完整复现原始比赛风格流程，可以继续使用这些历史脚本：

1. `python -m scripts.preprocess`
2. `python -m scripts.generate_handbook_name`
3. `python -m scripts.generate_catalog`
4. `python -m scripts.chunk`

但要注意：

- 这条链路对模型输出格式更敏感
- 原版脚本更依赖多模态描述质量
- 当前仓库的“真实可跑主线”已经切到 `scripts.build_kb.py`

## 启动服务

```powershell
.\.venv\Scripts\python interface.py
```

启动后可访问：

- API 文档：`http://127.0.0.1:8000/scalar`
- 调试页：`http://127.0.0.1:8000/playground`

## 快速验证

推荐先用这类问题验证知识库是否命中：

```text
使用吹风机时，人员需要佩戴哪些防护装备？
```

如果运行正常，你会看到：

- 服务返回 `200`
- 问题被路由到产品问答链路
- 日志中出现 `retriever:done hits=...`
- 返回内容会引用到吹风机手册中的“个人防护装备”段落

## 观察现象的方法

当前项目已经补了关键日志，建议重点看：

- `product:start`
- `retriever:start`
- `retriever:done hits=`
- `product:first_pass`
- `product:fallback`
- `product:final`

经验上可以这样判断：

- `hits=0`：优先检查建库、source、language、collection
- `hits>0` 但答得泛：优先检查 prompt、上下文拼接、精修链路
- 返回 500：优先看最后一层异常栈，通常是格式解析或图片占位问题

## 兼容说明

当前仓库已经做过以下适配：

- 将在线模型调用统一收敛到 `tracemind/model_factory.py`
- 支持 DeepSeek + DashScope 的组合配置
- 放宽了部分答案解析逻辑，减少因为 LLM 格式不稳定导致的 500
- 修正了 Windows 下路径与空环境变量覆盖的问题

## 后续建议

如果要继续往“更像真实客服系统”的方向推进，优先级建议是：

1. 把 `scripts.build_kb.py` 升级为正式离线建库主链路
2. 把图片占位描述升级成真正的视觉理解结果
3. 增加知识库健康检查脚本与 collection 统计接口
4. 把多轮澄清、降级、失败兜底做成更稳定的线上治理层
