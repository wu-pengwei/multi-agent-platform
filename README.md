# 多智能体 AI Agent 平台

主 Agent + 动态子 Agent 架构的 AI Agent 平台，围绕四个方向构建：**长期记忆管理**、**多智能体协作**、**工具调用**、**服务化部署**。支持 20+ LLM Provider、MCP 工具接入、15+ 聊天渠道，并通过 OpenAI 兼容 API 对外提供服务。

> 本仓库基于开源项目 [nanobot](https://github.com/HKUDS/nanobot)（MIT License）二次开发。基础 Agent 能力继承自上游，本仓库的重点改进见 [核心改进](#核心改进本仓库新增) 一节，均配有独立实现与测试。

## 总体架构

```
服务层（OpenAI 兼容 API：aiohttp / FastAPI 双实现）
        │
渠道层（Telegram / Discord / 飞书 / 微信 / WebSocket …）
        │
   消息总线（inbound / outbound · asyncio.Queue）
        │
调度层（AgentLoop：会话路由、命令分流、会话锁）
        │
推理层（AgentRunner + Provider 抽象：LLM 推理循环、工具执行）
        │
能力层（ToolRegistry / 动态子 Agent / MCP）
        │
持久层（JSONL 会话 / 记忆文件 / GitStore / 语义向量索引）
```

**为什么选择主 Agent + 子 Agent 架构**：复杂任务由主 Agent 拆解，通过 Spawn 工具动态创建后台子 Agent 并行执行，子 Agent 只携带任务描述、复用同一推理引擎，上下文天然隔离；结果经消息总线回注主 Agent 会话汇总。相比群聊式多 Agent（上下文互相污染、轮次不可控）和静态 Workflow（无法应对计划外分支），这种"编排者 + 执行者"模式在保持简单的同时获得了动态任务分解能力。

## 核心改进（本仓库新增）

### 1. 长期记忆：接通语义召回链路

**问题**：上游的语义记忆层（`SemanticMemory`）只有数据结构定义，从未接入主链路——历史对话写入后不会被向量化，对话时也不会检索，长期记忆实际只剩固定窗口的 "Recent History"。

**改进**：打通"写入 → 索引 → 召回 → 注入"完整路径——

- 每条历史写入 `history.jsonl` 时同步嵌入本地向量索引 `semantic_index.jsonl`；
- 每轮对话以当前用户消息作为查询，召回最相关的历史记忆，注入 system prompt 的 `## Semantic Memories` 段落；
- 与 Recent History 做去重（按 cursor 过滤），避免同一条记忆重复出现在两处；
- 未安装 embedding 依赖或模型加载失败时优雅降级为原有文件记忆，主链路不受影响。

**实现**：`nanobot/agent/memory.py`（写入与召回）、`nanobot/agent/semantic_memory.py`（同步嵌入/查询）、`nanobot/agent/context.py`（prompt 注入）、`nanobot/config/schema.py`（配置）。

### 2. 服务化：新增 FastAPI 服务层

**问题**：上游仅有 aiohttp 一种 API 实现，简历与实际技术栈对不上。

**改进**：新增与 aiohttp 完全同构的 FastAPI 适配层——

- 同样的 `/v1/chat/completions`、`/v1/models`、`/health` 协议；
- 支持 JSON / SSE 流式两种响应、base64 图片输入、session 隔离与并发锁、超时治理；
- CLI 通过 `--framework` 切换，默认仍为 aiohttp，保持向后兼容。

**实现**：`nanobot/api/fastapi_server.py`、`nanobot/cli/commands.py`。

### 3. 评测：24 轮长程记忆评测集

**问题**：长期记忆效果此前没有可复现的度量方式。

**改进**：`evals/` 目录提供 24 轮连续对话评测集与自动判定脚本——

- 前半段埋入项目代号、技术栈、部署环境等关键事实，中间隔足够多轮次超出 Recent History 窗口，末段提问验证召回；
- 每条 case 支持 `expected_in`（必须召回）与 `expected_not_in`（敏感信息不得泄露）双重断言；
- 判定完全确定性（关键词匹配，无 LLM-as-judge 噪声），可对"关闭 / 开启语义召回"两种配置跑对比基线。

**实现**：`evals/cases/memory_cases.json`、`evals/run_memory_eval.py`、`evals/eval_lib.py`。

## 继承的基础能力（来自上游）

- **多智能体协作**：Spawn 工具动态创建子 Agent，错误短路 + 部分进度上报；
- **工具调用治理**：ToolRegistry 统一注册、JSON Schema 校验、json-repair 修复、执行错误文本化回灌模型自愈；
- **记忆分层**：session 短期上下文 / history.jsonl 摘要归档 / MEMORY.md 长期事实 / SOUL.md 人格 / USER.md 用户画像，GitStore 版本化；
- **MCP**：stdio / SSE / Streamable HTTP 三种传输；
- **渠道**：Telegram、Discord、飞书、微信、Slack、WebSocket 等 15+；
- **Provider 抽象**：OpenAI、Anthropic 及大量兼容提供商。

## 快速开始

**1. 安装**（Python ≥ 3.11）

```bash
git clone <your-repo-url>
cd <repo>
pip install -e .

# 按需安装本仓库新增功能的可选依赖
pip install -e ".[fastapi]"    # FastAPI 服务层
pip install -e ".[semantic]"   # 语义记忆召回（sentence-transformers）
```

**2. 初始化与配置**（`~/.nanobot/config.json`）

```bash
nanobot onboard
```

最少配置两处：Provider API key 和模型。

**3. 开启语义记忆召回**（可选，对应改进 1）

```json
{
  "agents": {
    "defaults": {
      "semantic": {
        "enabled": true,
        "embedModel": "all-MiniLM-L6-v2",
        "topK": 5
      }
    }
  }
}
```

**4. 启动**

```bash
nanobot agent                        # CLI 对话
nanobot serve                        # aiohttp API 服务（默认）
nanobot serve --framework fastapi    # FastAPI API 服务
nanobot gateway                      # 全渠道网关
```

**5. 调用 OpenAI 兼容接口**（以 8000 端口为例）

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"agent","messages":[{"role":"user","content":"用一句话介绍你自己"}]}'
```

流式输出加 `"stream": true` 即可（SSE，OpenAI chunk 格式）。

**6. 运行记忆评测**（对应改进 3）

把被测系统的逐轮回复导出为 `responses.jsonl`（每行 `{"turn": 20, "text": "..."}`），然后：

```bash
python -m evals.run_memory_eval --input responses.jsonl
```

输出每轮 PASS/FAIL 与总体准确率，任一 case 失败则以非零码退出，可直接接入 CI。分别在关闭 / 开启 `agents.defaults.semantic.enabled` 两种配置下运行，即可得到语义召回的对比基线。

**7. 运行测试**

```bash
python -m pytest tests/agent/test_semantic_recall.py tests/test_fastapi_server.py tests/test_eval_cases.py -q
```

## 更多文档

- 渠道接入：[docs/chat-apps.md](./docs/chat-apps.md)
- 完整配置：[docs/configuration.md](./docs/configuration.md)
- 部署：[docs/deployment.md](./docs/deployment.md)
- OpenAI 兼容 API：[docs/openai-api.md](./docs/openai-api.md)

## 致谢

本仓库基于 [nanobot](https://github.com/HKUDS/nanobot)（MIT License）二次开发。基础 Agent 循环、渠道插件、MCP 集成、记忆文件体系等核心能力来自上游项目及贡献者。第三方依赖声明见 [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)。

## License

MIT — 见 [LICENSE](./LICENSE)。本项目继承上游许可证与版权声明。
