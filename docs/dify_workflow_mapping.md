# 三语智能客服 Agent ↔ Dify 工作流映射

> 本项目的 7 节点工作流（`app/orchestrator.py`）是按低代码平台的节点模型显式拆分的——每个节点对应 Dify Chatflow 中的一个原子节点，迁移成本主要在配置而非重构。本文给出逐节点映射、模型配置建议与「代码版 vs 低代码版」的取舍说明。

## 1. 节点映射总览

| # | 本项目节点 | 代码位置 | Dify 对应节点 | 迁移要点 |
|---|---|---|---|---|
| 1 | `language_detect` | `app/language.py` | **Code 节点**（或 LLM 节点 + 分类器） | 粤语标记字 + 繁体检测是纯规则，直接搬进 Code 节点；归一化逻辑（粤语词→普通话→繁转简）必须保持「先粤语后繁简」的顺序 |
| 2 | `intent_route` | `app/intent.py` | **问题分类器**（Question Classifier） | 7 类意图 = 7 个分类标签；把加权正则规则写成分类描述；安全意图（转人工/工单）在 Dify 中同样应配置为高优先分支 |
| 3 | `slot_filling` | `orchestrator._flow_enterprise/_flow_ticket` | **会话变量 + 条件分支（IF/ELSE）** | 留资/工单槽位状态存 Dify 会话变量（`ent_ask_contact`、`ticket_contact`）；每轮先判会话变量再走路由，等价于本项目的「状态覆写」 |
| 4 | `tool_exec` | `app/tools.py` | **自定义工具（HTTP/代码）** | 8 个工具包成 API 或插件；资费查询、工单落库、商机登记为只读/写两类，Dify 侧按工具描述控制模型可调用面 |
| 5 | `compose` | `orchestrator.compose_with_llm` + `app/prompts.py` | **LLM 节点** | 生成 Prompt 模板直接复用 `prompts.py` 结构：系统规则 + 工具结果 + 用户问题；变量注入工具输出文本 |
| 6 | `guardrail` | `app/guardrails.py` | **Code 节点（后处理）** | `mask_pii` / `validate_amounts` / `ensure_disclaimer` 三个函数逐个搬入 Code 节点；金额越界回退模板的逻辑用条件分支实现 |
| 7 | `reply` | `app/server.py` ChatReply | **直接回复节点** | 输出正文 + 来源标注；trace 用 Dify 执行日志替代 |

## 2. Chatflow 拓扑建议

```
开始
 │
 ▼
[Code] 语言识别 + 归一化 ──▶ 输出 {lang, norm_text}
 │
 ▼
[会话变量检查] ent_ask_contact / ticket_contact 非空？
 │ 是 ──▶ 跳过分类，进入对应多轮分支（等价「状态覆写」）
 │ 否
 ▼
[问题分类器] 7 类意图（转人工/工单/漫游/推荐/企业/知识/闲聊）
 │
 ├─ 转人工 ──▶ [Tool] transfer_human ──▶ [直接回复] 排队编号
 ├─ 投诉工单 ──▶ [IF] 有联系方式？──是▶ [Tool] create_ticket ──▶ 回执
 │                                    └─否▶ [直接回复] 追问联系方式
 ├─ 漫游咨询 ──▶ [IF] 含目的地？──是▶ [Tool] query_roaming_packs ──▶ 话术
 │                                 └─否▶ [IF] 开通/充值类？──是▶ [知识检索] ──▶ 抽取回答
 ├─ 套餐推荐 ──▶ [Tool] recommend_plan ──▶ [LLM] 话术
 ├─ 企业方案 ──▶ [Tool] search_solutions ──▶ 方案 ──▶ [IF] 留资？──▶ [Tool] create_lead
 ├─ 知识问答 ──▶ [知识检索] BM25/向量 ──▶ [LLM] 抽取式+来源
 └─ 闲聊 ──▶ [直接回复] 能力引导（三语模板）
 │
 ▼
[Code] 护栏：PII 脱敏 → 金额溯源（越界→回退模板）→ 免责声明
 │
 ▼
[直接回复] {text, citations, lang}
```

## 3. 模型与知识库配置

| 配置项 | 建议 | 对应本项目 |
|---|---|---|
| 分类器/话术模型 | 任意 OpenAI 兼容模型（DeepSeek/Qwen 等国产模型均可） | `.env` 中 `APP_LLM_BASE_URL/MODEL` 即为此预留 |
| 温度 | 话术 0.2–0.3（资费场景压低创造性） | prompts.py 已按此基调编写 |
| 知识库 | 6 篇 `data/kb/*.md` 直接上传；分段用「标题分段」，长度 ≈400 字 | rag.py 的结构化切片规则 |
| 检索模式 | 混合检索 + Rerank 可选；小语料下 BM25 权重可调高 | 本项目纯 BM25（Hit@1 94.4%）为基线 |
| 金额防幻觉 | LLM 节点后必须接 Code 节点做溯源校验（Dify 无内置金额校验） | validate_amounts 整体搬入 |

## 4. 代码版 vs 低代码版：取舍说明

| 维度 | 代码版（本项目） | 低代码版（Dify Chatflow） |
|---|---|---|
| 迭代主体 | 开发者 | 运营/业务可自行调整话术与流程 |
| 离线评测 | 72 测试 + evals 全离线可复现 | 依赖平台执行日志，自动化评测需 API 驱动 |
| 护栏确定性 | 代码级强制（金额越界必回退） | 依赖节点编排正确性，需评审流程图 |
| 适用阶段 | 高危写操作（工单/留资）先行验证 | 流程稳定后开放给运营迭代话术 |

**推荐路径**：先以代码版验证流程与护栏（当前状态），流程稳定后把「话术微调、快捷回复、知识库更新」迁到 Dify 由运营维护；金额溯源、PII 脱敏等安全节点无论在哪个平台都必须保留为代码级校验。

## 5. 迁移清单（Checklist）

- [ ] 8 个工具包装为 HTTP 服务或 Dify 插件（保持 `TOOLS_SCHEMA` 参数定义）
- [ ] 7 类意图写为问题分类器标签，规则描述从 `intent.py` 的正则表转译
- [ ] 会话变量建 `ent_*` / `ticket_*` 两组，首轮清空逻辑对齐 `_flow_*` 重置时机
- [ ] `language.py` 归一化函数整体放入 Code 节点（顺序：粤语词 → 粤语字 → 繁转简）
- [ ] `guardrails.py` 三个函数放入后处理 Code 节点，回退分支用模板文本兜底
- [ ] 知识库 6 篇上传并配置标题分段；检索失败分支引导客服热线或转人工
- [ ] 三语回复模板（orchestrator 各 flow 内联字典）转译为 Dify 模板变量
