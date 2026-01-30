---
name: persistent-memory
description: 持久化对话记忆与外挂大脑。用于：自动写入对话摘要、手动高权重“记住/配置/坑点”、本地文件持久化、检索最近相关记忆并注入 System Prompt、以及“回忆一下…”的显式全量搜索与输出。适用于需要长期用户偏好、项目状态、问题复盘的聊天或代理流程。
---

# 持久化记忆（Rust 本地存储）

## 目标
- 把每轮对话的关键结论写入本地记忆（自动）。
- 支持“记住这个…”的手动高权重写入。
- 用户提问时自动检索 3 条相关记忆并注入 System Prompt。
- 用户显式“回忆一下…”时做全量检索并输出详细结果。
- 使用 HNSW 向量索引进行相似度检索（查询时内存构建）。

## 资源
- Rust CLI（二进制）：`scripts/memstore`（本地文件存储 + 检索）
- Rust 源码：`/Users/c.chen/dev/local-skill/src/memstore`
- 记录格式参考：`references/memory-format.md`

## 初始化
1) 在 `src/memstore` 构建 CLI：
   - `cargo build --release`
2) 拷贝二进制到 Skill 目录：
   - `cp /Users/c.chen/dev/local-skill/src/memstore/target/release/memstore /Users/c.chen/dev/local-skill/skills/persistent-memory/scripts/memstore`
3) 运行时使用 Skill 内二进制：
   - `/Users/c.chen/dev/local-skill/skills/persistent-memory/scripts/memstore`
3) 默认存储路径：`memory/memories.hnsw`（可用 `MEMSTORE_PATH` 或 `--path` 覆盖）

## 写入流程
### 自动摘要（每轮对话结束）
- 用轻量模型或 API 生成一句话摘要（聚焦结论/决定/新信息）。
- 调用：
  - `memstore add --kind summary --weight 1.0 --text "<一句话摘要>"`

### 手动记忆（用户说“记住…”）
- 提取用户原句（或略作结构化）。
- 赋予更高权重：
  - `memstore add --kind profile --weight 3.0 --text "<用户偏好>"`
  - `memstore add --kind state --weight 2.5 --text "<项目状态>"`
  - `memstore add --kind manual --weight 5.0 --text "<坑点/配置>"`

## 读取与注入
### 自动注入（每次用户提问）
1) 用当前用户问题作为检索 query：
   - `memstore search --query "<用户问题>" --limit 3`
2) 将返回的 3 条记忆拼接到 System Prompt（或“额外上下文”区块）。
3) 明确标注为“历史记忆”，避免与当前指令混淆。

### 显式回忆（用户说“回忆一下…”）
- 进行更高数量的检索并返回详细列表：
  - `memstore search --query "<问题>" --limit 20`
- 如果结果过多，先做一次摘要再答复用户。

## 记录格式与评分
- 详见 `references/memory-format.md`。
- 默认评分为“向量相似度 + 权重 + 新近度”。
- 如需更强召回，可替换为更强向量模型或 BM25/TF-IDF。

## 常见触发词
- “记住这个配置/坑/偏好/结论” → 手动高权重写入
- “回忆一下/之前怎么做的/我上次如何处理” → 显式回忆
- 普通问题 → 自动注入相关记忆
