# 🧠 Local Skill: Persistent Memory (本地持久化记忆)

`local-skill` 是一个为 AI Agent 设计的本地持久化记忆解决方案。它通过轻量级的 Rust CLI 工具实现对话关键信息的存储、检索与回忆，旨在为 Agent 提供长期记忆能力。

## ✨ 核心特性

- **🚀 高性能后端**: 基于 Rust 编写的 `memstore` 工具，无 Python/Node 依赖，毫秒级响应。
- **🔒 本地隐私**: 所有记忆数据存储在单一文件 (`memory/memories.hnsw`)，完全掌控数据安全。
- **🤝 向量检索**: 使用 **HNSW (hnsw_rs)** 做近似最近邻召回，结合向量相似度、权重 (Weight) 和时间衰减 (Recency) 综合评分。
- **🤖 智能集成**: 支持“记住...”(手动高权重) 和“回忆...”(显式检索) 等多种交互模式。

## 📂 项目结构 (Project Structure)

```
.
├── skills/
│   └── persistent-memory/   # Skill 定义、脚本与配置
│       ├── SKILL.md         # 集成文档、Prompt 模版
│       ├── scripts/         # 运行时脚本目录 (存放编译后的 memstore)
│       └── references/      # 参考文档 (记忆格式规范等)
└── src/
    └── memstore/            # Rust CLI 源码
```

## 🛠️ 快速开始 (Quick Start)

### 1. 编译 Memstore 工具

项目核心依赖 Rust 环境，请先编译 `memstore` 工具：

```bash
cd src/memstore
cargo build --release --offline
```

### 2. 安装/部署

将编译好的二进制文件复制到 Skill 的脚本目录下，以便 Agent 调用：

```bash
# 在项目根目录执行
mkdir -p skills/persistent-memory/scripts
cp src/memstore/target/release/memstore skills/persistent-memory/scripts/
```

## 📖 使用指南 (Usage)

`memstore` CLI 工具支持以下核心命令：

### 添加记忆 (Add)

```bash
# 自动摘要 (权重默认 1.0)
./memstore add --text "用户计划下周启动新项目" --kind summary

# 手动高权重记忆 (权重建议 > 2.0)
./memstore add --text "用户偏好使用暗色主题" --kind profile --weight 3.0
```

### 搜索记忆 (Search)

基于 Query 检索最相关的记忆片段：

```bash
# 检索 Top 3
./memstore search --query "用户有什么偏好" --limit 3
```

### 其他命令

```bash
# 查看最近写入的记忆
./memstore recent --limit 10

# 压缩/清理数据库 (保留最新的 N 条)
./memstore compact --keep 5000
```

## ⚙️ 配置 (Configuration)

可以通过环境变量覆盖默认存储路径：

- `MEMSTORE_PATH`: 记忆数据库文件路径 (默认: `memory/memories.hnsw`)

---

参考文档：

- [Skill 定义与 Prompt (SKILL.md)](skills/persistent-memory/SKILL.md)
- [记忆格式规范 (memory-format.md)](skills/persistent-memory/references/memory-format.md)
