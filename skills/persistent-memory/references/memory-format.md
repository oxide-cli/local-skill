# Memory Store 格式

## 记录字段
每行一条记录，字段用 `|` 分隔，顺序如下：

```
id|ts|kind|weight|text
```

- `id`: 毫秒级时间戳（u128）
- `ts`: 秒级时间戳（i64）
- `kind`: 分类（profile/state/summary/issue/solution 等）
- `weight`: 权重（高权重用于“手动记忆”）
- `text`: 纯文本内容（可包含换行）

## 转义规则
为了避免分隔符冲突，写入时进行转义：

- `\\` → `\\\\`
- `\n` → `\\n`
- `|` → `\\|`

读取时反向还原。

## 向量索引文件
默认向量索引文件：`memory/memories.vec`（可用 `MEMSTORE_VEC_PATH` 或 `--vec-path` 覆盖）。

每行一条向量，字段用 `|` 分隔，顺序如下：

```
id|dim|v1,v2,...,vN
```

- `id`: 对应 `memories.log` 的记录 ID
- `dim`: 向量维度（默认 256）
- `vN`: 归一化后的浮点向量分量

## 近似检索索引（HNSW）
- 使用 `hnsw_rs` 在查询时构建 HNSW 索引（内存中），用于近似最近邻检索。
- 当前实现不单独持久化 HNSW 结构，仅持久化向量文件。

## 向量生成（默认实现）
- 使用 token 哈希到固定维度（feature hashing）。
- 词频累加后做 L2 归一化。
- 检索使用余弦相似度。

## 检索打分（默认实现）
`memstore search` 的默认打分由三部分组成：

- 余弦相似度（向量检索）
- `weight * 0.5`
- 新近度：`1 / (1 + age_days)`

你可以按需求替换为更强的向量模型或 BM25/TF-IDF。
