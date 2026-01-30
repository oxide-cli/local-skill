# Memory Store 格式

## 单文件存储
默认数据库文件：`memory/memories.hnsw`（可用 `MEMSTORE_PATH` 或 `--path` 覆盖）。

该文件为二进制格式（`bincode` 序列化），内容结构如下：

```
Store {
  version: u32,
  vector_dim: usize,
  records: Vec<Record>
}

Record {
  id: u128,
  ts: i64,
  kind: String,
  weight: f32,
  text: String,
  vector: Vec<f32>
}
```

## 近似检索索引（HNSW）
- 使用 `hnsw_rs` 在查询时构建 HNSW 索引（内存中）。
- 索引本身不落盘，向量随记录持久化在同一 `.hnsw` 文件中。

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
