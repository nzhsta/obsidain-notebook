---
last-updated: 2026-04-17
topics:
  - RAG 混合检索策略
---

# RAG

## RAG 混合检索策略

> 来源：2026-04-17 对话归档

**问题/场景：**
用户询问 RAG 检索时如何混合 dense 和 sparse 检索方式。

**核心知识点：**
- Dense retrieval：使用向量嵌入（embedding）进行语义相似度检索，擅长理解语义
- Sparse retrieval：使用传统关键词匹配（如 BM25），擅长精确匹配关键词
- Hybrid retrieval：结合两者优势，提升召回率和准确性
- 常用融合方法：RRF（Reciprocal Rank Fusion）对两种检索结果进行加权排序
- 可调整权重参数 alpha 控制 dense/sparse 的侧重

**代码示例：**
```python
# 伪代码示例
 dense_results = vector_store.similarity_search(query, k=50)
sparse_results = bm25_search(query, k=50)

# RRF 融合
fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60)
```

---
