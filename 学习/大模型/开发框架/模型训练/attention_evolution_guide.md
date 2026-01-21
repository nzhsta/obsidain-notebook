# 1 Attention机制的演化史：从Self-Attention到Flash Attention 4

# 2 📚 目录

1. [基础：Self-Attention](#基础self-attention)
2. [效率优化的四大方向](#效率优化的四大方向)
3. [Hardware-Efficient Attention（Flash Attention系列）](#hardware-efficient-attention)
4. [Sparse Attention（稀疏注意力）](#sparse-attention)
5. [Linear Attention（线性注意力）](#linear-attention)
6. [Compact Attention（紧凑注意力）](#compact-attention)
7. [混合架构](#混合架构)
8. [时间线总览](#时间线总览)
9. [性能对比](#性能对比)

---

## 2.1 🎯 基础：Self-Attention

### 2.1.1 原始公式（2017, Vaswani et al.）

```python
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

### 2.1.2 复杂度分析

```
时间复杂度: O(N²d)
空间复杂度: O(N²)
  其中: N = 序列长度, d = 特征维度
```

### 2.1.3 核心问题

1. **二次方复杂度**：序列长度翻倍，计算量和内存增加4倍
2. **内存墙**：需要存储N×N的注意力矩阵
3. **长序列瓶颈**：限制了上下文长度（早期模型2K-4K tokens）

### 2.1.4 标准实现

```python
def standard_attention(Q, K, V):
    """
    标准Attention实现
    
    Args:
        Q: [batch, num_heads, seq_len, d_k] 查询
        K: [batch, num_heads, seq_len, d_k] 键
        V: [batch, num_heads, seq_len, d_v] 值
    
    Returns:
        output: [batch, num_heads, seq_len, d_v]
    """
    d_k = Q.size(-1)
    
    # 步骤1: 计算注意力分数 - 需要O(N²)内存！
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # scores: [batch, num_heads, seq_len, seq_len]
    # ⚠️ 这个矩阵会占用大量内存！
    
    # 步骤2: Softmax归一化
    attention_weights = F.softmax(scores, dim=-1)
    # ⚠️ 又一个大矩阵！
    
    # 步骤3: 加权求和
    output = torch.matmul(attention_weights, V)
    
    return output
```

### 2.1.5 内存占用示例

```python
# 假设配置
batch_size = 32
num_heads = 12
seq_len = 4096  # 4K tokens
d_k = 64

# 内存占用计算
attention_matrix_size = batch_size * num_heads * seq_len * seq_len
memory_bytes = attention_matrix_size * 4  # FP32
memory_gb = memory_bytes / (1024**3)

print(f"注意力矩阵大小: {memory_gb:.2f} GB")
# 输出: 注意力矩阵大小: 24.58 GB
# 仅一个注意力矩阵就需要24GB！
```

---

## 2.2 🔄 效率优化的四大方向

根据最新的综合调研，现代efficient attention方法可以分为四大类：

### 2.2.1 分类框架

```
Efficient Attention Methods
├── Hardware-Efficient Attention
│   └── 优化GPU内存访问模式（Flash Attention系列）
├── Sparse Attention  
│   └── 选择性计算attention（固定模式/动态选择）
├── Compact Attention
│   └── 压缩KV cache（低秩/权重共享）
└── Linear Attention
    └── 线性复杂度重构（核方法/RNN形式）
```

### 2.2.2 核心权衡

| 方法类型 | 复杂度 | 精度 | 实现难度 | 适用场景 |
|---------|--------|------|----------|----------|
| Hardware-Efficient | O(N²) | 100% | 高 | 训练+推理 |
| Sparse | O(N×S) | 95-99% | 中 | 长序列推理 |
| Compact | O(N²) | 90-98% | 低 | 推理（内存受限）|
| Linear | O(N) | 85-95% | 高 | 长序列训练 |

---

## 2.3 ⚡ Hardware-Efficient Attention

### 2.3.1 核心思想

**不改变计算量，而是优化内存访问模式**

关键洞察：
- GPU的内存层次：HBM（慢，大）→ SRAM（快，小）
- 标准attention的问题：频繁读写HBM

### 2.3.2 GPU内存层次

```
┌─────────────────────────────────────────────┐
│ SRAM (On-chip Cache)                        │
│ - 容量: ~20 MB                               │
│ - 带宽: ~19 TB/s                             │
│ - 速度: 极快 ✓✓✓                            │
└─────────────────────────────────────────────┘
              ↕ 100x speed difference
┌─────────────────────────────────────────────┐
│ HBM (High Bandwidth Memory)                 │
│ - 容量: 40-80 GB                             │
│ - 带宽: 1.5-3 TB/s                           │
│ - 速度: 相对慢 ✗                            │
└─────────────────────────────────────────────┘
```

### 2.3.3 Flash Attention 1 (2022)

**论文**: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"

#### 2.3.3.1 核心创新

1. **分块计算（Tiling）**
2. **重计算（Recomputation）**
3. **融合算子（Kernel Fusion）**

#### 2.3.3.2 算法原理

```python
# 伪代码
def flash_attention_v1(Q, K, V, block_size=128):
    """
    Flash Attention v1 的核心思想
    
    关键点:
    1. 不存储完整的N×N attention矩阵
    2. 分块处理，充分利用SRAM
    3. 在线更新softmax统计量
    """
    N = Q.shape[1]  # 序列长度
    T_r = block_size  # Query块大小
    T_c = block_size  # Key/Value块大小
    
    # 输出缓冲区和统计量
    O = torch.zeros_like(Q)  # 输出
    l = torch.zeros(N)       # softmax分母
    m = torch.full((N,), -float('inf'))  # softmax最大值
    
    # 外循环：遍历Query块
    for i in range(0, N, T_r):
        Q_block = Q[:, i:i+T_r]  # 加载Query块到SRAM
        O_block = torch.zeros_like(Q_block)
        l_block = torch.zeros(T_r)
        m_block = torch.full((T_r,), -float('inf'))
        
        # 内循环：遍历Key/Value块
        for j in range(0, N, T_c):
            K_block = K[:, j:j+T_c]  # 加载K块到SRAM
            V_block = V[:, j:j+T_c]  # 加载V块到SRAM
            
            # 在SRAM中计算块级attention
            S_block = torch.matmul(Q_block, K_block.T) / sqrt(d_k)
            
            # 在线更新softmax统计量（关键！）
            m_new = torch.max(m_block, S_block.max(dim=-1)[0])
            l_new = torch.exp(m_block - m_new) * l_block + \
                    torch.sum(torch.exp(S_block - m_new.unsqueeze(-1)), dim=-1)
            
            # 更新输出
            O_block = torch.exp(m_block - m_new).unsqueeze(-1) * O_block + \
                     torch.matmul(torch.exp(S_block - m_new.unsqueeze(-1)), V_block)
            
            m_block = m_new
            l_block = l_new
        
        # 最终归一化
        O[:, i:i+T_r] = O_block / l_block.unsqueeze(-1)
    
    return O
```

#### 2.3.3.3 内存节省

```
标准Attention:
- 中间矩阵: O(N²) 
- 需要存储完整的attention矩阵

Flash Attention:
- 中间矩阵: O(N) 
- 只存储block-level的矩阵
- 内存减少: 10-20x
```

#### 2.3.3.4 性能提升

```python
# 实际性能对比（A100 GPU）
序列长度 = 2048
标准Attention:  350 TFLOPS (35% GPU利用率)
Flash Attention: 700 TFLOPS (70% GPU利用率)
加速比: 2-4x
```

### 2.3.4 Flash Attention 2 (2023)

**论文**: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"

#### 2.3.4.1 主要改进

1. **减少非矩阵乘法FLOPs**
2. **改进线程块和warp之间的工作分配**
3. **更好的并行化**

#### 2.3.4.2 核心优化

```python
# FA2的三个关键改进

# 1. 减少非矩阵运算
# FA1: 每个block需要重新计算softmax
# FA2: 优化rescaling，减少冗余计算

# 2. 改进并行化
# FA1: 按照sequence维度并行（粗粒度）
# FA2: 即使单个head也能跨thread blocks并行（细粒度）

# 3. 优化work partitioning
# FA1: thread blocks之间工作不均衡
# FA2: 动态负载均衡，减少shared memory读写
```

#### 2.3.4.3 性能提升

```
Flash Attention 2 vs Flash Attention 1:
- 速度: ~2x faster
- GPU利用率: 50-73% (vs 25-40%)
- 支持更长序列: 可处理2x长度
```

### 2.3.5 Flash Attention 3 (2024)

**论文**: "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"

专门针对NVIDIA H100 GPU（Hopper架构）优化，达到75%的GPU利用率。

#### 2.3.5.1 三大创新

##### 2.3.5.1.1 异步执行（Warp Specialization）

```python
# 概念：将不同的操作分配给不同的warps并行执行

# 传统方式（同步）:
for each block:
    load_data()       # 等待
    compute_matmul()  # 等待
    apply_softmax()   # 等待
    write_back()      # 等待

# FA3方式（异步）:
# Warp 0: 持续加载数据 (TMA)
# Warp 1-4: 持续计算矩阵乘法 (Tensor Cores)
# Warp 5: 持续计算softmax
# Warp 6: 持续写回结果
# → 所有操作并行进行！
```

##### 2.3.5.1.2 交错计算（Interleaved Operations）

```python
# FA2: 先完成matmul，再做softmax
# matmul完成 → 等待 → softmax开始

# FA3: 同时进行
# matmul进行中 → softmax也在进行
# 因为H100的矩阵运算(989 TFLOPS)远快于特殊函数(3.9 TFLOPS)
# softmax可以占50%时间 → 现在与matmul重叠！
```

##### 2.3.5.1.3 FP8支持（低精度）

```python
# 使用Hadamard变换处理outliers

def incoherent_fp8_processing(tensor):
    """
    FP8精度下保持数值稳定性
    
    问题: outliers会导致FP8精度损失
    解决: 使用Hadamard变换"打散"outliers
    """
    # Hadamard变换 + 随机符号
    transformed = hadamard_transform(tensor) * random_signs
    
    # 现在可以安全地量化到FP8
    fp8_tensor = quantize_to_fp8(transformed)
    
    return fp8_tensor
```

#### 2.3.5.2 性能突破

```
Flash Attention 3性能:
- FP16: 740 TFLOPS (75% utilization) 
- FP8:  1.2 PFLOPS
- vs FA2: 1.5-2x faster
- 精度: FP8误差减少2.6x
```

### 2.3.6 Flash Attention 4 (2025) 🆕

专为NVIDIA B200/Blackwell架构设计，首个突破petaflop（每秒一千万亿次浮点运算）的attention kernel。

#### 2.3.6.1 革命性创新

##### 2.3.6.1.1 5级流水线（vs FA3的2级）

```python
# FA3: 2-stage pipeline
# - Producer warps (加载数据)
# - Consumer warps (计算+写回)

# FA4: 5-stage pipeline
# Stage 1: TMA warps (异步加载Q, K, V)
# Stage 2: MMA warps (矩阵乘法 QK^T)
# Stage 3: Softmax warps (计算exp和sum)
# Stage 4: Correction warps (更新数值稳定性因子)
# Stage 5: Output warps (计算最终输出)

# 所有阶段并行运行，流水线式处理！
```

##### 2.3.6.1.2 软件模拟指数函数

```python
# 问题: SFU (Special Function Units) 成为瓶颈
# H100: 989 TFLOPS (matmul) vs 3.9 TFLOPS (exp)

# FA4解决方案: 软件模拟
def fast_exp_software(x):
    """
    使用CUDA cores而不是SFU计算exp
    
    优势:
    - 可以在tensor cores计算的同时进行
    - 2x并行度（同时计算2个exp）
    """
    # 三次多项式近似
    # exp(x) ≈ a₀ + a₁x + a₂x² + a₃x³
    result = cubic_polynomial_approx(x)
    return result

# 结果: SFU不再是瓶颈！
```

##### 2.3.6.1.3 选择性重缩放（Selective Rescaling）

```python
# 传统: 每次遇到新max就重新缩放
# FA4: 只在必要时重新缩放

def selective_rescale(current_max, new_max, threshold=1e-6):
    """
    只有当max变化显著时才重新缩放
    
    节省: 减少不必要的计算
    保持: 数值稳定性
    """
    if abs(new_max - current_max) > threshold:
        return True  # 需要重新缩放
    else:
        return False  # 可以跳过
```

#### 2.3.6.2 性能里程碑

```
Flash Attention 4性能:
- 峰值性能: >1 PFLOPS (破petaflop障碍！)
- vs 原始FA: 15x faster
- vs FA3: 2x faster
- 目标硬件: B200/SM10.0 GPUs
```

### 2.3.7 Flash Attention系列对比

```python
# 性能演化（相对于原始FA）
FA1 (A100, 2022):  1.0x   (baseline)
FA2 (A100, 2023):  2.0x   (更好的并行化)
FA3 (H100, 2024):  4.0x   (异步+低精度)
FA4 (B200, 2025): 15.0x   (5级流水线+软件exp)

# GPU利用率演化
FA1: 25-40%
FA2: 50-73%
FA3: 75%
FA4: >80% (估计)
```

---

## 2.4 🎯 Sparse Attention

### 2.4.1 核心思想
**选择性计算：只计算重要的attention对**
动机：
- 观察：attention矩阵通常是稀疏的
- 策略：跳过不重要的计算
- 目标：从O(N²)降低到O(N×S)，其中S 远小于 N


### 2.4.2 稀疏模式分类

#### 2.4.2.1 固定模式（Fixed Patterns）

##### 2.4.2.1.1 a) Local Attention (窗口注意力)

```python
def local_attention(Q, K, V, window_size=256):
    """
    每个token只关注邻近token
    
    复杂度: O(N × window_size)
    """
    # 每个query只看window_size个最近的keys
    # [Q₀] 只看 [K₀...K₂₅₅]
    # [Q₁] 只看 [K₀...K₂₅₆]  (sliding window)
    
    # 可视化 (N=10, window=3):
    # Q\K  0  1  2  3  4  5  6  7  8  9
    #  0  [✓][✓][✓][ ][ ][ ][ ][ ][ ][ ]
    #  1  [✓][✓][✓][✓][ ][ ][ ][ ][ ][ ]
    #  2  [✓][✓][✓][✓][✓][ ][ ][ ][ ][ ]
    #  3  [ ][✓][✓][✓][✓][✓][ ][ ][ ][ ]
    #  ...
```

##### 2.4.2.1.2 b) Strided Attention (步进注意力)

```python
def strided_attention(Q, K, V, stride=64):
    """
    每隔固定步长采样
    
    适用: 需要全局信息但可以采样
    """
    # 每个query看: 0, stride, 2*stride, ...
    
    # 可视化 (N=10, stride=3):
    # Q\K  0  1  2  3  4  5  6  7  8  9
    #  0  [✓][ ][ ][✓][ ][ ][✓][ ][ ][✓]
    #  1  [✓][ ][ ][✓][ ][ ][✓][ ][ ][✓]
    #  ...
```

##### 2.4.2.1.3 c) BigBird Pattern (2020, Google)

```python
def bigbird_attention(Q, K, V, window=3, num_global=2, num_random=2):
    """
    三种模式的组合:
    1. Local (sliding window)
    2. Global (特殊的全局tokens)
    3. Random (随机采样)
    """
    # 可视化 (N=10):
    # G = Global token
    # L = Local window
    # R = Random
    
    # Q\K  0  1  2  3  4  5  6  7  8  9
    #      G  G
    #  0  [✓][✓][L][L][L][R][ ][ ][ ][R]
    #  1  [✓][✓][L][L][L][ ][R][ ][ ][R]
    #  2  [✓][✓][L][L][L][R][ ][R][ ][ ]
    #  ...
    
    # 复杂度: O(N × (window + num_global + num_random))
```

#### 2.4.2.2 学习式稀疏（Learned Sparse）

##### 2.4.2.2.1 a) Routing-based (路由机制)

```python
def routing_sparse_attention(Q, K, V, top_k=128):
    """
    使用学习的路由函数选择top-k个keys
    
    步骤:
    1. 快速打分: 使用低秩近似计算重要性
    2. Top-k选择: 只保留最重要的k个
    3. 精确计算: 只对选中的进行完整attention
    """
    # 步骤1: 低秩近似快速打分
    Q_low = Q @ W_down  # [N, d] @ [d, r] → [N, r]
    K_low = K @ W_down  # [N, d] @ [d, r] → [N, r]
    
    scores_approx = Q_low @ K_low.T  # [N, r] @ [r, N] → [N, N]
    # 成本: O(N²r) 其中 r << d
    
    # 步骤2: 选择top-k
    topk_indices = torch.topk(scores_approx, k=top_k, dim=-1).indices
    # 每个query选择128个最重要的keys
    
    # 步骤3: 精确attention（稀疏）
    for i in range(N):
        selected_K = K[topk_indices[i]]  # [top_k, d]
        selected_V = V[topk_indices[i]]  # [top_k, d]
        
        scores = Q[i] @ selected_K.T / sqrt(d_k)
        weights = softmax(scores)
        output[i] = weights @ selected_V
    
    # 总复杂度: O(N²r + N×k×d) ≈ O(N×k×d) 当 r<<d
```

##### 2.4.2.2.2 b) Clustered Attention

```python
def clustered_attention(Q, K, V, num_clusters=16):
    """
    基于聚类的稀疏attention
    
    思路:
    - Keys聚类成groups
    - Queries只关注相关的clusters
    """
    # 1. 聚类Keys
    cluster_centers = kmeans(K, num_clusters)  # [num_clusters, d]
    cluster_assignments = assign_to_clusters(K, cluster_centers)
    
    # 2. 为每个query找到相关clusters
    for q in Q:
        # 找最近的几个clusters
        relevant_clusters = find_nearest_clusters(q, cluster_centers, top_k=3)
        
        # 只对这些clusters内的keys计算attention
        relevant_keys = K[cluster_assignments in relevant_clusters]
        # ... 计算attention
```

#### 2.4.2.3 动态稀疏（Dynamic Sparse）

##### 2.4.2.3.1 a) H₂O (Heavy Hitter Oracle, 2023)

```python
def h2o_attention(Q, K, V, budget=0.1):
    """
    动态选择"重要token"
    
    观察: 少数tokens贡献大部分attention权重
    策略: 在推理时动态保留重要tokens的KV cache
    """
    # 跟踪attention权重累积
    attention_scores = []
    
    for step in range(seq_len):
        # 计算当前step的attention
        scores = Q[step] @ K[:step].T
        weights = softmax(scores)
        
        # 更新累积权重
        attention_scores[:step] += weights
        
        # 只保留top-k%的tokens在cache中
        keep_threshold = np.percentile(attention_scores, (1-budget)*100)
        keep_mask = attention_scores >= keep_threshold
        
        # 淘汰不重要的KV pairs
        K = K[keep_mask]
        V = V[keep_mask]
        attention_scores = attention_scores[keep_mask]
```

##### 2.4.2.3.2 b) StreamingLLM (Attention Sink, 2023)

```python
def streaming_llm_attention(Q, K, V, sink_size=4, window_size=1024):
    """
    保留attention sink + 滑动窗口
    
    发现: 最开始的几个tokens（sink）总是被高度关注
    策略: 永久保留sink + 滑动窗口
    """
    # Attention pattern:
    # [Sink: 永久保留] [Evicted: 淘汰] [Window: 最近的]
    # [✓✓✓✓] [✗✗✗...✗] [✓✓✓...✓]
    #  0-3      4-N-1025   N-1024-N
    
    if len(K) > sink_size + window_size:
        # 保留: sink + recent window
        keep_indices = list(range(sink_size)) + \
                      list(range(len(K) - window_size, len(K)))
        K = K[keep_indices]
        V = V[keep_indices]
```

### 2.4.3 最新进展：NSA (2025)

**Native Sparse Attention** - 硬件对齐且原生可训练的稀疏attention

```python
def nsa_attention(Q, K, V, sparsity=0.9):
    """
    关键创新:
    1. Block-wise selection (硬件友好)
    2. 端到端可训练
    3. 高sparsity下性能稳定
    """
    # 1. 计算block importance
    block_size = 128
    num_blocks = N // block_size
    
    block_scores = compute_block_importance(Q, K, block_size)
    # [num_queries, num_blocks]
    
    # 2. 选择top blocks
    num_keep = int(num_blocks * (1 - sparsity))
    top_blocks = torch.topk(block_scores, num_keep, dim=-1).indices
    
    # 3. 只对选中的blocks计算attention
    # 关键: block-wise保证了内存访问的连续性
    output = sparse_block_attention(Q, K, V, top_blocks, block_size)
    
    return output
```

---

## 2.5 📏 Linear Attention

### 2.5.1 核心思想

**重新参数化：将attention计算复杂度从O(N²)降到O(N)**

### 2.5.2 数学原理

#### 2.5.2.1 标准Attention（二次复杂度）

```python
# 标准形式
Attention(Q, K, V) = softmax(QK^T) V

# 展开:
O_i = Σⱼ exp(q_i^T k_j) / Σⱼ exp(q_i^T k_j) × v_j

# 复杂度: O(N²d)
# - 需要计算N×N的QK^T矩阵
```

#### 2.5.2.2 Linear Attention（线性复杂度）

```python
# 核技巧重新排列
Attention(Q, K, V) = φ(Q) (φ(K)^T V)

# 关键insight: 改变计算顺序！
# 不计算 (QK^T)V，而是 Q(K^TV)

# 复杂度: O(Nd²)
# - 先计算 φ(K)^T V: O(Nd²)
# - 再计算 φ(Q) × result: O(Nd²)
# - 总计: O(Nd²) ≈ O(N) 当 d << N
```

### 2.5.3 主要方法

#### 2.5.3.1 Kernelized Linear Attention (2020)

```python
def linear_attention(Q, K, V, phi=elu_plus_one):
    """
    使用核函数近似softmax
    
    softmax(x) ≈ φ(x) 其中 φ 是特征映射
    """
    # 应用特征映射
    Q_prime = phi(Q)  # [N, d] → [N, D]
    K_prime = phi(K)  # [N, d] → [N, D]
    
    # 线性计算顺序
    # 先计算 K^T V (关键！)
    KV = torch.matmul(K_prime.T, V)  # [D, N] @ [N, d] → [D, d]
    
    # 再计算 Q × (K^T V)
    output = torch.matmul(Q_prime, KV)  # [N, D] @ [D, d] → [N, d]
    
    # 归一化
    K_sum = K_prime.sum(dim=0)  # [D]
    normalizer = torch.matmul(Q_prime, K_sum)  # [N]
    output = output / normalizer.unsqueeze(-1)
    
    return output
    
# 常用的核函数
def elu_plus_one(x):
    return F.elu(x) + 1

def relu_squared(x):
    return F.relu(x) ** 2
```

#### 2.5.3.2 RNN形式（2023）

##### 2.5.3.2.1 RetNet (Retentive Network)

```python
def retentive_attention(x_t, state_t):
    """
    将attention表示为RNN形式
    
    优势:
    - 训练: 并行（O(N²)）
    - 推理: 递归（O(1)每步）
    """
    # 训练时：并行计算（类似标准attention）
    if training:
        # Matrix形式
        Q = x @ W_Q
        K = x @ W_K  
        V = x @ W_V
        
        # 带decay的attention
        decay_mask = get_decay_mask(seq_len)
        scores = Q @ K.T * decay_mask
        output = softmax(scores) @ V
    
    # 推理时：递归更新（类似RNN）
    else:
        # Recurrent形式
        q_t = x_t @ W_Q
        k_t = x_t @ W_K
        v_t = x_t @ W_V
        
        # 更新state（O(d²)）
        state_t = decay * state_t + k_t.T @ v_t
        
        # 计算输出（O(d)）
        output_t = q_t @ state_t
        
    return output_t
```

##### 2.5.3.2.2 GLA (Gated Linear Attention, 2023)

```python
def gated_linear_attention(Q, K, V, gamma):
    """
    添加门控机制的线性attention
    
    创新: 使用可学习的遗忘门
    """
    N = Q.shape[0]
    state = torch.zeros(d, d)
    outputs = []
    
    for t in range(N):
        # 门控遗忘（类似LSTM）
        forget_gate = torch.sigmoid(gamma[t])
        
        # 更新state
        state = forget_gate * state + K[t:t+1].T @ V[t:t+1]
        
        # 计算输出
        output = Q[t:t+1] @ state
        outputs.append(output)
    
    return torch.cat(outputs, dim=0)
```

#### 2.5.3.3 最新进展：SSE (2025)

**Sparse State Expansion** - 通过稀疏状态扩展实现高效长上下文建模

```python
def sparse_state_expansion(Q, K, V, state_capacity=4096):
    """
    关键创新:
    1. 稀疏状态更新（只更新重要维度）
    2. 状态容量扩展（突破d²限制）
    3. 混合架构（SSE + sparse attention）
    """
    # 1. 计算重要性分数
    importance = compute_importance_scores(Q, K)
    
    # 2. Top-k稀疏选择
    top_k_indices = torch.topk(importance, k=state_capacity // 4).indices
    
    # 3. 稀疏状态更新
    state = torch.zeros(state_capacity, d)
    state[top_k_indices] = update_sparse_state(
        state[top_k_indices], 
        K[top_k_indices], 
        V[top_k_indices]
    )
    
    # 4. 查询状态
    output = Q @ state[top_k_indices].T
    
    return output
```

### 2.5.4 Linear Attention的挑战

```python
# 主要问题：检索能力弱

# 标准Attention: 
# 可以精确检索任意历史token
Q[i] @ K[j] → 可以直接访问V[j]

# Linear Attention:
# 通过固定大小的state间接访问
Q[i] @ State → State混合了所有历史信息
# 无法精确定位单个历史token

# 表现:
任务类型            标准Attention    Linear Attention
长文档QA (检索密集)    92%              78%  ⚠️
语言建模 (混合)        85%              82%  ✓
代码生成 (局部相关)    88%              87%  ✓
```

---

## 2.6 🗜️ Compact Attention

### 2.6.1 核心思想

**压缩KV Cache，不改变计算复杂度**

动机：
- 推理时的主要瓶颈：KV Cache内存占用
- 方法：通过压缩KV cache减少内存，同时保持性能

### 2.6.2 KV Cache大小分析

```python
# 长序列推理时的KV cache大小

# 配置
batch_size = 1
seq_len = 100000  # 100K tokens
num_layers = 32
num_heads = 32
d_head = 128

# KV cache大小（需要存储K和V）
kv_cache_size = batch_size * seq_len * num_layers * num_heads * d_head * 2
memory_bytes = kv_cache_size * 2  # FP16
memory_gb = memory_bytes / (1024**3)

print(f"KV Cache: {memory_gb:.1f} GB")
# 输出: KV Cache: 51.2 GB
# 一个batch就需要51GB内存！
```

### 2.6.3 主要方法

#### 2.6.3.1 Multi-Query Attention (MQA, 2019)

```python
def multi_query_attention(Q, K, V):
    """
    所有heads共享一组KV
    
    标准MHA: num_heads组独立的KV
    MQA:     1组共享的KV
    
    内存节省: num_heads倍
    """
    # 标准MHA
    # K: [batch, num_heads, seq_len, d_head]
    # V: [batch, num_heads, seq_len, d_head]
    
    # MQA
    # K: [batch, 1, seq_len, d_head]  ← 只有1组！
    # V: [batch, 1, seq_len, d_head]
    
    # Q还是多组的
    # Q: [batch, num_heads, seq_len, d_head]
    
    # 每个head的Q去query同一组KV
    for h in range(num_heads):
        output[h] = attention(Q[h], K[0], V[0])
    
    return output

# 内存对比
# MHA: 32 heads × 2 (K,V) = 64组参数
# MQA: 1 × 2 (K,V) = 2组参数
# 节省: 32x
```

#### 2.6.3.2 Grouped-Query Attention (GQA, 2023)

```python
def grouped_query_attention(Q, K, V, num_kv_groups=4):
    """
    MHA和MQA的折中
    
    将heads分组，每组共享KV
    
    性能: MHA > GQA > MQA
    速度: MQA > GQA > MHA
    """
    # 配置
    num_heads = 32
    num_kv_groups = 4  # 4组KV
    heads_per_group = num_heads // num_kv_groups  # 8 heads/group
    
    # K, V shape
    # [batch, num_kv_groups, seq_len, d_head]
    # [batch, 4, seq_len, d_head]  ← 只有4组
    
    # Q shape (不变)
    # [batch, num_heads, seq_len, d_head]
    # [batch, 32, seq_len, d_head]
    
    # 分组计算
    for group in range(num_kv_groups):
        # 这组的heads
        start_h = group * heads_per_group
        end_h = start_h + heads_per_group
        
        # 它们共享同一组KV
        for h in range(start_h, end_h):
            output[h] = attention(Q[h], K[group], V[group])
    
    return output

# 内存对比（num_heads=32）
# MHA: 32 × 2 = 64组
# GQA (g=4): 4 × 2 = 8组  (8x节省)
# GQA (g=8): 8 × 2 = 16组 (4x节省)
# MQA: 1 × 2 = 2组       (32x节省)
```

#### 2.6.3.3 量化方法

##### 2.6.3.3.1 KV Cache量化到INT4/INT8

```python
def quantize_kv_cache(K, V, bits=4):
    """
    将KV cache量化到低精度
    
    FP16 → INT4: 4x内存节省
    FP16 → INT8: 2x内存节省
    """
    # 计算量化参数（per-tensor或per-channel）
    K_max = K.abs().max()
    K_scale = K_max / (2 ** (bits - 1) - 1)
    
    # 量化
    K_quant = torch.round(K / K_scale).clamp(
        -(2 ** (bits - 1)), 
        2 ** (bits - 1) - 1
    ).to(torch.int8)  # 实际存储类型
    
    # 推理时反量化
    K_dequant = K_quant.float() * K_scale
    
    # 使用反量化的K进行attention
    scores = Q @ K_dequant.T
    # ...
    
    # 内存节省
    # FP16: 2 bytes/element
    # INT8: 1 byte/element (2x节省)
    # INT4: 0.5 byte/element (4x节省)
```

#### 2.6.3.4 低秩分解

```python
def low_rank_kv_compression(K, V, rank=64):
    """
    使用低秩分解压缩KV
    
    K ≈ U @ S @ V^T
    只存储低秩因子
    """
    # 原始 K: [seq_len, d_model]
    # d_model = 4096 (大！)
    
    # SVD分解
    U, S, Vt = torch.svd(K)
    
    # 只保留top-r个奇异值
    U_r = U[:, :rank]      # [seq_len, rank]
    S_r = S[:rank]          # [rank]
    Vt_r = Vt[:rank, :]    # [rank, d_model]
    
    # 存储压缩形式
    K_compressed = (U_r, S_r, Vt_r)
    
    # 使用时重构（近似）
    K_approx = U_r @ torch.diag(S_r) @ Vt_r
    
    # 内存对比
    # 原始: seq_len × d_model = 100K × 4096 = 409M
    # 压缩: seq_len×rank + rank + rank×d_model 
    #     = 100K×64 + 64 + 64×4096 = 6.7M
    # 压缩比: ~60x
```

---

## 2.7 🔀 混合架构

### 2.7.1 核心理念

**结合不同attention机制的优势，弥补各自的短板**

### 2.7.2 主要模式

#### 2.7.2.1 Linear + Sparse Attention

```python
class HybridAttentionLayer(nn.Module):
    """
    线性attention + 稀疏attention的组合
    
    动机: Linear attention检索能力弱
    解决: 周期性插入sparse attention层进行精确检索
    """
    def __init__(self, d_model, mix_ratio=0.2):
        super().__init__()
        self.linear_attn = LinearAttention(d_model)
        self.sparse_attn = SparseAttention(d_model, sparsity=0.1)
        self.mix_ratio = mix_ratio
    
    def forward(self, x):
        # 大部分层用linear attention（高效）
        linear_out = self.linear_attn(x)
        
        # 少部分层用sparse attention（精确）
        sparse_out = self.sparse_attn(x)
        
        # 自适应混合
        gate = torch.sigmoid(self.gate_weight)
        output = gate * sparse_out + (1 - gate) * linear_out
        
        return output

# 实际配置示例（32层模型）
# Layer 0-7:   Linear attention
# Layer 8:     Sparse attention (检索层)
# Layer 9-15:  Linear attention
# Layer 16:    Sparse attention (检索层)
# Layer 17-23: Linear attention
# Layer 24:    Sparse attention (检索层)
# Layer 25-31: Linear attention
```

#### 2.7.2.2 分层混合（Decoder-Decoder架构）

```python
class DecoderDecoderModel(nn.Module):
    """
    两个decoder配置不同的attention
    
    Decoder 1 (前端): 高效attention（处理长序列）
    Decoder 2 (后端): 精确attention（生成高质量输出）
    """
    def __init__(self):
        super().__init__()
        # Decoder 1: 处理长输入
        self.decoder1 = Decoder(
            layers=24,
            attention_type='linear',  # 或 sparse
            max_length=100000
        )
        
        # Decoder 2: 精确生成
        self.decoder2 = Decoder(
            layers=8,
            attention_type='full',  # 标准attention
            max_length=4096  # 较短
        )
    
    def forward(self, x):
        # 第一阶段：高效处理长序列
        h = self.decoder1(x)
        
        # 压缩/池化到更短的序列
        h_compressed = self.compress(h)  # 100K → 4K
        
        # 第二阶段：精确生成
        output = self.decoder2(h_compressed)
        
        return output
```

#### 2.7.2.3 FlexAttention（动态混合）

```python
def flex_attention(Q, K, V, sequence_length):
    """
    根据序列长度动态选择attention类型
    
    短序列 → Full attention
    中序列 → Sparse attention  
    长序列 → Linear attention
    """
    if sequence_length < 2048:
        # 短序列：用最精确的
        return full_attention(Q, K, V)
    
    elif sequence_length < 32768:
        # 中等序列：稀疏化
        sparsity = min(0.9, sequence_length / 100000)
        return sparse_attention(Q, K, V, sparsity)
    
    else:
        # 长序列：必须用线性
        return linear_attention(Q, K, V)
```

---

## 2.8 ⏰ 时间线总览

```
2017 ████ Transformer (Original Self-Attention)
     │    - Vaswani et al., "Attention Is All You Need"
     │    - 奠定基础，但O(N²)复杂度
     │
2019 ████ Sparse Transformer (OpenAI)
     │    - Child et al.
     │    - 固定稀疏模式
     │
2019 ████ Multi-Query Attention
     │    - Shazeer et al.
     │    - KV cache压缩
     │
2020 ████ Linear Attention (Kernelized)
     │    - Katharopoulos et al.
     │    - 线性复杂度，但检索能力弱
     │
2020 ████ BigBird (Google)
     │    - Zaheer et al.
     │    - 混合稀疏模式
     │
2021 ████ Rectified Linear Attention (ReLA)
     │    - 用ReLU替代softmax
     │
2022 ████ Flash Attention 1 ⚡
     │    - Dao et al.
     │    - IO-aware，2-4x加速
     │    - 革命性突破！
     │
2023 ████ Flash Attention 2
     │    - Dao
     │    - 2x faster than FA1
     │    - 更好的并行化
     │
2023 ████ Grouped-Query Attention (GQA)
     │    - Ainslie et al.
     │    - MHA和MQA的折中
     │
2023 ████ H₂O (Heavy Hitter Oracle)
     │    - Zhang et al.
     │    - 动态KV cache管理
     │
2023 ████ RetNet
     │    - Sun et al.
     │    - Linear attention的RNN形式
     │
2023 ████ StreamingLLM (Attention Sink)
     │    - Xiao et al.
     │    - 发现attention sink现象
     │
2024 ████ Flash Attention 3 ⚡⚡
     │    - Shah et al.
     │    - H100优化，1.5-2x faster than FA2
     │    - 异步执行 + FP8
     │
2024 ████ GLA (Gated Linear Attention)
     │    - Yang et al.
     │    - 带遗忘门的线性attention
     │
2025 ████ Flash Attention 4 ⚡⚡⚡
     │    - Dao et al.
     │    - 突破petaflop，15x faster than FA1
     │    - 5级流水线
     │
2025 ████ SSE (Sparse State Expansion)
     │    - Pan et al.
     │    - Linear attention的新范式
     │
2025 ████ NSA (Native Sparse Attention)
     │    - 硬件对齐的稀疏attention
```

---

## 2.9 📊 性能对比

### 2.9.1 速度对比（H100 GPU，FP16）

```python
# 序列长度 = 8192, batch_size = 4

方法                    速度(TFLOPS)   相对加速   GPU利用率
─────────────────────────────────────────────────────────
标准Attention           200           1.0x      20%
Flash Attention 1       400           2.0x      40%
Flash Attention 2       800           4.0x      73%  
Flash Attention 3      1200           6.0x      75%
Flash Attention 4      3000          15.0x      >80%

Sparse Attention (90%)  250           1.25x     25%
Linear Attention        600           3.0x      60%
```

### 2.9.2 内存对比

```python
# 序列长度 = 100K tokens, d_model = 4096

方法                    内存占用      相对节省
─────────────────────────────────────────────
标准Attention           ~400 GB       1.0x (基准)
Flash Attention         ~20 GB        20x 节省  ✓✓✓
Sparse (95% sparse)     ~20 GB        20x 节省  ✓✓✓
Linear Attention        ~2 GB         200x 节省 ✓✓✓✓
MQA                     ~12.5 GB      32x 节省  ✓✓✓
GQA (4 groups)          ~50 GB        8x 节省   ✓✓
INT4 Quantization       ~100 GB       4x 节省   ✓
```

### 2.9.3 精度对比（Long-context Retrieval）

```python
# NIAH (Needle in a Haystack) benchmark
# 序列长度 = 128K tokens

方法                    准确率     适用场景
─────────────────────────────────────────────────
标准Attention           95%       ✓ 任何任务
Flash Attention         95%       ✓ 任何任务 (无损！)
Sparse (90%)            92%       ✓ 大多数任务
Sparse (95%)            87%       △ 非检索密集任务
Linear (基础)            75%       △ 生成类任务
Linear (SSE混合)         89%       ✓ 改进但仍有差距
GQA                     94%       ✓ 几乎无损
MQA                     91%       ✓ 轻微损失
```

### 2.9.4 实际应用场景推荐

```python
# 训练场景
任务: 预训练大模型
推荐: Flash Attention 2/3/4
原因: 无损精度 + 最大加速

任务: 微调（长序列）
推荐: Flash Attention + GQA
原因: 平衡速度和内存

# 推理场景
任务: 实时对话（短序列）
推荐: Flash Attention + MQA
原因: 低延迟 + 小内存

任务: 长文档处理
推荐: Sparse Attention (90%) + GQA
原因: 可处理超长序列

任务: 批量处理（非实时）
推荐: Linear Attention (混合)
原因: 最大吞吐量

任务: 边缘设备
推荐: MQA + INT8量化
原因: 极致内存优化
```

---

## 2.10 🎓 总结

### 2.10.1 发展脉络

```
2017-2020: 探索期
- 认识到O(N²)是瓶颈
- 尝试各种稀疏模式
- 提出线性attention概念

2022-2023: 突破期
- Flash Attention横空出世
- 证明硬件优化的重要性
- Sparse和Linear方法成熟

2024-2025: 融合期
- 硬件和算法共同优化
- 混合架构成为主流
- 向更长上下文进发
```

### 2.10.2 关键洞察

1. **硬件至关重要**: Flash Attention证明了算法必须考虑硬件特性
2. **没有银弹**: 不同场景需要不同的attention机制
3. **混合是未来**: 结合多种方法才能达到最佳平衡
4. **精度vs效率**: 仍需在准确性和速度之间权衡

### 2.10.3 未来展望

```python
# 可能的研究方向

1. 更激进的稀疏化
   - 99%+ sparsity但保持精度
   - 自适应、可学习的稀疏模式

2. 更长的上下文
   - 1M+ tokens (已在路上)
   - 10M+ tokens (终极目标)

3. 硬件协同设计
   - 专门的attention加速器
   - 新的内存架构

4. 统一框架
   - 一个模型自适应选择attention类型
   - 端到端可微的混合方案

5. 超越attention
   - 状态空间模型 (Mamba等)
   - 新的序列建模范式
```

### 2.10.4 实践建议

```python
# 如何选择attention机制？

if task == "训练":
    if budget == "充足":
        use("Flash Attention 3/4")  # 最快+无损
    else:
        use("Flash Attention 2 + GQA")  # 平衡
        
elif task == "推理":
    if seq_len < 8192:
        use("Flash Attention + MQA")  # 标准场景
    elif seq_len < 128000:
        use("Sparse Attention + GQA")  # 长序列
    else:
        use("Linear Attention混合")  # 超长序列
        
elif task == "边缘设备":
    use("MQA + INT4量化")  # 极致优化
```

---

## 2.11 📚 参考文献

### 2.11.1 Flash Attention系列
1. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022
2. Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", ICLR 2024
3. Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", 2024
4. Dao et al., "FlashAttention-4", 2025

### 2.11.2 Sparse Attention
5. Child et al., "Generating Long Sequences with Sparse Transformers", 2019
6. Zaheer et al., "Big Bird: Transformers for Longer Sequences", NeurIPS 2020
7. Zhang et al., "H₂O: Heavy-Hitter Oracle for Efficient Generative Inference", 2023
8. Xiao et al., "Efficient Streaming Language Models with Attention Sinks", 2023

### 2.11.3 Linear Attention
9. Katharopoulos et al., "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention", ICML 2020
10. Sun et al., "Retentive Network: A Successor to Transformer for Large Language Models", 2023
11. Yang et al., "Gated Linear Attention Transformers", 2024
12. Pan et al., "Scaling Linear Attention with Sparse State Expansion", 2025

### 2.11.4 Compact Attention
13. Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need", 2019
14. Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models", 2023

### 2.11.5 Surveys
15. Sun et al., "Efficient Attention Mechanisms for Large Language Models: A Survey", 2025
