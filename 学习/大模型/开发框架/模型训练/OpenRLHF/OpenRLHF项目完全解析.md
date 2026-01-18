# 1 OpenRLHF 项目完全解析（小白友好版）

## 1.1 📚 一、项目概述

### 1.1.1 什么是 OpenRLHF？

OpenRLHF 是一个**人类反馈强化学习**（RLHF）训练框架，用于训练像 ChatGPT 这样的大语言模型。

**通俗理解：**
- 想象你在教一个小孩说话
- **SFT（监督微调）**：你给它看很多例子，让它学习模仿
- **RLHF（强化学习）**：你不断给它反馈（好/不好），让它自己学会什么样的回答更好

### 1.1.2 核心特点

1. **分布式架构**：使用 Ray + vLLM + DeepSpeed，可以在多台机器、多张 GPU 上训练
2. **高性能**：使用 vLLM 加速文本生成，大幅提升训练速度
3. **多种算法**：支持 PPO、DPO、REINFORCE++ 等多种 RLHF 算法
4. **易用性**：提供命令行工具，配置简单

---

## 1.2 🎯 二、项目入口点详解

### 1.2.1 主要入口文件

项目的所有入口点都在 `openrlhf/cli/` 目录下：

```
openrlhf/cli/
├── train_sft.py          # 入口1：监督微调（第一步）
├── train_rm.py           # 入口2：奖励模型训练（第二步）
├── train_ppo_ray.py      # 入口3：PPO强化学习训练（第三步）
├── train_dpo.py          # 入口4：DPO训练（替代PPO的方法）
├── train_kto.py          # 入口5：KTO训练（另一种RLHF方法）
├── train_kd.py           # 入口6：知识蒸馏
├── batch_inference.py    # 工具：批量推理
└── serve_rm.py           # 工具：奖励模型服务
```

### 1.2.2 入口使用方式

**方式一：直接命令行调用**
```bash
python -m openrlhf.cli.train_sft --参数1 值1 --参数2 值2
```

**方式二：使用 shell 脚本（推荐）**
```bash
bash examples/scripts/train_sft.sh
```

---

## 1.3 🔄 三、完整训练流程（三步走）

### 1.3.1 第一步：监督微调（SFT）

**目的：** 让模型学会基本对话能力

**入口：** `openrlhf/cli/train_sft.py`

**核心流程：**

```python
# 1. 初始化分布式策略
strategy = get_strategy(args)
strategy.setup_distributed()

# 2. 加载预训练模型
model = Actor(
    args.pretrain,              # 基座模型路径
    lora_rank=args.lora_rank,   # LoRA参数（节省显存）
    # ... 其他参数
)

# 3. 加载数据集
train_dataset = SFTDataset(
    train_data,
    tokenizer,
    args.max_len,  # 最大序列长度
)

# 4. 创建训练器
trainer = SFTTrainer(
    model=model,
    optim=optim,
    train_dataloader=train_dataloader,
    # ... 其他参数
)

# 5. 开始训练
trainer.fit(args)
```

**实战示例：**
```bash
deepspeed --module openrlhf.cli.train_sft \
   --max_len 2048 \                      # 最大序列长度
   --dataset Open-Orca/OpenOrca \        # 数据集
   --train_batch_size 256 \              # 全局批大小
   --micro_train_batch_size 2 \          # 每张GPU批大小
   --pretrain meta-llama/Meta-Llama-3-8B \  # 基座模型
   --save_path ./checkpoint/llama3-sft \    # 保存路径
   --zero_stage 2 \                      # DeepSpeed ZeRO优化级别
   --max_epochs 1 \                      # 训练轮数
   --learning_rate 5e-6                  # 学习率
```

---

### 1.3.2 第二步：奖励模型训练（RM）

**目的：** 训练一个能判断回答好坏的模型

**入口：** `openrlhf/cli/train_rm.py`

**核心概念：**
- 输入：同一个问题的两个回答（一好一坏）
- 输出：哪个回答更好
- 训练目标：让模型学会判断回答质量

**数据格式：**
```json
{
  "prompt": "什么是人工智能？",
  "chosen": "人工智能是让机器模拟人类智能的技术...",
  "rejected": "不知道"
}
```

**实战示例：**
```bash
deepspeed --module openrlhf.cli.train_rm \
   --save_path ./checkpoint/llama3-rm \
   --dataset Anthropic/hh-rlhf \
   --pretrain ./checkpoint/llama3-sft \  # 使用SFT模型初始化
   --train_batch_size 128 \
   --learning_rate 9e-6
```

---

### 1.3.3 第三步：强化学习训练（PPO/REINFORCE++）

**目的：** 通过奖励信号优化模型，让它生成更好的回答

**入口：** `openrlhf/cli/train_ppo_ray.py`

**核心流程图：**

```
┌────────────┐
│  提示词库   │ (Prompt Dataset)
└─────┬──────┘
      │
      ▼
┌────────────────┐
│  Actor Model   │ 生成回答
│  (策略模型)     │
└────────┬───────┘
         │
         ▼
┌────────────────┐
│  生成的回答     │
└────────┬───────┘
         │
         ├──────► ┌──────────────┐
         │        │ Reward Model │ 打分
         │        │  (奖励模型)   │
         │        └──────┬───────┘
         │               │
         ▼               ▼
┌────────────────────────────┐
│  计算优势函数 (Advantage)   │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│  PPO 优化 Actor 和 Critic   │
└────────────────────────────┘
```

**关键组件：**

1. **Actor Model（演员/策略模型）**
   - 作用：生成回答
   - 训练目标：最大化奖励

2. **Critic Model（评论家/价值模型）**
   - 作用：评估 Actor 生成的回答有多好
   - 训练目标：准确预测未来奖励

3. **Reward Model（奖励模型）**
   - 作用：给 Actor 的回答打分
   - 不参与训练（已在第二步训练好）

4. **Reference Model（参考模型）**
   - 作用：防止 Actor 偏离太远
   - 计算 KL 散度作为惩罚

**核心代码结构：**

```python
# train_ppo_ray.py 主要步骤

# 1. 初始化 Ray（分布式框架）
ray.init()

# 2. 创建 vLLM 引擎（加速生成）
vllm_engines = create_vllm_engines(
    num_engines=4,              # vLLM引擎数量
    tensor_parallel_size=2,     # 张量并行度
)

# 3. 创建各个模型组
actor_model = RayActorGroup(...)      # Actor模型组
critic_model = RayActorGroup(...)     # Critic模型组
reward_model = RayActorGroup(...)     # Reward模型组
ref_model = RayActorGroup(...)        # Reference模型组

# 4. 初始化 PPO Trainer
trainer = PPOTrainer(
    strategy,
    actor_model,
    critic_model,
    reward_model,
    ref_model,
    vllm_engines,
)

# 5. 开始训练
trainer.fit(prompts_dataloader)
```

**实战示例：**
```bash
python -m openrlhf.cli.train_ppo_ray \
   --actor_num_nodes 1 \                # Actor模型节点数
   --actor_num_gpus_per_node 8 \        # 每节点GPU数
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 8 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 8 \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --vllm_num_engines 4 \               # vLLM引擎数
   --vllm_tensor_parallel_size 2 \      # 张量并行
   --pretrain ./checkpoint/llama3-sft \      # Actor初始模型
   --reward_pretrain ./checkpoint/llama3-rm \ # Reward模型
   --train_batch_size 128 \             # 训练批大小
   --rollout_batch_size 1024 \          # Rollout批大小
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6
```

---

## 1.4 📋 四、关键参数详解

### 1.4.1 通用参数

| 参数名 | 含义 | 推荐值 | 说明 |
|--------|------|--------|------|
| `--pretrain` | 预训练模型路径 | `meta-llama/Meta-Llama-3-8B` | HuggingFace 模型路径或本地路径 |
| `--save_path` | 模型保存路径 | `./checkpoint/model` | 训练后的模型保存位置 |
| `--max_len` | 最大序列长度 | `2048` | 输入+输出的总长度 |
| `--zero_stage` | DeepSpeed ZeRO 级别 | `2` 或 `3` | 0/1/2/3，数字越大省显存 |
| `--param_dtype` | 模型精度 | `bf16` | `bf16` 或 `fp16` |
| `--seed` | 随机种子 | `42` | 保证可复现性 |

### 1.4.2 SFT 专用参数

| 参数名 | 含义 | 推荐值 | 说明 |
|--------|------|--------|------|
| `--dataset` | 数据集路径 | `Open-Orca/OpenOrca` | HuggingFace 数据集或本地路径 |
| `--input_key` | 输入字段名 | `"question"` | JSON 数据中的输入字段 |
| `--output_key` | 输出字段名 | `"response"` | JSON 数据中的输出字段 |
| `--train_batch_size` | 全局批大小 | `256` | 所有 GPU 的总批大小 |
| `--micro_train_batch_size` | 单 GPU 批大小 | `2` | 每张 GPU 处理的样本数 |
| `--max_epochs` | 训练轮数 | `1-3` | 数据过几遍 |
| `--learning_rate` | 学习率 | `5e-6` | Adam 优化器学习率 |
| `--packing_samples` | 打包样本 | 开启 | 提升训练效率，推荐开启 |
| `--gradient_checkpointing` | 梯度检查点 | 开启 | 省显存，推荐开启 |

**计算公式：**
- **梯度累积步数** = `train_batch_size / (micro_train_batch_size × GPU数量)`
- **实际批大小** = `micro_train_batch_size × GPU数量`

**示例：**
- 如果有 8 张 GPU，`train_batch_size=256`，`micro_train_batch_size=2`
- 那么梯度累积步数 = 256 / (2 × 8) = 16 步
- 每 16 步更新一次参数

### 1.4.3 RM 专用参数

| 参数名 | 含义 | 推荐值 |
|--------|------|--------|
| `--dataset` | 偏好数据集 | `Anthropic/hh-rlhf` |
| `--learning_rate` | 学习率 | `9e-6` |
| `--loss` | 损失函数 | `"sigmoid"` 或 `"ranking"` |

**数据格式要求：**
```json
{
  "prompt": "问题",
  "chosen": "好的回答",
  "rejected": "差的回答"
}
```

### 1.4.4 PPO 专用参数（核心）

#### 1.4.4.1 资源配置参数

| 参数名 | 含义 | 说明 |
|--------|------|------|
| `--actor_num_nodes` | Actor 模型节点数 | 运行 Actor 的机器数量 |
| `--actor_num_gpus_per_node` | 每节点 GPU 数 | 每台机器用几张 GPU |
| `--critic_num_nodes` | Critic 模型节点数 | 同上 |
| `--reward_num_nodes` | Reward 模型节点数 | 同上 |
| `--ref_num_nodes` | Reference 模型节点数 | 同上 |
| `--vllm_num_engines` | vLLM 引擎数量 | 用于加速生成 |
| `--vllm_tensor_parallel_size` | vLLM 张量并行度 | 每个引擎用几张 GPU |

**资源配置示例：**

假设你有 **1 台机器，8 张 GPU**：

```bash
# 配置 1：所有模型共享（省显存）
--actor_num_nodes 1 \
--actor_num_gpus_per_node 8 \
--critic_num_nodes 1 \
--critic_num_gpus_per_node 8 \
--reward_num_nodes 1 \
--reward_num_gpus_per_node 8 \
--ref_num_nodes 1 \
--ref_num_gpus_per_node 8 \
--colocate_all_models \   # 关键：共享显存

# 配置 2：分离模型（需要更多GPU）
--actor_num_gpus_per_node 2 \    # Actor 用 2 张
--critic_num_gpus_per_node 2 \   # Critic 用 2 张
--reward_num_gpus_per_node 2 \   # Reward 用 2 张
--ref_num_gpus_per_node 2        # Ref 用 2 张
```

#### 1.4.4.2 训练控制参数

| 参数名 | 含义 | 推荐值 | 说明 |
|--------|------|--------|------|
| `--prompt_data` | 提示词数据集 | `OpenRLHF/prompt-collection-v0.1` | 用于生成的提示词 |
| `--prompt_max_len` | 提示词最大长度 | `1024` | 问题的最大长度 |
| `--generate_max_len` | 生成最大长度 | `1024` | 回答的最大长度 |
| `--train_batch_size` | 训练批大小 | `128` | PPO 更新时的批大小 |
| `--rollout_batch_size` | Rollout 批大小 | `512-1024` | 一次生成多少条数据 |
| `--n_samples_per_prompt` | 每个提示词生成数 | `1-4` | 每个问题生成几个回答 |
| `--max_epochs` | 训练轮数 | `1` | 数据过几遍 |

**关键理解：**

$$
\text{总生成样本数} = \text{rollout\_batch\_size} \times \text{n\_samples\_per\_prompt}
$$

例如：
- `rollout_batch_size=512`，`n_samples_per_prompt=2`
- 每次生成 $512 \times 2 = 1024$ 个样本

#### 1.4.4.3 PPO 算法参数（重要！）

| 参数名                      | 含义           | 推荐值         | 说明            |
| ------------------------ | ------------ | ----------- | ------------- |
| `--actor_learning_rate`  | Actor 学习率    | `5e-7`      | 越大学得越快，但可能不稳定 |
| `--critic_learning_rate` | Critic 学习率   | `9e-6`      | 通常比 Actor 大   |
| `--init_kl_coef`         | KL 散度系数      | `0.01-0.05` | 防止模型偏离太远      |
| `--clip_range_value`     | Value 裁剪范围   | `5.0`       | PPO 裁剪参数      |
| `--clip_range_ratio`     | Ratio 裁剪范围   | `0.2`       | PPO 裁剪参数      |
| `--gae_lambda`           | GAE 参数       | `0.95`      | 计算优势函数的参数     |
| `--normalize_reward`     | 标准化奖励        | 开启          | 推荐开启，稳定训练     |
| `--adam_offload`         | Adam 卸载到 CPU | 可选          | 省显存但变慢        |


**KL 散度解释：**

$$
\text{KL}(P_{\text{actor}} \| P_{\text{ref}}) = \text{衡量 Actor 和 Reference 模型的差异}
$$

- KL 太大：Actor 偏离太远，可能输出奇怪内容
- KL 太小：Actor 学不到新东西
- 通过 `init_kl_coef` 控制这个平衡

#### 1.4.4.4 性能优化参数

| 参数名 | 含义 | 推荐值 |
|--------|------|--------|
| `--vllm_gpu_memory_utilization` | vLLM 显存利用率 | `0.5-0.8` |
| `--enable_prefix_caching` | 前缀缓存 | 开启 |
| `--packing_samples` | 打包样本 | 开启 |
| `--gradient_checkpointing` | 梯度检查点 | 开启 |
| `--attn_implementation` | 注意力实现 | `flash_attention_2` |
| `--use_liger_kernel` | Liger 内核 | 可选 |

### 1.4.5 LoRA 参数（省显存利器）

| 参数名 | 含义 | 推荐值 | 说明 |
|--------|------|--------|------|
| `--lora_rank` | LoRA 秩 | `8-64` | 越大效果越好但占显存 |
| `--lora_alpha` | LoRA alpha | `16-128` | 通常设为 rank 的 2 倍 |
| `--target_modules` | 目标模块 | `all-linear` | 对哪些层使用 LoRA |
| `--lora_dropout` | LoRA dropout | `0.05` | 防止过拟合 |
| `--load_in_4bit` | 4-bit 量化 | 可选 | 极度省显存 |

**什么是 LoRA？**

- 不训练整个模型，只训练小部分参数
- **显存需求**：全量微调 > LoRA > 推理
- **7 B 模型 LoRA 示例**：只需 1 张 24 GB GPU

---

## 1.5 🚀 五、实战案例（一步步来）

### 1.5.1 案例 1：训练一个 7 B 聊天模型（单卡）

**硬件要求：** 1 张 24 GB GPU（如 RTX 4090）

#### 1.5.1.1 Step 1：准备数据

```python
# 数据格式示例 (data.jsonl)
{"question": "什么是深度学习？", "response": "深度学习是机器学习的一个分支..."}
{"question": "如何学习Python？", "response": "学习Python可以从基础语法开始..."}
```

#### 1.5.1.2 Step 2：SFT 训练

```bash
# train_sft_7b.sh
deepspeed --module openrlhf.cli.train_sft \
   --max_len 2048 \
   --dataset ./data.jsonl \
   --input_key question \
   --output_key response \
   --train_batch_size 128 \
   --micro_train_batch_size 1 \      # 单卡只能设1
   --pretrain meta-llama/Llama-2-7b-hf \
   --save_path ./checkpoint/llama2-7b-sft \
   --zero_stage 2 \
   --max_epochs 3 \
   --learning_rate 5e-6 \
   --lora_rank 64 \                  # 使用LoRA省显存
   --lora_alpha 128 \
   --gradient_checkpointing \        # 必须开启
   --packing_samples
```

#### 1.5.1.3 Step 3：RM 训练（可选）

```bash
# 如果有偏好数据
deepspeed --module openrlhf.cli.train_rm \
   --save_path ./checkpoint/llama2-7b-rm \
   --dataset ./preference_data.jsonl \
   --pretrain ./checkpoint/llama2-7b-sft \
   --train_batch_size 64 \
   --micro_train_batch_size 1 \
   --learning_rate 9e-6 \
   --lora_rank 64
```

#### 1.5.1.4 Step 4：PPO 训练（需要更多显存）

```bash
# 注意：单卡PPO比较困难，建议4卡以上
python -m openrlhf.cli.train_ppo_ray \
   --actor_num_gpus_per_node 1 \
   --critic_num_gpus_per_node 1 \
   --reward_num_gpus_per_node 1 \
   --ref_num_gpus_per_node 1 \
   --colocate_all_models \           # 共享显存
   --pretrain ./checkpoint/llama2-7b-sft \
   --reward_pretrain ./checkpoint/llama2-7b-rm \
   --prompt_data ./prompts.jsonl \
   --train_batch_size 32 \           # 减小批大小
   --rollout_batch_size 128 \
   --lora_rank 64                    # 使用LoRA
```

---

### 1.5.2 案例 2：训练 70 B 模型（8 卡）

**硬件要求：** 8 张 80 GB GPU（如 A 100）

```bash
# SFT 训练 70B
deepspeed --module openrlhf.cli.train_sft \
   --max_len 4096 \
   --dataset Open-Orca/OpenOrca \
   --train_batch_size 512 \
   --micro_train_batch_size 1 \
   --pretrain meta-llama/Llama-2-70b-hf \
   --save_path ./checkpoint/llama2-70b-sft \
   --zero_stage 3 \                  # ZeRO-3 节省显存
   --max_epochs 1 \
   --learning_rate 2e-6 \
   --gradient_checkpointing \
   --packing_samples \
   --ds_tensor_parallel_size 8      
```

```bash
# 张量并行
# PPO 训练 70B
python -m openrlhf.cli.train_ppo_ray \
   --actor_num_gpus_per_node 8 \
   --critic_num_gpus_per_node 8 \
   --reward_num_gpus_per_node 8 \
   --ref_num_gpus_per_node 8 \
   --colocate_all_models \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 4 \   # vLLM 张量并行
   --pretrain ./checkpoint/llama2-70b-sft \
   --train_batch_size 256 \
   --rollout_batch_size 2048 \
   --zero_stage 3 \
   --ds_tensor_parallel_size 8
```

---

## 1.6 🔧 六、常见问题与解决方案

### 1.6.1 Q 1：显存不足怎么办？

**方案 1：使用 LoRA**
```bash
--lora_rank 8 \
--lora_alpha 16
```

**方案 2：开启梯度检查点**
```bash
--gradient_checkpointing
```

**方案 3：提高 ZeRO 级别**
```bash
--zero_stage 3
```

**方案 4：减小批大小**
```bash
--micro_train_batch_size 1 \
--train_batch_size 32
```

**方案 5：4-bit 量化**
```bash
--load_in_4bit
```

**方案 6：卸载 Adam 到 CPU**
```bash
--adam_offload
```

### 1.6.2 Q 2：训练太慢怎么办？

**方案 1：使用 FlashAttention**
```bash
--attn_implementation flash_attention_2
```

**方案 2：打包样本**
```bash
--packing_samples
```

**方案 3：使用 vLLM**
```bash
--vllm_num_engines 4
```

**方案 4：增加 GPU 数量**
```bash
# 多机训练
--num_nodes 2
```

### 1.6.3 Q 3：如何监控训练？

**方案 1：使用 WandB**
```bash
--use_wandb YOUR_WANDB_TOKEN \
--wandb_project my_project
```

**方案 2：使用 TensorBoard**
```bash
--use_tensorboard ./logs
```

### 1.6.4 Q 4：训练不稳定怎么办？

1. **降低学习率**
   ```bash
   --actor_learning_rate 1e-7  # 降低10倍
   ```

2. **增加 warmup**
   ```bash
   --lr_warmup_ratio 0.1  # 10% 步数用于 warmup
   ```

3. **启用奖励标准化**
   ```bash
   --normalize_reward
   ```

4. **调整 KL 系数**
   ```bash
   --init_kl_coef 0.02  # 增加约束
   ```

---

## 1.7 📊 七、关键公式解析

### 1.7.1 PPO 损失函数

PPO 的目标是最大化：
$L^{CLIP}(\theta) = \mathbb{E}_t[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$


**通俗理解：**
- $r_t(\theta)$：新策略和旧策略的概率比
- $\hat{A}_t$：优势函数（这个动作有多好）
- $\epsilon$：裁剪范围（防止更新太大）

**示例：**
- 如果 $\hat{A}_t > 0$（好动作），增加该动作概率
- 如果 $\hat{A}_t < 0$（坏动作），减少该动作概率
- 但不要变化太剧烈（通过 clip 限制）

### 1.7.2 优势函数（GAE）

$$
\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \cdots + (\gamma\lambda)^{T-t+1}\delta_{T-1}
$$

其中：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

**参数：**
- $\gamma$：折扣因子（未来奖励的权重）
- $\lambda$：GAE 参数（权衡偏差和方差）

**直观理解：**
- 计算每一步的"意外收益"
- 如果实际奖励 > 预期，优势为正
- 如果实际奖励 < 预期，优势为负

### 1.7.3 KL 散度惩罚

总奖励：

$$
R_{total} = R_{reward} - \beta \cdot KL(P_{\theta} \| P_{\theta_{ref}})
$$

**作用：**
- 防止 Actor 偏离 Reference 太远
- $\beta$ 由 `--init_kl_coef` 控制

**示例：**
```python
# 代码中的实现
reward = reward_model(response)  # 奖励模型打分
kl_div = compute_kl(actor_logprobs, ref_logprobs)  # 计算KL
total_reward = reward - 0.01 * kl_div  # 减去KL惩罚
```

---

## 1.8 🎓 八、核心代码走读

### 1.8.1 SFTTrainer 核心逻辑

```python
# openrlhf/trainer/sft_trainer.py

class SFTTrainer:
    def fit(self, args):
        for epoch in range(self.max_epochs):
            for batch in self.train_dataloader:
                # 1. 前向传播
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                
                # 2. 计算损失
                loss = outputs.loss
                
                # 3. 反向传播
                self.strategy.backward(loss, self.model, self.optim)
                
                # 4. 更新参数
                self.strategy.optimizer_step(self.optim, self.model, self.scheduler)
                
                # 5. 记录日志
                if step % logging_steps == 0:
                    wandb.log({"loss": loss.item()})
```

**关键点：**
- 使用 DeepSpeed `strategy` 管理分布式
- 自动处理梯度累积
- 支持混合精度训练

### 1.8.2 PPOTrainer 核心逻辑

```python
# openrlhf/trainer/ppo_trainer.py (简化版)

class PPOTrainer:
    def fit(self, prompts_dataloader):
        for epoch in range(self.max_epochs):
            for prompts in prompts_dataloader:
                # === Step 1: Rollout ===
                # 使用 vLLM 生成回答
                sequences = self.generate(prompts)
                
                # === Step 2: 计算奖励 ===
                # 奖励模型打分
                rewards = self.reward_model(sequences)
                # 参考模型 log probs
                ref_log_probs = self.ref_model(sequences)
                # 当前模型 log probs
                actor_log_probs = self.actor_model(sequences)
                # KL 散度
                kl_div = actor_log_probs - ref_log_probs
                # 总奖励
                total_rewards = rewards - self.kl_coef * kl_div
                
                # === Step 3: 计算优势 ===
                # Critic 预测价值
                values = self.critic_model(sequences)
                # 计算 GAE
                advantages = self.compute_gae(total_rewards, values)
                
                # === Step 4: PPO 更新 ===
                for _ in range(self.num_update_epochs):
                    # 采样 mini-batch
                    batch = self.sample_batch(sequences, advantages)
                    
                    # Actor 损失
                    actor_loss = self.compute_actor_loss(batch)
                    # Critic 损失
                    critic_loss = self.compute_critic_loss(batch)
                    
                    # 更新 Actor
                    self.actor_optim.zero_grad()
                    actor_loss.backward()
                    self.actor_optim.step()
                    
                    # 更新 Critic
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    self.critic_optim.step()
```

**关键函数：**

```python
def compute_actor_loss(self, batch):
    """计算 PPO Actor 损失"""
    # 新策略 log probs
    new_log_probs = self.actor_model(batch["sequences"])
    
    # 概率比
    ratio = torch.exp(new_log_probs - batch["old_log_probs"])
    
    # PPO clip
    clip_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
    
    # 取最小值
    loss1 = ratio * batch["advantages"]
    loss2 = clip_ratio * batch["advantages"]
    loss = -torch.min(loss1, loss2).mean()
    
    return loss

def compute_gae(self, rewards, values):
    """计算广义优势估计 (GAE)"""
    advantages = []
    gae = 0
    
    # 从后往前计算
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + self.gamma * values[t+1] - values[t]
        gae = delta + self.gamma * self.gae_lambda * gae
        advantages.insert(0, gae)
    
    return torch.tensor(advantages)
```

---

## 1.9 🌟 九、高级技巧

### 1.9.1 多轮对话训练

```bash
# 使用 multiturn 格式
--multiturn \
--apply_chat_template \
--tokenizer_chat_template "{% for message in messages %}..."
```

**数据格式：**
```json
{
  "messages": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮你的吗？"},
    {"role": "user", "content": "介绍一下你自己"},
    {"role": "assistant", "content": "我是一个AI助手..."}
  ]
}
```

### 1.9.2 自定义奖励函数

```python
# 创建自定义 Agent
# agent_func.py

def custom_reward_function(prompts, responses):
    """自定义奖励函数"""
    rewards = []
    for prompt, response in zip(prompts, responses):
        # 示例：奖励长度在 50-100 之间的回答
        length = len(response.split())
        if 50 <= length <= 100:
            reward = 1.0
        else:
            reward = -abs(length - 75) / 75
        rewards.append(reward)
    return rewards
```

```bash
# 使用自定义奖励
python -m openrlhf.cli.train_ppo_ray \
   --agent_func_path ./agent_func.py \
   ...
```

### 1.9.3 异步训练（提速）

```bash
# 启用异步 RLHF
python -m openrlhf.cli.train_ppo_ray \
   --async_train \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep
```

**原理：**
- 生成和训练并行进行
- 提升 GPU 利用率

### 1.9.4 Ring Attention（超长上下文）

```bash
# 支持更长的序列
--ring_attn_size 4 \
--ring_head_stride 2 \
--max_len 32768  # 支持 32K 上下文
```

**原理：**
- 将注意力计算分块
- 类似 FlashAttention 但支持更长序列

---

## 1.10 📝 十、总结与最佳实践

### 1.10.1 训练流程总结

```
1. SFT（监督微调）
   ↓
   目标：让模型会对话
   数据：问答对
   时间：1-3 天
   
2. RM（奖励模型）
   ↓
   目标：训练判断器
   数据：偏好数据（好/坏回答）
   时间：1-2 天
   
3. PPO（强化学习）
   ↓
   目标：优化模型输出
   数据：提示词
   时间：3-7 天
```

### 1.10.2 最佳实践

**1. 数据准备**
- SFT：至少 10 K 高质量问答对
- RM：至少 50 K 偏好对
- PPO：至少 10 K 多样化提示词

**2. 超参数设置**
- 学习率：从小开始（5 e-7），逐步尝试
- 批大小：尽量大（受显存限制）
- KL 系数：0.01-0.05 之间

**3. 监控指标**
- SFT：训练损失、验证损失
- RM：准确率、AUC
- PPO：平均奖励、KL 散度、策略熵

**4. 调试技巧**
- 先在小数据集上验证流程
- 使用小模型（如 1.5 B）快速迭代
- 记录所有超参数和结果

### 1.10.3 常用命令速查

```bash
# SFT 训练
deepspeed --module openrlhf.cli.train_sft \
   --pretrain <模型> --dataset <数据> --save_path <路径>

# RM 训练
deepspeed --module openrlhf.cli.train_rm \
   --pretrain <SFT模型> --dataset <偏好数据> --save_path <路径>

# PPO 训练
python -m openrlhf.cli.train_ppo_ray \
   --pretrain <SFT模型> --reward_pretrain <RM模型> \
   --prompt_data <提示词> --save_path <路径>

# 推理测试
python -m openrlhf.cli.interactive_chat \
   --pretrain <训练好的模型>

# LoRA 合并
python -m openrlhf.cli.lora_combiner \
   --base_model <基座模型> --lora_model <LoRA权重> \
   --output_path <输出路径>
```

---

## 1.11 🔗 十一、参考资源

### 1.11.1 官方资源
- **GitHub**：https://github.com/OpenRLHF/OpenRLHF
- **文档**：https://openrlhf.readthedocs.io/
- **技术报告**：https://www.researchgate.net/publication/393414548

### 1.11.2 学习资料
- **PPO 论文**：[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **DeepSpeed**：https://www.deepspeed.ai/
- **vLLM**：https://github.com/vllm-project/vllm
- **Ray**：https://docs.ray.io/

### 1.11.3 数据集推荐
- **SFT 数据**：Open-Orca/OpenOrca, databricks/databricks-dolly-15 k
- **RM 数据**：Anthropic/hh-rlhf, OpenAssistant/oasst 1
- **Prompt 数据**：OpenRLHF/prompt-collection-v 0.1

---

## 1.12 🎉 结语

恭喜你完成了 OpenRLHF 项目的完整学习！现在你应该能够：

✅ 理解 RLHF 的完整流程
✅ 知道如何配置和运行训练脚本
✅ 掌握关键参数的含义和调优方法
✅ 解决常见问题和错误

**下一步建议：**
1. 在小数据集上跑通整个流程
2. 阅读关键代码文件，理解实现细节
3. 尝试调整参数，观察效果
4. 加入社区，和其他人交流

祝你训练出优秀的模型！🚀
