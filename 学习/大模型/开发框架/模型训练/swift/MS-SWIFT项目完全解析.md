# 1 MS-SWIFT 项目完全解析指南

## 1.1 📚 一、项目概述

### 1.1.1 什么是MS-SWIFT？

MS-SWIFT（Scalable lightWeight Infrastructure for Fine-Tuning）是魔搭社区开发的**大模型微调与部署框架**。

**核心能力**：
- 支持 **600+ 纯文本大模型**和 **300+ 多模态大模型**的训练
- 覆盖训练、推理、评测、量化、部署**全链路**
- 包括预训练、微调、人类对齐等多种训练任务

### 1.1.2 项目结构总览

```
ms-swift-main/
├── swift/                    # 核心代码目录
│   ├── cli/                 # 命令行入口
│   ├── arguments/           # 参数定义
│   ├── pipelines/           # 主要流程实现
│   ├── trainers/            # 训练器
│   ├── model/               # 模型相关
│   ├── dataset/             # 数据集处理
│   ├── tuners/              # 微调方法（LoRA等）
│   ├── template/            # 对话模板
│   ├── infer_engine/        # 推理引擎
│   ├── rlhf_trainers/       # 强化学习训练器
│   └── megatron/            # Megatron并行技术
├── examples/                # 示例脚本
├── docs/                    # 文档
├── tests/                   # 测试代码
└── setup.py                 # 安装配置
```

---

## 1.2 🚀 二、项目入口解析

### 1.2.1 安装后的命令行入口

通过 `setup.py` 的配置，安装后会创建两个命令行工具：

```python
entry_points={
    'console_scripts': [
        'swift=swift.cli.main:cli_main',           # 主命令
        'megatron=swift.cli._megatron.main:cli_main'  # Megatron命令
    ]
}
```

**使用示例**：
```bash
# 主命令格式
swift <子命令> [参数]

# 可用的子命令
swift sft        # 监督微调（Supervised Fine-Tuning）
swift pt         # 预训练（Pre-Training）
swift rlhf       # 人类反馈强化学习
swift infer      # 推理
swift deploy     # 部署
swift export     # 导出模型
swift eval       # 模型评测
swift app        # Web界面
```

### 1.2.2 CLI主入口分析

**文件位置**：`swift/cli/main.py`

**核心流程**：

```python
def cli_main(route_mapping=None, is_megatron=False):
    # 1. 解析命令行参数
    argv = sys.argv[1:]  # 获取命令行参数
    method_name = argv[0].replace('_', '-')  # 第一个参数是子命令
    argv = argv[1:]  # 剩余参数
    
    # 2. 路由映射：根据子命令找到对应的模块
    ROUTE_MAPPING = {
        'pt': 'swift.cli.pt',
        'sft': 'swift.cli.sft',
        'infer': 'swift.cli.infer',
        'rlhf': 'swift.cli.rlhf',
        # ... 其他命令
    }
    
    # 3. 找到目标文件路径
    file_path = importlib.util.find_spec(route_mapping[method_name]).origin
    
    # 4. 处理配置文件（如果使用 --config）
    prepare_config_args(argv)
    
    # 5. 判断是否需要分布式训练
    torchrun_args = get_torchrun_args()
    
    # 6. 执行命令
    if torchrun_args is None:
        # 单卡训练
        args = [python_cmd, file_path, *argv]
    else:
        # 多卡训练，使用torchrun
        args = [python_cmd, '-m', 'torch.distributed.run', 
                *torchrun_args, file_path, *argv]
    
    subprocess.run(args)
```

**关键点**：
1. **命令路由**：通过 `ROUTE_MAPPING` 字典将子命令映射到具体的Python模块
2. **配置文件支持**：可以使用 `--config xxx.yaml` 代替命令行参数
3. **分布式训练检测**：通过环境变量（如 `NPROC_PER_NODE`）判断是否启用torchrun

---

## 1.3 🎯 三、SFT训练流程详解

SFT（Supervised Fine-Tuning）是最常用的微调方式，我们以它为例深入分析。

### 1.3.1 入口链路

```
用户命令: swift sft --model xxx --dataset yyy
    ↓
swift/cli/main.py::cli_main()
    ↓
swift/cli/sft.py::main块
    ↓
swift/pipelines/train/sft.py::sft_main()
    ↓
SwiftSft类的执行流程
```

### 1.3.2 SwiftSft类核心流程

**文件位置**：`swift/pipelines/train/sft.py`

```python
class SwiftSft(SwiftPipeline, TunerMixin):
    args_class = SftArguments  # 参数类
    
    def __init__(self, args):
        """初始化阶段"""
        super().__init__(args)
        self.train_msg = {}
        
        # Step 1: 准备模型和分词器
        self._prepare_model_tokenizer()
        
        # Step 2: 准备对话模板
        self._prepare_template()
        
        # Step 3: 准备Flash Checkpoint（可选）
        self._prepare_flash_ckpt()
    
    def run(self):
        """主执行流程"""
        # Step 1: 准备数据集
        train_dataset, val_dataset = self._prepare_dataset()
        
        # Step 2: 保存参数配置
        self.args.save_args()
        
        # Step 3: 准备模型（应用LoRA等微调方法）
        self.model = self.prepare_model(
            self.args, 
            self.model, 
            template=self.template,
            train_dataset=train_dataset
        )
        
        # Step 4: 创建训练器
        trainer_cls = TrainerFactory.get_trainer_cls(self.args)
        trainer = trainer_cls(
            model=self.model,
            args=self.args.training_args,
            template=self.template,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        # Step 5: 开始训练
        return self.train(trainer)
```

### 1.3.3 关键步骤详解

#### 1.3.3.1 步骤1：准备模型和分词器

```python
def _prepare_model_tokenizer(self, **kwargs):
    # 1. 加载模型和处理器
    self.model, self.processor = self.args.get_model_processor(**kwargs)
    
    # 2. 序列并行处理（如果启用）
    if self.args.sequence_parallel_size > 1:
        sequence_parallel.prepare(
            self.args.sequence_parallel_size,
            model=self.model,
            tokenizer=self.processor,
            padding_free=self.args.padding_free
        )
    
    # 3. 准备生成配置
    self._prepare_generation_config()
```

**涉及参数**：
- `--model`：模型ID或路径
- `--model_type`：模型类型
- `--torch_dtype`：模型精度（fp16/bf16/fp32）
- `--sequence_parallel_size`：序列并行数

#### 1.3.3.2 步骤2：准备对话模板

```python
def _prepare_template(self):
    # 1. 获取模板
    template = self.args.get_template(self.processor)
    
    # 2. 设置为训练模式
    template.set_mode('train')
    
    # 3. 检查是否支持padding_free和packing
    if (self.args.padding_free or self.args.packing) and \
       not template.support_padding_free:
        raise ValueError('Template does not support padding free or packing.')
    
    self.template = template
```

**涉及参数**：
- `--template`：对话模板类型（如chatml、qwen等）
- `--system`：系统提示词
- `--max_length`：最大序列长度

#### 1.3.3.3 步骤3：准备数据集

```python
def _prepare_dataset(self):
    # 1. 加载数据集
    if self.args.dataset or self.args.val_dataset:
        train_dataset, val_dataset = self._get_dataset()
        
        # 2. 编码数据集
        train_dataset, val_dataset = self._encode_dataset(
            train_dataset, 
            val_dataset,
            pre_process=True
        )
    
    # 3. 后处理数据集（packing、lazy loading等）
    datasets = self._post_process_datasets([train_dataset, val_dataset])
    
    # 4. 显示数据集样例
    self._show_dataset(*datasets)
    
    return datasets
```

**数据集编码流程**：

```python
def _encode_dataset(self, train_dataset, val_dataset, pre_process=True):
    for i, dataset in enumerate([train_dataset, val_dataset]):
        if dataset is None:
            continue
        
        if not self.args.lazy_tokenize and not self.args.streaming:
            # 立即编码：使用EncodePreprocessor或AddLengthPreprocessor
            preprocessor = EncodePreprocessor(template=self.template)
            dataset = preprocessor(
                dataset,
                num_proc=self.args.dataset_num_proc,  # 并行处理
                load_from_cache_file=self.args.load_from_cache_file,
                strict=self.args.strict,
                batch_size=1000
            )
    
    return train_dataset, val_dataset
```

**涉及参数**：
- `--dataset`：训练数据集
- `--val_dataset`：验证数据集
- `--split_dataset_ratio`：从训练集分割验证集的比例
- `--dataset_num_proc`：数据处理的并行进程数
- `--lazy_tokenize`：是否延迟分词
- `--streaming`：是否流式加载数据

#### 1.3.3.4 步骤4：应用微调方法（LoRA等）

```python
def prepare_model(self, args, model, template=None, train_dataset=None):
    """TunerMixin中的方法"""
    
    if args.tuner_backend == 'peft':
        # 使用PEFT库（LoRA、QLoRA等）
        from peft import get_peft_model, LoraConfig
        
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
    
    elif args.tuner_backend == 'unsloth':
        # 使用Unsloth加速
        from unsloth import FastLanguageModel
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=args.lora_target_modules,
        )
    
    return model
```

**涉及参数**：
- `--tuner_backend`：微调后端（peft/unsloth/swift）
- `--tuner_type`：微调类型（lora/qlora/full等）
- `--lora_rank`：LoRA秩（通常8-64）
- `--lora_alpha`：LoRA缩放系数
- `--lora_target_modules`：应用LoRA的模块
- `--lora_dropout`：LoRA dropout率

#### 1.3.3.5 步骤5：训练执行

```python
def train(self, trainer):
    # 1. 开始训练
    trainer.train(trainer.args.resume_from_checkpoint)
    
    # 2. 保存训练状态
    self._save_trainer_state(trainer)
    
    # 3. 返回训练信息
    return self.train_msg
```

**训练器参数**（继承自Transformers的`TrainingArguments`）：
- `--num_train_epochs`：训练轮数
- `--per_device_train_batch_size`：每个设备的批次大小
- `--gradient_accumulation_steps`：梯度累积步数
- `--learning_rate`：学习率
- `--lr_scheduler_type`：学习率调度器类型
- `--warmup_ratio`：预热比例
- `--save_steps`：保存检查点的步数
- `--eval_steps`：评估的步数
- `--logging_steps`：日志记录步数

---

## 1.4 📊 四、关键参数详解

### 1.4.1 参数继承关系

```
SftArguments (sft_args.py)
    ├── SwanlabArguments          # SwanLab实验跟踪
    ├── TunerArguments            # 微调方法参数
    ├── BaseArguments             # 基础参数
    │   ├── ModelArguments        # 模型参数
    │   ├── DataArguments         # 数据参数
    │   ├── TemplateArguments     # 模板参数
    │   ├── QuantArguments        # 量化参数
    │   └── GenerationArguments   # 生成参数
    └── Seq2SeqTrainingArguments  # Transformers训练参数
```

### 1.4.2 必需参数

```bash
# 最小启动命令
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \  # 模型ID或路径
    --dataset alpaca-zh              # 数据集
```

### 1.4.3 常用参数组合

#### 1.4.3.1 (1) 基础LoRA微调

```bash
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset alpaca-zh \
    --tuner_type lora \              # 使用LoRA
    --lora_rank 8 \                  # LoRA秩
    --lora_alpha 32 \                # alpha系数
    --num_train_epochs 3 \           # 训练3轮
    --per_device_train_batch_size 1 \  # 批次大小
    --learning_rate 1e-4 \           # 学习率
    --gradient_accumulation_steps 16 \  # 梯度累积
    --output_dir output/qwen-lora    # 输出目录
```

**显存估算**：
- 7B模型 + LoRA：约需 **9-12GB** 显存
- 实际批次大小 = `batch_size × gradient_accumulation_steps` = 1 × 16 = 16

#### 1.4.3.2 (2) 量化训练（QLoRA）

```bash
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset alpaca-zh \
    --tuner_type qlora \             # 使用QLoRA
    --quantization_bit 4 \           # 4bit量化
    --lora_rank 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 3
```

**显存优势**：
- 4bit量化可降低显存约 **70%**
- 7B模型仅需 **6-8GB** 显存

#### 1.4.3.3 (3) 全参数微调

```bash
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset alpaca-zh \
    --tuner_type full \              # 全参数训练
    --deepspeed default-zero2 \      # 使用DeepSpeed ZeRO-2
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \           # 全参数学习率较小
    --num_train_epochs 1
```

**硬件要求**：
- 单卡至少 **40GB** 显存（A100）
- 推荐使用多卡 + DeepSpeed

#### 1.4.3.4 (4) 多模态模型微调

```bash
swift sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \  # 多模态模型
    --dataset coco-caption \         # 多模态数据集
    --tuner_type lora \
    --lora_target_modules ALL \      # 对所有模块应用LoRA
    --num_train_epochs 3
```

#### 1.4.3.5 (5) 长文本训练（序列并行）

```bash
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset long-context-data \
    --max_length 32768 \             # 32k上下文
    --sequence_parallel_size 2 \     # 序列并行
    --use_flash_attn true \          # Flash Attention
    --num_train_epochs 3
```

### 1.4.4 参数速查表

| 参数类别 | 关键参数 | 说明 | 默认值 |
|---------|---------|------|--------|
| **模型** | `--model` | 模型ID或路径 | - |
| | `--torch_dtype` | 模型精度 | auto |
| | `--model_type` | 模型类型 | auto |
| **数据** | `--dataset` | 训练数据集 | - |
| | `--val_dataset` | 验证数据集 | - |
| | `--max_length` | 最大序列长度 | 2048 |
| | `--truncation_strategy` | 截断策略 | delete |
| **微调** | `--tuner_type` | 微调类型 | lora |
| | `--lora_rank` | LoRA秩 | 8 |
| | `--lora_alpha` | LoRA alpha | 32 |
| | `--quantization_bit` | 量化位数 | 4 |
| **训练** | `--num_train_epochs` | 训练轮数 | 3 |
| | `--per_device_train_batch_size` | 批次大小 | 1 |
| | `--learning_rate` | 学习率 | 1e-4 |
| | `--gradient_accumulation_steps` | 梯度累积 | 16 |
| **优化** | `--deepspeed` | DeepSpeed配置 | - |
| | `--use_flash_attn` | Flash Attention | false |
| | `--gradient_checkpointing` | 梯度检查点 | true |
| **并行** | `--sequence_parallel_size` | 序列并行数 | 1 |
| | `--nproc_per_node` | 每节点GPU数 | 1 |

---

## 1.5 🔧 五、实战示例

### 1.5.1 快速开始：微调Qwen模型

```bash
# 1. 安装SWIFT
pip install ms-swift -U

# 2. 准备数据（使用内置数据集）
# alpaca-zh: 中文指令数据集

# 3. 开始训练
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset alpaca-zh \
    --num_train_epochs 3 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --learning_rate 1e-4 \
    --output_dir output/qwen-alpaca

# 4. 推理测试
swift infer \
    --adapters output/qwen-alpaca/xxx/checkpoint-xxx \
    --stream true
```

### 1.5.2 使用自定义数据集

**数据格式**（jsonl）：

```json
{"query": "什么是机器学习？", "response": "机器学习是人工智能的一个分支..."}
{"query": "如何学习Python？", "response": "学习Python可以从以下几个步骤开始..."}
```

**训练命令**：

```bash
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset my_data.jsonl \
    --dataset_test_ratio 0.01 \      # 1%用于验证
    --num_train_epochs 3
```

### 1.5.3 多卡训练

```bash
# 设置环境变量
export NPROC_PER_NODE=4  # 使用4张GPU

# 训练命令（自动使用torchrun）
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset alpaca-zh \
    --deepspeed default-zero2 \      # DeepSpeed ZeRO-2
    --num_train_epochs 3
```

### 1.5.4 使用配置文件

**创建配置文件** `config.yaml`：

```yaml
model: Qwen/Qwen2.5-7B-Instruct
dataset: alpaca-zh
tuner_type: lora
lora_rank: 8
lora_alpha: 32
num_train_epochs: 3
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 1e-4
output_dir: output/qwen-lora
```

**使用配置文件训练**：

```bash
swift sft --config config.yaml
```

---

## 1.6 🎓 六、核心概念解释

### 1.6.1 什么是LoRA？

**LoRA（Low-Rank Adaptation）**是一种参数高效的微调方法。

**原理**：

假设原始权重矩阵为 $W \in \mathbb{R}^{d \times k}$，LoRA通过两个低秩矩阵来近似更新：

$$
W' = W + \Delta W = W + BA
$$

其中：
- $B \in \mathbb{R}^{d \times r}$
- $A \in \mathbb{R}^{r \times k}$
- $r$ 是秩（rank），通常 $r \ll \min(d, k)$

**优势**：
- **显存占用低**：只训练 $BA$ 参数，参数量为 $r \times (d + k)$
- **训练速度快**：参数量减少约 **10000倍**
- **效果接近全参数**：在多数任务上效果相当

**关键参数**：
- `lora_rank (r)`：秩越大，表达能力越强，但显存占用越高
  - 推荐范围：8-64
- `lora_alpha`：缩放系数，通常设为 `rank × 2` 或 `rank × 4`
- `lora_target_modules`：应用LoRA的模块
  - 常见值：`['q_proj', 'k_proj', 'v_proj']`（仅注意力层）
  - 或 `ALL`（所有线性层）

### 1.6.2 什么是QLoRA？

**QLoRA = LoRA + 量化**

**核心思想**：
1. 将预训练模型量化为 **4bit**（INT4）
2. 在量化模型上应用LoRA微调

**优势**：
- **显存极低**：7B模型仅需 **6-8GB** 显存
- **精度损失小**：使用NF4（Normal Float 4）量化

**实现细节**：
```python
# 4bit量化配置
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # 计算时转回FP16
    bnb_4bit_use_double_quant=True,        # 二次量化
    bnb_4bit_quant_type="nf4"              # NF4量化
)
```

### 1.6.3 什么是Gradient Accumulation？

**梯度累积**允许在小批次上模拟大批次训练。

**原理**：

假设真实批次大小为 $B_{\text{real}}$，但显存只能容纳 $B_{\text{device}}$：

$$
B_{\text{real}} = B_{\text{device}} \times N_{\text{accum}}
$$

**训练流程**：
1. 前向传播 $N_{\text{accum}}$ 次，累积梯度
2. 反向传播一次，更新参数
3. 清零梯度，重复

**示例**：
```bash
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16
```
等效于批次大小为 **16**

### 1.6.4 什么是序列并行？

**序列并行**将长序列切分到多个设备上处理。

**Ulysses序列并行**：

假设序列长度为 $L$，设备数为 $N$：

1. 将序列分割：$L_i = L / N$
2. 每个设备处理 $L_i$ 长度
3. 通过AllGather同步结果

**优势**：
- 支持 **超长上下文**（32k+）
- 显存占用降低 $N$ 倍

**使用示例**：
```bash
--max_length 32768 \
--sequence_parallel_size 2  # 使用2个设备并行
```

---

## 1.7 🧩 七、高级功能

### 1.7.1 数据Packing

**Packing**将多个短样本拼接成一个批次，提升训练效率。

**启用方式**：
```bash
--packing true \
--packing_length 4096  # 拼接后的最大长度
```

**适用场景**：
- 数据集样本长度差异大
- 存在大量短样本

### 1.7.2 Flash Attention

**Flash Attention**是优化的注意力计算，降低显存和提升速度。

**启用方式**：
```bash
--use_flash_attn true
```

**优势**：
- 显存降低 **2-4倍**
- 速度提升 **2-3倍**

### 1.7.3 DeepSpeed

**DeepSpeed ZeRO**是微软开发的显存优化技术。

**可用配置**：
```bash
--deepspeed default-zero2  # ZeRO-2: 优化器状态分片
--deepspeed default-zero3  # ZeRO-3: 优化器+梯度+参数分片
```

**ZeRO各阶段对比**：

| 阶段 | 优化内容 | 显存节省 | 通信开销 |
|------|---------|---------|---------|
| ZeRO-1 | 优化器状态 | 4倍 | 低 |
| ZeRO-2 | 优化器+梯度 | 8倍 | 中 |
| ZeRO-3 | 优化器+梯度+参数 | N倍 | 高 |

### 1.7.4 RLHF训练

**强化学习人类反馈**（RLHF）用于对齐模型。

**GRPO（Group Relative Policy Optimization）**：

```bash
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset hh-rlhf-helpful \
    --num_train_epochs 1
```

**支持的算法**：
- GRPO、DAPO、GSPO
- DPO、KTO、CPO
- SimPO、ORPO

---

## 1.8 📝 八、常见问题

### 1.8.1 Q1: 如何选择LoRA rank？

**推荐策略**：
- **小模型（<3B）**：rank=4-8
- **中模型（7B-13B）**：rank=8-16
- **大模型（>30B）**：rank=16-64

### 1.8.2 Q2: 显存不足怎么办？

**优化方案**：
1. **使用量化**：`--quantization_bit 4`
2. **梯度检查点**：`--gradient_checkpointing true`（默认开启）
3. **减小批次**：`--per_device_train_batch_size 1`
4. **增加梯度累积**：`--gradient_accumulation_steps 32`
5. **使用DeepSpeed**：`--deepspeed default-zero3`

### 1.8.3 Q3: 训练速度慢怎么办？

**加速方案**：
1. **Flash Attention**：`--use_flash_attn true`
2. **数据Packing**：`--packing true`
3. **减少日志**：`--logging_steps 100`
4. **多卡训练**：`export NPROC_PER_NODE=4`

### 1.8.4 Q4: 如何查看训练进度？

**方法1：TensorBoard**
```bash
tensorboard --logdir output/xxx/runs
```

**方法2：SwanLab**
```bash
--report_to swanlab \
--swanlab_project my-project
```

### 1.8.5 Q5: 如何恢复训练？

```bash
swift sft \
    --resume_from_checkpoint output/xxx/checkpoint-100 \
    # 其他参数保持一致
```

---

## 1.9 🎯 九、总结

### 1.9.1 核心要点回顾

1. **入口**：`swift sft` → `cli/main.py` → `pipelines/train/sft.py`
2. **流程**：模型加载 → 模板准备 → 数据处理 → 微调 → 训练
3. **必需参数**：`--model` 和 `--dataset`
4. **推荐配置**：LoRA + 梯度累积 + Flash Attention

### 1.9.2 学习路径建议

**初学者**：
1. 使用内置数据集快速体验
2. 理解基础参数（model、dataset、lora_rank）
3. 掌握单卡LoRA训练

**进阶用户**：
1. 学习自定义数据集
2. 尝试多卡训练和DeepSpeed
3. 了解量化训练（QLoRA）

**高级用户**：
1. 深入序列并行和长文本训练
2. 探索RLHF和强化学习
3. 使用Megatron进行大规模训练

---

## 1.10 📚 参考资源

- **官方文档**：https://swift.readthedocs.io/
- **GitHub仓库**：https://github.com/modelscope/ms-swift
- **论文**：https://arxiv.org/abs/2408.05517
- **示例脚本**：`examples/train/`目录

---

**祝你学习顺利！**🚀
