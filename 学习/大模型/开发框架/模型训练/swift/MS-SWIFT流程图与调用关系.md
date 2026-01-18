# 1 MS-SWIFT 流程图与代码调用关系

## 1.1 一、整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                       用户命令行                              │
│              swift sft --model xxx --dataset yyy             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    CLI入口层                                  │
│              swift/cli/main.py::cli_main()                   │
│              • 解析命令                                       │
│              • 路由分发                                       │
│              • 判断是否使用torchrun                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  命令入口层                                   │
│              swift/cli/sft.py::main                          │
│              • try_init_unsloth()                            │
│              • try_init_ray()                                │
│              • sft_main()                                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Pipeline层                                   │
│         swift/pipelines/train/sft.py::SwiftSft               │
│              ┌──────────────────────────────┐                │
│              │ __init__()                   │                │
│              │  • _prepare_model_tokenizer()│                │
│              │  • _prepare_template()       │                │
│              │  • _prepare_flash_ckpt()     │                │
│              └──────────────────────────────┘                │
│              ┌──────────────────────────────┐                │
│              │ run()                        │                │
│              │  • _prepare_dataset()        │                │
│              │  • prepare_model() [LoRA]    │                │
│              │  • create_trainer()          │                │
│              │  • train()                   │                │
│              └──────────────────────────────┘                │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
   ┌─────────┐    ┌──────────┐    ┌──────────┐
   │ 模型层   │    │  数据层   │    │ 训练器层  │
   └─────────┘    └──────────┘    └──────────┘
```

## 1.2 二、SFT训练详细流程图

```
用户执行：swift sft --model Qwen/Qwen2.5-7B --dataset alpaca-zh
│
├─ 步骤1: CLI解析与路由
│  └─ swift/cli/main.py::cli_main()
│     ├─ 解析命令行参数
│     ├─ 读取--config（如果有）
│     ├─ 路由到 swift/cli/sft.py
│     └─ 判断是否使用torchrun
│
├─ 步骤2: 初始化Pipeline
│  └─ swift/pipelines/train/sft.py::SwiftSft.__init__()
│     │
│     ├─ 2.1 加载模型和分词器
│     │  └─ _prepare_model_tokenizer()
│     │     ├─ args.get_model_processor()
│     │     │  └─ swift/arguments/base_args/model_args.py
│     │     │     ├─ 从ModelScope/HuggingFace加载模型
│     │     │     ├─ 应用torch_dtype（fp16/bf16）
│     │     │     ├─ 应用量化配置（如果启用）
│     │     │     └─ 返回 (model, tokenizer)
│     │     │
│     │     ├─ 序列并行处理（如果启用）
│     │     │  └─ swift/sequence_parallel/prepare()
│     │     │
│     │     └─ _prepare_generation_config()
│     │        └─ 配置生成参数（temperature, max_new_tokens等）
│     │
│     ├─ 2.2 准备对话模板
│     │  └─ _prepare_template()
│     │     └─ args.get_template()
│     │        └─ swift/template/目录
│     │           ├─ 根据model_type选择模板
│     │           ├─ 设置system prompt
│     │           └─ 设置为训练模式
│     │
│     └─ 2.3 准备Flash Checkpoint
│        └─ _prepare_flash_ckpt()
│           └─ 如果use_flash_ckpt，导入dlrover
│
├─ 步骤3: 执行训练
│  └─ SwiftSft.run()
│     │
│     ├─ 3.1 准备数据集
│     │  └─ _prepare_dataset()
│     │     │
│     │     ├─ 3.1.1 加载数据集
│     │     │  └─ _get_dataset()
│     │     │     └─ swift/dataset/load_dataset()
│     │     │        ├─ 内置数据集：从预定义映射加载
│     │     │        ├─ 自定义数据集：从文件加载
│     │     │        └─ 返回 (train_dataset, val_dataset)
│     │     │
│     │     ├─ 3.1.2 编码数据集
│     │     │  └─ _encode_dataset()
│     │     │     │
│     │     │     ├─ 如果lazy_tokenize=False
│     │     │     │  └─ EncodePreprocessor(template).process()
│     │     │     │     ├─ 并行处理（dataset_num_proc进程）
│     │     │     │     ├─ 应用template.encode()
│     │     │     │     │  └─ 将对话转为token_ids
│     │     │     │     │     ├─ query + response → 模板格式
│     │     │     │     │     ├─ tokenizer编码
│     │     │     │     │     └─ 截断到max_length
│     │     │     │     └─ 计算长度统计
│     │     │     │
│     │     │     └─ 如果lazy_tokenize=True
│     │     │        └─ 仅添加长度信息，延迟编码
│     │     │
│     │     ├─ 3.1.3 后处理数据集
│     │     │  └─ _post_process_datasets()
│     │     │     │
│     │     │     ├─ 如果packing=True
│     │     │     │  └─ PackingDataset包装
│     │     │     │     ├─ 将多个短样本拼接
│     │     │     │     └─ 优化填充效率
│     │     │     │
│     │     │     ├─ 如果lazy_tokenize=True
│     │     │     │  └─ LazyLLMDataset包装
│     │     │     │     └─ 在__getitem__时编码
│     │     │     │
│     │     │     └─ 如果streaming=True
│     │     │        └─ IterableDataset处理
│     │     │
│     │     └─ 3.1.4 显示数据集样例
│     │        └─ _show_dataset()
│     │           ├─ 打印第一个样本
│     │           └─ 统计长度分布
│     │
│     ├─ 3.2 应用微调方法
│     │  └─ prepare_model()
│     │     └─ swift/pipelines/train/tuner.py::TunerMixin
│     │        │
│     │        ├─ 如果tuner_type='lora'
│     │        │  └─ swift/tuners/lora/
│     │        │     ├─ 创建LoraConfig
│     │        │     │  ├─ r=lora_rank
│     │        │     │  ├─ lora_alpha
│     │        │     │  ├─ target_modules
│     │        │     │  └─ lora_dropout
│     │        │     ├─ get_peft_model(model, config)
│     │        │     └─ 冻结原始权重，仅训练LoRA
│     │        │
│     │        ├─ 如果tuner_type='qlora'
│     │        │  └─ 量化模型 + LoRA
│     │        │     ├─ BitsAndBytesConfig(4bit)
│     │        │     └─ 应用LoRA
│     │        │
│     │        └─ 如果tuner_type='full'
│     │           └─ 返回原始模型（全参数训练）
│     │
│     ├─ 3.3 创建训练器
│     │  └─ TrainerFactory.get_trainer_cls()
│     │     └─ swift/trainers/
│     │        ├─ 根据task_type选择Trainer
│     │        │  ├─ causal_lm → Seq2SeqTrainer
│     │        │  ├─ seq_cls → SequenceClassificationTrainer
│     │        │  └─ embedding → EmbeddingTrainer
│     │        │
│     │        └─ 初始化Trainer
│     │           ├─ model（应用LoRA后的模型）
│     │           ├─ args（TrainingArguments）
│     │           ├─ train_dataset
│     │           ├─ eval_dataset
│     │           ├─ data_collator（数据整理器）
│     │           └─ callbacks（回调函数）
│     │
│     └─ 3.4 开始训练
│        └─ train(trainer)
│           │
│           ├─ trainer.train()
│           │  └─ transformers.Trainer的训练循环
│           │     │
│           │     ├─ 训练循环
│           │     │  └─ for epoch in range(num_epochs):
│           │     │     └─ for batch in dataloader:
│           │     │        ├─ forward pass
│           │     │        │  └─ model(batch)
│           │     │        ├─ compute loss
│           │     │        │  └─ 交叉熵损失
│           │     │        ├─ backward pass
│           │     │        │  └─ loss.backward()
│           │     │        ├─ gradient accumulation
│           │     │        │  └─ 累积N步后更新
│           │     │        ├─ optimizer step
│           │     │        │  └─ AdamW.step()
│           │     │        ├─ lr scheduler step
│           │     │        │  └─ 调整学习率
│           │     │        └─ logging
│           │     │           └─ TensorBoard/SwanLab
│           │     │
│           │     ├─ 评估循环（每eval_steps）
│           │     │  └─ for batch in val_dataloader:
│           │     │     ├─ forward pass（无梯度）
│           │     │     ├─ compute metrics
│           │     │     └─ 记录验证loss
│           │     │
│           │     └─ 保存检查点（每save_steps）
│           │        ├─ 保存模型权重
│           │        ├─ 保存优化器状态
│           │        └─ 保存训练状态
│           │
│           └─ _save_trainer_state()
│              ├─ 保存最佳检查点
│              ├─ 保存最后检查点
│              ├─ 可视化训练曲线
│              └─ 记录训练日志
│
└─ 步骤4: 训练完成
   └─ 返回训练信息
      ├─ last_model_checkpoint
      ├─ best_model_checkpoint
      ├─ best_metric
      └─ log_history
```

## 1.3 三、数据流转图

```
原始数据
   │
   │ {"query": "你好", "response": "你好！有什么可以帮助你的吗？"}
   │
   ▼
┌────────────────────────────────────────┐
│  数据加载（DatasetLoader）              │
│  swift/dataset/loader.py               │
│  └─ 支持格式：jsonl, csv, 内置数据集   │
└─────────────────┬──────────────────────┘
                  │
                  │ HuggingFace Dataset对象
                  │
                  ▼
┌────────────────────────────────────────┐
│  模板处理（Template）                   │
│  swift/template/xxx_template.py        │
│  └─ 将对话转为模型格式                 │
│     例如：                              │
│     <|im_start|>user                   │
│     你好<|im_end|>                     │
│     <|im_start|>assistant              │
│     你好！有什么可以帮助你的吗？<|im_end|>│
└─────────────────┬──────────────────────┘
                  │
                  │ 格式化的文本
                  │
                  ▼
┌────────────────────────────────────────┐
│  分词编码（Tokenizer）                  │
│  └─ tokenizer.encode()                 │
│     文本 → token_ids                   │
│     [101, 872, 543, ..., 102]          │
└─────────────────┬──────────────────────┘
                  │
                  │ token_ids + labels
                  │
                  ▼
┌────────────────────────────────────────┐
│  数据整理（DataCollator）               │
│  └─ 批次padding、创建attention_mask    │
│     batch = {                          │
│       'input_ids': [...],              │
│       'attention_mask': [...],         │
│       'labels': [...]                  │
│     }                                  │
└─────────────────┬──────────────────────┘
                  │
                  │ 批次数据
                  │
                  ▼
┌────────────────────────────────────────┐
│  模型前向传播                           │
│  └─ model(**batch)                     │
│     └─ 计算loss                        │
└─────────────────┬──────────────────────┘
                  │
                  │ loss值
                  │
                  ▼
┌────────────────────────────────────────┐
│  反向传播与优化                         │
│  └─ loss.backward()                    │
│     └─ optimizer.step()                │
└────────────────────────────────────────┘
```

## 1.4 四、参数继承与处理流程

```
命令行参数
   │ swift sft --model xxx --dataset yyy --lora_rank 8
   │
   ▼
┌─────────────────────────────────────────────────────┐
│  解析为SftArguments对象                              │
│  swift/arguments/sft_args.py                        │
│                                                     │
│  继承关系：                                          │
│  SftArguments                                       │
│   ├─ SwanlabArguments (实验跟踪)                    │
│   ├─ TunerArguments (微调方法)                      │
│   │   ├─ tuner_type                                 │
│   │   ├─ lora_rank                                  │
│   │   ├─ lora_alpha                                 │
│   │   └─ lora_target_modules                        │
│   ├─ BaseArguments                                  │
│   │   ├─ ModelArguments (模型参数)                  │
│   │   │   ├─ model                                  │
│   │   │   ├─ model_type                             │
│   │   │   └─ torch_dtype                            │
│   │   ├─ DataArguments (数据参数)                   │
│   │   │   ├─ dataset                                │
│   │   │   ├─ val_dataset                            │
│   │   │   └─ max_length                             │
│   │   ├─ TemplateArguments (模板参数)               │
│   │   │   ├─ template                               │
│   │   │   └─ system                                 │
│   │   ├─ QuantArguments (量化参数)                  │
│   │   │   └─ quantization_bit                       │
│   │   └─ GenerationArguments (生成参数)             │
│   │       ├─ temperature                            │
│   │       └─ max_new_tokens                         │
│   └─ Seq2SeqTrainingArguments (训练参数)            │
│       ├─ num_train_epochs                           │
│       ├─ per_device_train_batch_size                │
│       ├─ learning_rate                              │
│       ├─ gradient_accumulation_steps                │
│       └─ output_dir                                 │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
         参数后处理（__post_init__）
                   │
                   ├─ 补全默认值
                   │  └─ output_dir = f'output/{model_name}'
                   │
                   ├─ 参数验证
                   │  └─ 检查参数合法性
                   │
                   └─ 衍生参数
                      ├─ training_args (TrainingArguments对象)
                      ├─ model_kwargs (模型加载参数)
                      └─ dataset_kwargs (数据加载参数)
```

## 1.5 五、LoRA应用流程详解

```
原始模型（Qwen2.5-7B）
   │ parameters: 7B
   │
   ▼
┌──────────────────────────────────────┐
│  1. 加载模型                          │
│  args.get_model_processor()          │
│  └─ model = AutoModelForCausalLM(...) │
└─────────────────┬────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────┐
│  2. 创建LoRA配置                      │
│  LoraConfig(                         │
│    r=8,                  # 秩        │
│    lora_alpha=32,        # 缩放      │
│    target_modules=['q_proj',         │
│                    'k_proj',         │
│                    'v_proj'],        │
│    lora_dropout=0.05,                │
│    task_type="CAUSAL_LM"             │
│  )                                   │
└─────────────────┬────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────┐
│  3. 应用LoRA                          │
│  get_peft_model(model, lora_config)  │
│                                      │
│  对每个target_module:                │
│                                      │
│  原始层：Linear(in=4096, out=4096)   │
│           ↓                          │
│  替换为：                             │
│  ┌────────────────────────────────┐ │
│  │ 原始权重W (冻结)                │ │
│  │    4096 × 4096                 │ │
│  └────────────────────────────────┘ │
│           +                          │
│  ┌────────────────────────────────┐ │
│  │ LoRA权重                       │ │
│  │ B (4096×8) × A (8×4096)        │ │
│  │ 可训练参数：8×(4096+4096)=65536│ │
│  └────────────────────────────────┘ │
│                                      │
│  前向传播：                           │
│  output = W·x + (B·A)·x              │
└─────────────────┬────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────┐
│  4. 统计参数量                        │
│  原始模型：7B参数（全冻结）            │
│  LoRA参数：约9M参数（可训练）         │
│  占比：0.13%                          │
└──────────────────────────────────────┘
```

## 1.6 六、关键类与方法调用图

```
SwiftSft (主类)
├── __init__()
│   ├── _prepare_model_tokenizer()
│   │   ├── ModelArguments.get_model_processor()
│   │   │   ├── ModelScope.from_pretrained()
│   │   │   └── AutoModel.from_pretrained()
│   │   ├── sequence_parallel.prepare()
│   │   └── _prepare_generation_config()
│   ├── _prepare_template()
│   │   └── TemplateArguments.get_template()
│   │       └─ TemplateFactory.get_template()
│   └── _prepare_flash_ckpt()
│
└── run()
    ├── _prepare_dataset()
    │   ├── _get_dataset()
    │   │   └── DatasetLoader.load_dataset()
    │   │       ├── 内置数据集 → DATASET_MAPPING
    │   │       └── 自定义数据集 → from_jsonl/csv
    │   ├── _encode_dataset()
    │   │   └── EncodePreprocessor.process()
    │   │       └── Template.encode()
    │   └── _post_process_datasets()
    │       ├── LazyLLMDataset (lazy_tokenize)
    │       ├── PackingDataset (packing)
    │       └── IterableDataset (streaming)
    │
    ├── prepare_model() [TunerMixin]
    │   └── TunerFactory.get_tuner()
    │       ├── LoRA → swift/tuners/lora/
    │       ├── QLoRA → swift/tuners/qlora/
    │       └── Full → 返回原模型
    │
    ├── TrainerFactory.get_trainer_cls()
    │   └── 根据task_type选择
    │       ├── Seq2SeqTrainer
    │       ├── SequenceClassificationTrainer
    │       └── EmbeddingTrainer
    │
    └── train()
        ├── Trainer.train()
        │   └── transformers训练循环
        └── _save_trainer_state()
            ├── 保存检查点
            └── 可视化
```

## 1.7 七、多卡训练流程

```
用户命令：
export NPROC_PER_NODE=4
swift sft --model xxx --dataset yyy --deepspeed default-zero2
│
├─ CLI检测环境变量
│  └─ get_torchrun_args()
│     └─ 检测到NPROC_PER_NODE=4
│
├─ 使用torchrun启动
│  └─ python -m torch.distributed.run \
│        --nproc_per_node=4 \
│        --master_port=29500 \
│        swift/cli/sft.py [其他参数]
│
├─ 初始化进程组
│  └─ torch.distributed.init_process_group()
│     ├─ rank 0: GPU 0（主进程）
│     ├─ rank 1: GPU 1
│     ├─ rank 2: GPU 2
│     └─ rank 3: GPU 3
│
├─ DeepSpeed初始化
│  └─ deepspeed.initialize()
│     ├─ ZeRO-2配置
│     │  ├─ 优化器状态分片（每GPU存1/4）
│     │  └─ 梯度分片（每GPU存1/4）
│     └─ 模型复制到所有GPU
│
├─ 数据并行
│  └─ DistributedSampler
│     ├─ GPU 0: 样本 0, 4, 8, 12, ...
│     ├─ GPU 1: 样本 1, 5, 9, 13, ...
│     ├─ GPU 2: 样本 2, 6, 10, 14, ...
│     └─ GPU 3: 样本 3, 7, 11, 15, ...
│
├─ 训练循环
│  └─ for batch in dataloader:
│     ├─ 前向传播（各GPU独立）
│     ├─ 计算loss（各GPU独立）
│     ├─ 反向传播（各GPU独立）
│     ├─ AllReduce梯度
│     │  └─ 同步所有GPU的梯度
│     └─ 更新参数（各GPU同步）
│
└─ 保存（仅rank 0）
   └─ 保存模型和检查点
```

## 1.8 八、关键文件速查表

| 功能 | 文件路径 | 说明 |
|-----|---------|------|
| **入口** | `swift/cli/main.py` | CLI主入口，命令路由 |
| **SFT主流程** | `swift/pipelines/train/sft.py` | SwiftSft类，训练主逻辑 |
| **参数定义** | `swift/arguments/sft_args.py` | SftArguments参数类 |
| **模型加载** | `swift/arguments/base_args/model_args.py` | 模型加载逻辑 |
| **数据加载** | `swift/dataset/loader.py` | 数据集加载 |
| **对话模板** | `swift/template/` | 各模型的对话模板 |
| **LoRA** | `swift/tuners/lora/` | LoRA实现 |
| **训练器** | `swift/trainers/` | 各种Trainer实现 |
| **推理引擎** | `swift/infer_engine/` | vLLM等推理引擎 |
| **RLHF** | `swift/rlhf_trainers/` | GRPO等强化学习 |
| **Megatron** | `swift/megatron/` | Megatron并行 |

## 1.9 九、调试技巧

### 1.9.1 打印模型结构

```python
# 在SwiftSft.__init__后添加
print(self.model)
```

### 1.9.2 查看数据样例

```python
# 在_prepare_dataset后添加
dataset = train_dataset[0]
print(f"Input IDs: {dataset['input_ids']}")
print(f"Labels: {dataset['labels']}")
```

### 1.9.3 监控显存

```bash
# 训练时监控
watch -n 1 nvidia-smi
```

### 1.9.4 设置断点

```python
# 在关键位置
import pdb; pdb.set_trace()
```

### 1.9.5 启用详细日志

```bash
export SWIFT_LOG_LEVEL=DEBUG
swift sft [参数...]
```

---

**这份文档帮助你理解MS-SWIFT的整体架构和数据流转！**
