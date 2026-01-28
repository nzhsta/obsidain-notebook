# 代码调用关系分析文档

## 📊 整体架构

本框架采用**分层架构**设计，主要分为以下几层：

1. **配置层** - ModelConfig
2. **模板层** - PromptTemplate  
3. **客户端层** - VLLMClient
4. **抽象层** - BaseChain
5. **实现层** - LLMChain, SequentialChain
6. **工具层** - OutputParser, create_chain()

---

## 🏗️ 类层次结构

```
BaseChain (抽象基类)
├── LLMChain (LLM调用链)
└── SequentialChain (顺序链)

ModelConfig (数据类)
PromptTemplate (模板类)
VLLMClient (客户端类)
OutputParser (工具类)
```

---

## 🔗 详细调用关系

### 1️⃣ 基础调用流程

#### 场景A: 直接调用LLM

```
用户代码
  └─> VLLMClient.invoke(prompt)
        ├─> ModelConfig.to_dict()  # 获取配置
        ├─> OpenAI.chat.completions.create()  # 调用API
        └─> 返回响应文本
```

**调用链:**
```python
llm = VLLMClient(config)
response = llm.invoke("你好")

# 内部调用:
# 1. self.config.to_dict() -> 获取模型参数
# 2. self.client.chat.completions.create() -> OpenAI SDK调用
# 3. response.choices[0].message.content -> 提取响应
```

---

#### 场景B: 异步调用

```
用户代码
  └─> await VLLMClient.ainvoke(prompt)
        ├─> ModelConfig.to_dict()
        ├─> await AsyncOpenAI.chat.completions.create()
        └─> 返回响应文本
```

---

### 2️⃣ 模板调用流程

#### 场景C: 使用提示词模板

```
用户代码
  └─> LLMChain.run(**kwargs)
        ├─> PromptTemplate.format(**kwargs)  # 格式化模板
        │     └─> template.format()  # Python内置format
        ├─> VLLMClient.invoke(formatted_prompt)  # 调用LLM
        │     ├─> ModelConfig.to_dict()
        │     └─> OpenAI.chat.completions.create()
        └─> OutputParser(response)  # 解析输出(可选)
              └─> 返回解析后的结果
```

**完整调用链:**
```python
# 1. 创建模板
template = PromptTemplate("翻译: {text}")
  └─> PromptTemplate.__init__()
        └─> _extract_variables()  # 提取变量名 {text}

# 2. 创建链
chain = LLMChain(llm, template, output_parser)
  └─> LLMChain.__init__()
        └─> 存储 llm, prompt, output_parser

# 3. 运行链
result = chain.run(text="Hello")
  └─> LLMChain.run()
        ├─> self.prompt.format(text="Hello")  # "翻译: Hello"
        ├─> self.llm.invoke("翻译: Hello")
        │     └─> [API调用]
        └─> self.output_parser(response)  # 解析输出
```

---

### 3️⃣ 并发调用流程

#### 场景D: 批量异步并发

```
用户代码
  └─> await VLLMClient.abatch(prompts, concurrency=5)
        ├─> asyncio.Semaphore(5)  # 创建信号量
        ├─> 创建多个任务: [_invoke_with_sem(p) for p in prompts]
        │     └─> 每个任务内部:
        │           ├─> async with semaphore:  # 获取许可
        │           └─> await self.ainvoke(prompt)
        │                 └─> [异步API调用]
        └─> await asyncio.gather(*tasks)  # 并发执行
              └─> 返回所有结果列表
```

**调用时序:**
```
Time ─────────────────────────────────────>

任务1: [─获取信号量─][─────API调用─────][完成]
任务2: [─获取信号量─][─────API调用─────][完成]
任务3: [─获取信号量─][─────API调用─────][完成]
任务4: [─获取信号量─][─────API调用─────][完成]
任务5: [─获取信号量─][─────API调用─────][完成]
任务6: [等待信号量...][─获取─][─API调用─][完成]
任务7: [等待信号量...][─获取─][─API调用─][完成]
```

---

#### 场景E: 批量线程池并发

```
用户代码
  └─> VLLMClient.batch(prompts, max_workers=5)
        ├─> ThreadPoolExecutor(max_workers=5)
        ├─> [executor.submit(invoke, p) for p in prompts]
        │     └─> 每个线程:
        │           └─> self.invoke(prompt)
        │                 └─> [API调用]
        └─> [future.result() for future in as_completed()]
              └─> 返回所有结果
```

---

### 4️⃣ 链式组合流程

#### 场景F: 顺序链

```
用户代码
  └─> SequentialChain.run(**kwargs)
        ├─> 初始化 result = kwargs
        ├─> for chain in self.chains:
        │     ├─> result = chain.run(**result)
        │     │     └─> [执行单个链的逻辑]
        │     └─> 将结果传递给下一个链
        └─> 返回最终结果
```

**示例:**
```python
chain1 = LLMChain(llm, "生成大纲: {topic}")
chain2 = LLMChain(llm, "扩展: {outline}")
seq_chain = SequentialChain([chain1, chain2])

result = seq_chain.run(topic="Python")

# 调用流程:
# 1. chain1.run(topic="Python") -> outline
# 2. chain2.run(outline=outline) -> expanded_content
# 3. 返回 expanded_content
```

---

## 📋 方法调用矩阵

| 调用者 | 被调用方法 | 目的 |
|--------|-----------|------|
| **用户** | `VLLMClient.invoke()` | 同步单次调用 |
| **用户** | `VLLMClient.ainvoke()` | 异步单次调用 |
| **用户** | `VLLMClient.stream()` | 同步流式调用 |
| **用户** | `VLLMClient.astream()` | 异步流式调用 |
| **用户** | `VLLMClient.batch()` | 线程池批量调用 |
| **用户** | `VLLMClient.abatch()` | 异步批量调用 |
| `VLLMClient.__init__()` | `ModelConfig()` | 获取配置 |
| `VLLMClient.invoke()` | `ModelConfig.to_dict()` | 转换配置为字典 |
| `VLLMClient.invoke()` | `OpenAI.chat.completions.create()` | 调用API |
| `VLLMClient.ainvoke()` | `AsyncOpenAI.chat.completions.create()` | 异步调用API |
| `VLLMClient.batch()` | `ThreadPoolExecutor.submit()` | 提交线程任务 |
| `VLLMClient.abatch()` | `asyncio.Semaphore()` | 并发控制 |
| `VLLMClient.abatch()` | `asyncio.gather()` | 收集异步结果 |
| **用户** | `LLMChain.run()` | 同步运行链 |
| **用户** | `LLMChain.arun()` | 异步运行链 |
| `LLMChain.__init__()` | `PromptTemplate()` | 创建模板 |
| `LLMChain.run()` | `PromptTemplate.format()` | 格式化提示词 |
| `LLMChain.run()` | `VLLMClient.invoke()` | 调用LLM |
| `LLMChain.run()` | `output_parser()` | 解析输出 |
| `PromptTemplate.__init__()` | `_extract_variables()` | 提取变量 |
| `PromptTemplate.format()` | `str.format()` | Python格式化 |
| **用户** | `create_chain()` | 快速创建链 |
| `create_chain()` | `PromptTemplate()` | 创建模板 |
| `create_chain()` | `LLMChain()` | 创建链 |
| **用户** | `OutputParser.json_parser()` | 解析JSON |
| **用户** | `OutputParser.list_parser()` | 解析列表 |
| **用户** | `OutputParser.number_parser()` | 解析数字 |

---

## 🔄 数据流向

### 基础调用数据流

```
用户输入 (prompt)
    ↓
VLLMClient.invoke()
    ↓
ModelConfig.to_dict() → {"model": "...", "temperature": 0.7, ...}
    ↓
OpenAI SDK → HTTP请求 → vLLM服务器
    ↓
vLLM服务器 → 生成响应 → HTTP响应
    ↓
OpenAI SDK → ChatCompletion对象
    ↓
response.choices[0].message.content
    ↓
返回给用户 (str)
```

### 链式调用数据流

```
用户输入 (**kwargs)
    ↓
LLMChain.run(**kwargs)
    ↓
PromptTemplate.format(**kwargs) → formatted_prompt (str)
    ↓
VLLMClient.invoke(formatted_prompt) → response (str)
    ↓
output_parser(response) → parsed_result (Any)
    ↓
返回给用户
```

### 并发调用数据流

```
用户输入 (List[prompt])
    ↓
VLLMClient.abatch(prompts)
    ↓
创建多个协程任务
    ├─> Task1: ainvoke(prompt1) ──┐
    ├─> Task2: ainvoke(prompt2) ──┤
    ├─> Task3: ainvoke(prompt3) ──┤ → asyncio.Semaphore控制并发
    ├─> Task4: ainvoke(prompt4) ──┤
    └─> Task5: ainvoke(prompt5) ──┘
           ↓
    asyncio.gather(*tasks)
           ↓
    List[response1, response2, ...]
           ↓
    返回给用户 (List[str])
```

---

## 🎯 关键调用场景

### 场景1: 简单问答

```python
llm = VLLMClient(ModelConfig(model="Qwen"))
response = llm.invoke("你好")
```

**调用栈:**
```
main()
  └─> llm.invoke("你好")
        ├─> config.to_dict()
        └─> client.chat.completions.create()
```

---

### 场景2: 模板化调用

```python
template = PromptTemplate("翻译: {text}")
chain = LLMChain(llm, template)
result = chain.run(text="Hello")
```

**调用栈:**
```
main()
  └─> chain.run(text="Hello")
        ├─> prompt.format(text="Hello")
        ├─> llm.invoke("翻译: Hello")
        │     └─> [API调用]
        └─> output_parser(response)
```

---

### 场景3: 并发批处理

```python
prompts = ["问题1", "问题2", "问题3"]
results = await llm.abatch(prompts, concurrency=2)
```

**调用栈:**
```
main()
  └─> await llm.abatch(prompts)
        ├─> Semaphore(2)
        ├─> _invoke_with_sem("问题1")
        │     └─> await ainvoke("问题1")
        ├─> _invoke_with_sem("问题2")
        │     └─> await ainvoke("问题2")
        ├─> _invoke_with_sem("问题3")
        │     └─> await ainvoke("问题3")
        └─> await gather(task1, task2, task3)
```

---

### 场景4: 链式处理

```python
chain1 = create_chain(llm, "大纲: {topic}")
chain2 = create_chain(llm, "扩展: {outline}")
seq = SequentialChain([chain1, chain2])
result = seq.run(topic="AI")
```

**调用栈:**
```
main()
  └─> seq.run(topic="AI")
        ├─> chain1.run(topic="AI")
        │     ├─> format(topic="AI")
        │     └─> invoke("大纲: AI")
        │           └─> outline
        └─> chain2.run(outline=outline)
              ├─> format(outline=outline)
              └─> invoke("扩展: ...")
                    └─> expanded_content
```

---

## 🧩 依赖关系图

```
langchain_vllm.py
├── openai (OpenAI, AsyncOpenAI)
├── asyncio
├── concurrent.futures (ThreadPoolExecutor)
├── typing (类型注解)
├── dataclasses (@dataclass)
├── abc (ABC, abstractmethod)
└── re (正则表达式)

用户代码
├── langchain_vllm (导入核心类)
└── asyncio (异步执行)
```

---

## 📊 方法复杂度分析

| 方法 | 复杂度 | 说明 |
|------|--------|------|
| `ModelConfig.to_dict()` | O(1) | 简单字典构建 |
| `PromptTemplate._extract_variables()` | O(n) | 正则匹配，n为模板长度 |
| `PromptTemplate.format()` | O(n) | 字符串格式化 |
| `VLLMClient.invoke()` | O(1) | 单次API调用 |
| `VLLMClient.stream()` | O(n) | 流式输出，n为响应长度 |
| `VLLMClient.batch()` | O(n/w) | n个任务，w个worker |
| `VLLMClient.abatch()` | O(n/c) | n个任务，c为并发度 |
| `LLMChain.run()` | O(1) | 单次调用 |
| `SequentialChain.run()` | O(k) | k为链数量 |
| `OutputParser.json_parser()` | O(n) | JSON解析 |

---

## 🎨 设计模式识别

| 模式 | 应用位置 | 说明 |
|------|---------|------|
| **模板方法** | `BaseChain` | 定义抽象方法`run()`, `arun()` |
| **策略模式** | `output_parser` | 可插拔的输出解析策略 |
| **建造者模式** | `create_chain()` | 便捷的链构建函数 |
| **装饰器模式** | `LLMChain` | 包装`VLLMClient`添加模板功能 |
| **组合模式** | `SequentialChain` | 组合多个链 |
| **单例模式** | `VLLMClient` | 复用OpenAI客户端实例 |
| **工厂模式** | `create_chain()` | 创建链的工厂函数 |

---

## 📈 性能优化点

1. **并发控制**: `asyncio.Semaphore` 限制并发数
2. **资源复用**: OpenAI客户端实例复用
3. **流式传输**: 使用生成器避免内存占用
4. **类型注解**: 支持IDE优化和JIT编译
5. **异步优先**: 异步方法性能优于线程池

---

## 🔍 使用示例对应的调用路径

### 示例1: 基础调用
```python
llm.invoke("你好")
```
→ `VLLMClient.invoke` → `OpenAI.create` → vLLM服务器

### 示例2: 模板调用
```python
chain.run(text="...")
```
→ `LLMChain.run` → `PromptTemplate.format` → `VLLMClient.invoke` → vLLM

### 示例3: 异步批量
```python
await llm.abatch(prompts)
```
→ `VLLMClient.abatch` → `asyncio.Semaphore` → `asyncio.gather` → 多个`ainvoke` → vLLM

### 示例4: 流式输出
```python
for chunk in llm.stream("..."):
```
→ `VLLMClient.stream` → `OpenAI.create(stream=True)` → yield chunks

---

## 📚 总结

本框架的调用关系呈现**分层解耦**的特点：

1. **配置层** → 提供参数
2. **模板层** → 处理提示词
3. **客户端层** → 管理API调用
4. **链层** → 组织工作流
5. **工具层** → 提供辅助功能

各层职责清晰，通过**依赖注入**和**接口抽象**实现松耦合，便于扩展和维护。
