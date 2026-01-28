# 🚀 LangChain风格的vLLM调用框架

一个优雅、高效的本地大模型调用框架，借鉴LangChain设计理念，支持提示词组装和高并发调用。

## ✨ 核心特性

- 🎯 **LangChain风格API** - 熟悉的PromptTemplate和Chain模式
- ⚡ **高性能并发** - 支持async/await和线程池并发
- 🔧 **灵活的提示词组装** - 强大的模板系统
- 🎨 **优雅的代码设计** - 类型注解、dataclass、生成器
- 📦 **开箱即用** - 只需一个依赖，简单配置即可开始

## 📦 安装

```bash
pip install openai
```

## 🎯 快速开始

### 1. 基础调用

```python
from langchain_vllm import VLLMClient, ModelConfig

# 初始化
config = ModelConfig(
    base_url="http://localhost:8000/v1",
    model="Qwen2.5-7B-Instruct",
    temperature=0.7
)
llm = VLLMClient(config)

# 调用
response = llm.invoke("什么是Python?")
print(response)
```

### 2. 提示词模板

```python
from langchain_vllm import PromptTemplate, LLMChain

# 创建模板
template = PromptTemplate(
    "将'{text}'翻译成{target_lang}"
)

# 创建链
chain = LLMChain(llm, template)

# 运行
result = chain.run(text="Hello", target_lang="中文")
```

### 3. 异步并发

```python
import asyncio

# 批量并发调用
questions = ["问题1", "问题2", "问题3"]
results = await llm.abatch(questions, concurrency=3)
```

### 4. 流式输出

```python
# 实时输出
for chunk in llm.stream("写一首诗"):
    print(chunk, end="", flush=True)
```

## 📚 核心组件

### ModelConfig - 模型配置

```python
config = ModelConfig(
    base_url="http://localhost:8000/v1",  # vLLM服务地址
    api_key="EMPTY",                      # API密钥
    model="your-model-name",              # 模型名称
    temperature=0.7,                      # 温度参数
    max_tokens=2048,                      # 最大token数
    top_p=0.9                            # top_p采样
)
```

### PromptTemplate - 提示词模板

```python
# 自动识别变量
template = PromptTemplate(
    "请扮演{role},回答:{question}"
)

# 手动指定变量
template = PromptTemplate(
    template="...",
    input_variables=["role", "question"]
)

# 格式化
prompt = template.format(role="专家", question="什么是AI?")
```

### VLLMClient - 核心客户端

**同步方法:**
- `invoke(prompt)` - 单次调用
- `stream(prompt)` - 流式输出
- `batch(prompts, max_workers=5)` - 批量并发（线程池）

**异步方法:**
- `ainvoke(prompt)` - 异步单次调用
- `astream(prompt)` - 异步流式输出
- `abatch(prompts, concurrency=5)` - 异步批量并发

### LLMChain - 调用链

```python
# 基础链
chain = LLMChain(llm, prompt_template)

# 同步运行
result = chain.run(var1="value1", var2="value2")

# 异步运行
result = await chain.arun(var1="value1", var2="value2")

# 添加输出解析器
chain = LLMChain(
    llm, 
    prompt_template,
    output_parser=OutputParser.json_parser
)
```

### OutputParser - 输出解析器

```python
from langchain_vllm import OutputParser

# JSON解析
result = OutputParser.json_parser(response)

# 列表解析
items = OutputParser.list_parser(response, delimiter="\n")

# 数字解析
number = OutputParser.number_parser(response)
```

## 🎨 高级用法

### 1. 复杂提示词组装

```python
template = PromptTemplate("""
角色: {role}
背景: {background}
任务: {task}

输入:
{input}

要求:
{requirements}
""")

chain = LLMChain(llm, template)
result = chain.run(
    role="高级工程师",
    background="重构遗留代码",
    task="优化性能",
    input="代码片段...",
    requirements="可读性+性能"
)
```

### 2. 多步骤处理

```python
# 步骤1: 生成大纲
outline_chain = create_chain(llm, "为{topic}生成大纲")
outline = outline_chain.run(topic="Python教程")

# 步骤2: 扩展内容
expand_chain = create_chain(llm, "扩展大纲:\n{outline}")
content = expand_chain.run(outline=outline)
```

### 3. 并发处理不同任务

```python
async def process_code(code: str):
    """并发执行多个分析任务"""
    tasks = [
        review_chain.arun(code=code),
        explain_chain.arun(code=code),
        optimize_chain.arun(code=code)
    ]
    return await asyncio.gather(*tasks)

results = await process_code("def hello(): pass")
```

### 4. 错误处理

```python
async def safe_batch_call(prompts: List[str]):
    """带错误处理的批量调用"""
    async def safe_call(prompt: str):
        try:
            return await llm.ainvoke(prompt)
        except Exception as e:
            return f"Error: {e}"
    
    tasks = [safe_call(p) for p in prompts]
    return await asyncio.gather(*tasks)
```

### 5. 自定义输出解析器

```python
def custom_parser(response: str) -> Dict:
    """自定义解析逻辑"""
    # 提取代码块
    code = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
    # 提取说明
    explanation = response.split("```")[0]
    
    return {
        "code": code.group(1) if code else "",
        "explanation": explanation.strip()
    }

chain = LLMChain(llm, template, output_parser=custom_parser)
```

## 📖 示例代码

### 基础示例
```bash
python quick_start.py          # 快速入门
```

### 提示词组装
```bash
python examples.py             # 各种提示词用法
```

### 并发处理
```bash
python concurrent_examples.py  # 并发和性能优化
```

## 🎓 设计亮点

### 1. 现代Python特性

```python
# dataclass - 简洁的数据类
@dataclass
class ModelConfig:
    temperature: float = 0.7
    max_tokens: int = 2048

# 类型注解 - IDE友好
def invoke(self, prompt: str) -> str:
    pass

# 生成器 - 内存高效
def stream(self, prompt: str):
    for chunk in response:
        yield chunk
```

### 2. 函数式编程

```python
# 高阶函数
chain = LLMChain(llm, prompt, output_parser=parser)

# 链式调用
result = chain.run(**kwargs)

# 管道操作符
chain = chain1 | chain2 | chain3
```

### 3. 异步编程

```python
# async/await
response = await llm.ainvoke(prompt)

# 并发控制
results = await llm.abatch(prompts, concurrency=5)

# Semaphore限流
async with semaphore:
    return await self.ainvoke(prompt)
```

### 4. 抽象与封装

```python
# 基类定义接口
class BaseChain(ABC):
    @abstractmethod
    def run(self, **kwargs): pass

# 具体实现
class LLMChain(BaseChain):
    def run(self, **kwargs):
        # 实现细节
```

## 🔧 配置说明

### vLLM服务配置

确保你的vLLM服务已启动:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/model \
    --host 0.0.0.0 \
    --port 8000
```

### 查看可用模型

```bash
curl http://localhost:8000/v1/models
```

### 修改配置

在代码中修改 `ModelConfig`:

```python
config = ModelConfig(
    base_url="http://192.168.1.100:8000/v1",  # 改成实际地址
    model="Qwen2.5-7B-Instruct"                # 改成实际模型名
)
```

## 💡 最佳实践

### 1. 并发度控制

```python
# 根据服务器性能调整并发数
# GPU少 -> concurrency=3
# GPU多 -> concurrency=10+
results = await llm.abatch(prompts, concurrency=5)
```

### 2. 提示词优化

```python
# ✅ 好的提示词: 清晰、具体
template = PromptTemplate("""
任务: 代码审查
语言: {language}
代码:
```{language}
{code}
```
重点: {focus}
输出: JSON格式
""")

# ❌ 差的提示词: 模糊、冗余
template = PromptTemplate("审查这段代码{code}")
```

### 3. 错误处理

```python
# 始终处理可能的异常
try:
    result = await llm.ainvoke(prompt)
except Exception as e:
    logger.error(f"调用失败: {e}")
    result = "默认响应"
```

### 4. 性能监控

```python
import time

start = time.time()
result = await llm.abatch(prompts, concurrency=5)
elapsed = time.time() - start

print(f"处理{len(prompts)}个请求耗时{elapsed:.2f}秒")
print(f"平均{elapsed/len(prompts):.2f}秒/个")
```

## 🆚 对比其他框架

| 特性 | 本框架 | LangChain | 原生OpenAI |
|-----|--------|-----------|-----------|
| 学习曲线 | ⭐⭐ 简单 | ⭐⭐⭐⭐ 复杂 | ⭐ 很简单 |
| 代码量 | 150行 | 10000+行 | - |
| 提示词管理 | ✅ | ✅ | ❌ |
| 并发支持 | ✅ 原生 | ⚠️ 有限 | ❌ |
| 本地模型 | ✅ 专门优化 | ⚠️ 需配置 | ❌ |
| 性能 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 可定制 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

## 🤝 贡献

欢迎提Issue和PR！

## 📄 许可

MIT License

## 🙏 致谢

- 设计灵感来自 [LangChain](https://github.com/langchain-ai/langchain)
- 底层使用 [vLLM](https://github.com/vllm-project/vllm) 推理引擎
- OpenAI SDK 提供统一接口

---

**Happy Coding! 🎉**

如有问题，欢迎提Issue或查看示例代码。
