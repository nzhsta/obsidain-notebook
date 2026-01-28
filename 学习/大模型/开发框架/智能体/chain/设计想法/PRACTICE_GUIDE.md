# 🛠️ 实践指南：从需求到实现的完整流程

> 手把手教你如何设计一个本地大模型调用框架

---

## 🎯 场景模拟

假设你现在是一名开发者，产品经理给你提了这个需求。你该如何思考和行动？

---

## 第一天：需求分析（2小时）

### Step 1: 明确需求 (30分钟)

#### 📝 写下需求清单

```markdown
# 需求文档

## 功能性需求
1. 调用本地vLLM服务
2. 管理和更新提示词
3. ?（需要确认：是否需要流式输出？）
4. ?（需要确认：是否需要并发？）

## 非功能性需求  
1. 代码简洁、模块化
2. 使用高级Python语法
3. 可复用性高
4. 调用方式主流

## 约束条件
1. 本地vLLM部署
2. Python语言
3. ?（需要确认：Python版本？）
```

#### 💬 与"产品经理"沟通

**你应该问的问题：**

```
Q1: 主要使用场景是什么？
   - 简单问答？
   - 批量处理？  
   - 复杂工作流？

Q2: 性能要求如何？
   - 需要处理多少并发？
   - 响应时间要求？

Q3: 未来可能的扩展？
   - 多模型支持？
   - 联网搜索？
   - 工具调用？

Q4: 团队技术栈？
   - Python版本？
   - 是否熟悉异步？
```

### Step 2: 定义成功标准 (30分钟)

#### ✅ MVP（最小可用产品）标准

```python
# MVP: 能完成基础调用即可
llm = VLLMClient()
response = llm.call("你好")
print(response)  # ✅ 能看到响应就算成功
```

#### ✅ 完整版标准

```python
# 完整版: 支持模板、链式、并发
template = PromptTemplate("翻译: {text}")
chain = LLMChain(llm, template)
result = chain.run(text="Hello")

# 并发
results = await llm.batch(prompts)
```

### Step 3: 技术调研 (1小时)

#### 📚 调研清单

```
□ vLLM API文档
  - 接口格式
  - 参数说明
  - 示例代码

□ OpenAI SDK
  - 是否兼容vLLM
  - 如何使用
  - 异步支持

□ LangChain
  - 核心设计理念
  - PromptTemplate实现
  - Chain模式

□ 竞品分析
  - LlamaIndex
  - Guidance
  - 其他框架
```

#### 📊 调研结果记录

```markdown
# 技术调研结果

## vLLM
- ✅ 兼容OpenAI API
- ✅ 支持流式输出
- ✅ 端点: /v1/chat/completions

## OpenAI SDK
- ✅ 有Python包: `openai`
- ✅ 支持自定义base_url
- ✅ 支持异步: AsyncOpenAI

## 设计参考
- LangChain: PromptTemplate + Chain模式
- 简化原则: 只实现核心功能

## 技术选型
- OpenAI SDK: 作为HTTP客户端
- asyncio: 异步支持
- dataclass: 配置管理
```

---

## 第二天：架构设计（3小时）

### Step 1: 画架构图 (1小时)

#### 🎨 在纸上或白板画出来

```
┌─────────────────────────────────┐
│         用户代码                 │
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│    工具层 (create_chain)        │
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│    链层 (LLMChain)              │
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│    客户端层 (VLLMClient)        │
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│    模板层 (PromptTemplate)      │
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│    配置层 (ModelConfig)         │
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│    OpenAI SDK                   │
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│    vLLM Server                  │
└─────────────────────────────────┘
```

### Step 2: 定义核心类 (1小时)

#### 📋 类设计清单

```python
# 1. ModelConfig - 配置类
"""
职责: 存储模型配置
属性: base_url, model, temperature, max_tokens, top_p
方法: to_dict() - 转换为API参数
"""

# 2. PromptTemplate - 模板类
"""
职责: 管理提示词模板
属性: template, input_variables
方法: 
  - __init__(template)
  - format(**kwargs) - 格式化
  - _extract_variables() - 提取变量
"""

# 3. VLLMClient - 客户端类
"""
职责: 与vLLM交互
属性: config, client
方法:
  - invoke(prompt) - 同步调用
  - ainvoke(prompt) - 异步调用
  - stream(prompt) - 流式输出
  - batch(prompts) - 批量调用
"""

# 4. BaseChain - 抽象基类
"""
职责: 定义链接口
方法:
  - run(**kwargs) - 抽象方法
  - arun(**kwargs) - 抽象方法
"""

# 5. LLMChain - 调用链
"""
职责: 组合模板+调用+解析
属性: llm, prompt, output_parser
方法:
  - run(**kwargs)
  - arun(**kwargs)
"""
```

### Step 3: 设计接口 (1小时)

#### 🔌 接口定义

```python
# 接口设计文档

## 基础调用接口
llm.invoke(prompt: str) -> str
llm.ainvoke(prompt: str) -> str  
llm.stream(prompt: str) -> Generator[str]

## 批量调用接口
llm.batch(prompts: List[str], max_workers: int) -> List[str]
llm.abatch(prompts: List[str], concurrency: int) -> List[str]

## 模板接口
template = PromptTemplate(template: str)
template.format(**kwargs) -> str

## 链接口  
chain = LLMChain(llm, prompt, parser)
chain.run(**kwargs) -> Any
chain.arun(**kwargs) -> Any

## 工具接口
create_chain(llm, template, parser) -> LLMChain
```

---

## 第三天：MVP实现（4小时）

### Step 1: 搭建项目结构 (30分钟)

```bash
# 创建项目目录
mkdir vllm-framework
cd vllm-framework

# 创建文件
touch langchain_vllm.py
touch requirements.txt
touch test_basic.py
touch README.md
```

```
vllm-framework/
├── langchain_vllm.py    # 核心代码
├── requirements.txt     # 依赖
├── test_basic.py        # 测试
└── README.md           # 文档
```

### Step 2: 实现MVP (2小时)

#### 版本1: 最简单能跑的版本

```python
# langchain_vllm.py - MVP版本

from openai import OpenAI

class ModelConfig:
    """配置类"""
    def __init__(self, base_url="http://localhost:8000/v1", model="your-model"):
        self.base_url = base_url
        self.model = model

class VLLMClient:
    """客户端类"""
    def __init__(self, config):
        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key="EMPTY"
        )
    
    def invoke(self, prompt):
        """调用LLM"""
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content


# 测试
if __name__ == "__main__":
    config = ModelConfig()
    llm = VLLMClient(config)
    response = llm.invoke("你好")
    print(response)
```

**目标：能跑起来！**

### Step 3: 测试MVP (1小时)

```python
# test_basic.py

from langchain_vllm import VLLMClient, ModelConfig

def test_basic_call():
    """测试基础调用"""
    config = ModelConfig(
        base_url="http://localhost:8000/v1",
        model="your-model"
    )
    llm = VLLMClient(config)
    
    # 测试
    response = llm.invoke("你好，请介绍一下你自己")
    assert isinstance(response, str)
    assert len(response) > 0
    print("✅ 基础调用测试通过")

if __name__ == "__main__":
    test_basic_call()
```

### Step 4: 代码审查 (30分钟)

#### 自问自答

```
Q1: 代码能跑吗？
A1: ✅ 能跑

Q2: 满足需求吗？
A2: ⚠️  部分满足，还缺模板、链式、并发

Q3: 代码质量如何？
A3: ⚠️  可以，但可以更好（缺少类型注解、文档）

Q4: 下一步做什么？
A4: 添加PromptTemplate
```

---

## 第四天：添加核心功能（4小时）

### Step 1: 添加PromptTemplate (1.5小时)

```python
import re
from typing import List

class PromptTemplate:
    """提示词模板"""
    
    def __init__(self, template: str):
        self.template = template
        self.input_variables = self._extract_variables(template)
    
    @staticmethod
    def _extract_variables(template: str) -> List[str]:
        """提取变量"""
        return list(set(re.findall(r'\{(\w+)\}', template)))
    
    def format(self, **kwargs) -> str:
        """格式化模板"""
        # 检查缺失的变量
        missing = set(self.input_variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"缺少变量: {missing}")
        return self.template.format(**kwargs)


# 测试
template = PromptTemplate("将 {text} 翻译成 {lang}")
print(template.input_variables)  # ['text', 'lang']
result = template.format(text="Hello", lang="中文")
print(result)  # "将 Hello 翻译成 中文"
```

### Step 2: 添加LLMChain (1.5小时)

```python
from typing import Any, Callable, Optional

class LLMChain:
    """LLM调用链"""
    
    def __init__(
        self,
        llm: VLLMClient,
        prompt: PromptTemplate,
        output_parser: Optional[Callable] = None
    ):
        self.llm = llm
        self.prompt = prompt
        self.output_parser = output_parser or (lambda x: x)
    
    def run(self, **kwargs) -> Any:
        """运行链"""
        # 1. 格式化提示词
        formatted_prompt = self.prompt.format(**kwargs)
        
        # 2. 调用LLM
        response = self.llm.invoke(formatted_prompt)
        
        # 3. 解析输出
        return self.output_parser(response)


# 测试
template = PromptTemplate("翻译: {text}")
chain = LLMChain(llm, template)
result = chain.run(text="Hello World")
print(result)
```

### Step 3: 测试核心功能 (1小时)

```python
def test_template():
    """测试模板"""
    template = PromptTemplate("问题: {question}, 要求: {requirement}")
    result = template.format(question="什么是Python", requirement="简短回答")
    assert "什么是Python" in result
    print("✅ 模板测试通过")

def test_chain():
    """测试链"""
    config = ModelConfig()
    llm = VLLMClient(config)
    template = PromptTemplate("翻译成英文: {text}")
    chain = LLMChain(llm, template)
    
    result = chain.run(text="你好")
    assert isinstance(result, str)
    print("✅ 链测试通过")
```

---

## 第五天：添加高级功能（4小时）

### Step 1: 添加异步支持 (2小时)

```python
from openai import AsyncOpenAI
import asyncio

class VLLMClient:
    def __init__(self, config):
        self.config = config
        self.client = OpenAI(...)
        self.async_client = AsyncOpenAI(...)  # 新增
    
    async def ainvoke(self, prompt: str) -> str:
        """异步调用"""
        response = await self.async_client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content


# 测试
async def test_async():
    llm = VLLMClient(config)
    response = await llm.ainvoke("你好")
    print(response)

asyncio.run(test_async())
```

### Step 2: 添加并发支持 (2小时)

```python
from typing import List

async def abatch(
    self,
    prompts: List[str],
    concurrency: int = 5
) -> List[str]:
    """异步批量调用"""
    semaphore = asyncio.Semaphore(concurrency)
    
    async def _invoke_with_sem(prompt: str):
        async with semaphore:
            return await self.ainvoke(prompt)
    
    tasks = [_invoke_with_sem(p) for p in prompts]
    return await asyncio.gather(*tasks)


# 测试
async def test_batch():
    llm = VLLMClient(config)
    prompts = ["问题1", "问题2", "问题3"]
    results = await llm.abatch(prompts, concurrency=2)
    assert len(results) == 3
    print("✅ 并发测试通过")
```

---

## 第六天：优化和完善（3小时）

### Step 1: 代码重构 (1小时)

#### 使用dataclass优化

```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    base_url: str = "http://localhost:8000/v1"
    model: str = "your-model"
    temperature: float = 0.7
    max_tokens: int = 2048
    
    def to_dict(self):
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
```

#### 添加类型注解

```python
from typing import List, Dict, Any, Optional

def invoke(
    self,
    prompt: str,
    system: str = "你是助手",
    **kwargs
) -> str:
    """
    同步调用LLM
    
    Args:
        prompt: 用户提示词
        system: 系统提示词
        **kwargs: 其他参数
    
    Returns:
        LLM响应
    """
    ...
```

### Step 2: 添加文档 (1小时)

```markdown
# vLLM调用框架

## 快速开始

\`\`\`python
from langchain_vllm import VLLMClient, ModelConfig

config = ModelConfig(model="Qwen2.5-7B")
llm = VLLMClient(config)
response = llm.invoke("你好")
\`\`\`

## 功能特性

- ✅ 提示词模板
- ✅ 链式调用
- ✅ 异步支持
- ✅ 并发控制
```

### Step 3: 创建示例 (1小时)

```python
# examples.py

"""使用示例"""

from langchain_vllm import *

# 示例1: 基础调用
def example_basic():
    llm = VLLMClient(ModelConfig())
    response = llm.invoke("你好")
    print(response)

# 示例2: 模板
def example_template():
    template = PromptTemplate("翻译: {text}")
    chain = LLMChain(llm, template)
    result = chain.run(text="Hello")
    print(result)

# 示例3: 并发
async def example_concurrent():
    llm = VLLMClient(ModelConfig())
    prompts = ["问题1", "问题2", "问题3"]
    results = await llm.abatch(prompts)
    print(results)
```

---

## 💡 关键经验总结

### 1. 需求分析阶段

**❌ 错误做法：**
- 拿到需求就开始写代码
- 不问清楚细节
- 不考虑扩展性

**✅ 正确做法：**
- 理解需求背后的意图
- 问清楚所有不确定的点
- 定义清晰的成功标准

### 2. 架构设计阶段

**❌ 错误做法：**
- 想到哪写到哪
- 一个文件几千行
- 类和函数职责不清

**✅ 正确做法：**
- 先画架构图
- 定义清晰的分层
- 每个类单一职责

### 3. 实现阶段

**❌ 错误做法：**
- 一次性写所有功能
- 不测试就继续写
- 代码没有注释

**✅ 正确做法：**
- MVP → 核心功能 → 高级功能
- 每个功能都测试
- 添加类型注解和文档

---

## 🎯 检查清单

### 需求分析 ✓

- [ ] 列出功能需求
- [ ] 列出非功能需求  
- [ ] 识别隐含需求
- [ ] 定义成功标准
- [ ] 技术调研完成

### 架构设计 ✓

- [ ] 画出架构图
- [ ] 定义核心类
- [ ] 设计接口
- [ ] 确定技术栈
- [ ] 评审设计方案

### 实现 ✓

- [ ] MVP能跑起来
- [ ] 核心功能完成
- [ ] 高级功能完成
- [ ] 代码有测试
- [ ] 代码有文档

### 质量 ✓

- [ ] 类型注解完整
- [ ] 文档字符串完整
- [ ] 代码格式规范
- [ ] 无明显bug
- [ ] 性能可接受

---

## 🚀 下一步

### 如果要继续优化

```
1. 添加更多测试
2. 性能基准测试
3. 错误处理优化
4. 添加日志功能
5. 发布到PyPI
```

### 如果要添加新功能

```
1. 评估是否必要
2. 设计接口
3. 实现MVP
4. 测试
5. 文档
```

---

## 💭 思考题

练习一下你的设计思维：

**Q1: 如果要添加"对话历史"功能，你会怎么设计？**

**Q2: 如果要支持"多模型切换"，应该如何改造？**

**Q3: 如果要添加"结果缓存"，应该放在哪一层？**

---

**记住：设计先行，实现在后。好的设计是迭代出来的！** 🎨
