---
last-updated: 2026-04-17
topics:
  - LangChain LLM 基类设计
---

# LangChain

## LangChain LLM 基类设计

> 来源：2026-04-17 对话归档

**问题/场景：**
用户询问 LangChain 中的 LLM 基类是否继承 Python 的 ABC（抽象基类），以及这样设计的原因。

**核心知识点：**
- LangChain 的核心基类（如 `BaseLLM`、`BaseChatModel`）继承自 `ABC`
- 抽象方法（如 `_generate()`、`_llm_type()`）强制所有具体 LLM 实现类必须实现核心接口
- 这种设计实现了多供应商（OpenAI、Anthropic、本地模型等）的统一调用接口
- 新版本中架构进化为 LCEL（LangChain Expression Language），核心接口变为 `Runnable`，但依然基于 `ABC`
- 自定义 LLM 接入 LangChain 时，本质上是在实现抽象基类规定的接口

**代码示例：**
```python
from abc import ABC, abstractmethod

# 以 BaseLLM 为例（简化示意）
class BaseLLM(BaseLanguageModel, ABC):
    @abstractmethod
    def _generate(self, prompts, stop=None):
        """子类必须实现的核心生成逻辑"""
        pass

    @abstractmethod
    def _llm_type(self) -> str:
        """返回 LLM 类型标识"""
        pass

# 新版本 Runnable 接口（同样继承 ABC）
class Runnable(ABC):
    @abstractmethod
    def invoke(self, input, config=None):
        pass

    @abstractmethod
    def stream(self, input, config=None):
        pass
```

---
