"""
LangChain风格的vLLM调用框架
支持提示词组装、并发调用、链式操作
"""

import asyncio
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from openai import OpenAI, AsyncOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import re


@dataclass
class ModelConfig:
    """模型配置"""
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model: str = "your-model-name"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p
        }


class PromptTemplate:
    """提示词模板类 - LangChain风格"""
    
    def __init__(
        self,
        template: str,
        input_variables: Optional[List[str]] = None
    ):
        """
        初始化提示词模板
        
        Args:
            template: 模板字符串,使用{variable}作为占位符
            input_variables: 输入变量列表,如果为None则自动解析
        """
        self.template = template
        self.input_variables = input_variables or self._extract_variables(template)
    
    @staticmethod
    def _extract_variables(template: str) -> List[str]:
        """从模板中提取变量名"""
        return list(set(re.findall(r'\{(\w+)\}', template)))
    
    def format(self, **kwargs) -> str:
        """格式化模板"""
        missing = set(self.input_variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"缺少必需的变量: {missing}")
        return self.template.format(**kwargs)
    
    def __repr__(self) -> str:
        return f"PromptTemplate(variables={self.input_variables})"


class BaseChain(ABC):
    """基础链类"""
    
    @abstractmethod
    def run(self, **kwargs) -> str:
        """同步运行"""
        pass
    
    @abstractmethod
    async def arun(self, **kwargs) -> str:
        """异步运行"""
        pass


class LLMChain(BaseChain):
    """LLM调用链"""
    
    def __init__(
        self,
        llm: 'VLLMClient',
        prompt: Union[PromptTemplate, str],
        output_parser: Optional[Callable[[str], Any]] = None
    ):
        """
        初始化LLM链
        
        Args:
            llm: VLLMClient实例
            prompt: 提示词模板或字符串
            output_parser: 输出解析函数
        """
        self.llm = llm
        self.prompt = prompt if isinstance(prompt, PromptTemplate) else PromptTemplate(prompt)
        self.output_parser = output_parser or (lambda x: x)
    
    def run(self, **kwargs) -> Any:
        """同步运行链"""
        formatted_prompt = self.prompt.format(**kwargs)
        response = self.llm.invoke(formatted_prompt)
        return self.output_parser(response)
    
    async def arun(self, **kwargs) -> Any:
        """异步运行链"""
        formatted_prompt = self.prompt.format(**kwargs)
        response = await self.llm.ainvoke(formatted_prompt)
        return self.output_parser(response)
    
    def __or__(self, other: 'BaseChain') -> 'SequentialChain':
        """支持 | 运算符组合链"""
        return SequentialChain(chains=[self, other])


class SequentialChain(BaseChain):
    """顺序链 - 按顺序执行多个链"""
    
    def __init__(self, chains: List[BaseChain]):
        self.chains = chains
    
    def run(self, **kwargs) -> Any:
        """同步运行"""
        result = kwargs
        for chain in self.chains:
            if isinstance(result, dict):
                result = chain.run(**result)
            else:
                result = chain.run(input=result)
        return result
    
    async def arun(self, **kwargs) -> Any:
        """异步运行"""
        result = kwargs
        for chain in self.chains:
            if isinstance(result, dict):
                result = await chain.arun(**result)
            else:
                result = await chain.arun(input=result)
        return result


class VLLMClient:
    """vLLM客户端 - 支持同步和异步"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """初始化客户端"""
        self.config = config or ModelConfig()
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key
        )
        self.async_client = AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key
        )
    
    def invoke(
        self,
        prompt: str,
        system: str = "你是一个有帮助的AI助手。",
        **kwargs
    ) -> str:
        """同步调用"""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
        
        params = {**self.config.to_dict(), **kwargs, "messages": messages}
        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content
    
    async def ainvoke(
        self,
        prompt: str,
        system: str = "你是一个有帮助的AI助手。",
        **kwargs
    ) -> str:
        """异步调用"""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
        
        params = {**self.config.to_dict(), **kwargs, "messages": messages}
        response = await self.async_client.chat.completions.create(**params)
        return response.choices[0].message.content
    
    def stream(
        self,
        prompt: str,
        system: str = "你是一个有帮助的AI助手。",
        **kwargs
    ):
        """同步流式调用"""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
        
        params = {**self.config.to_dict(), **kwargs, "messages": messages, "stream": True}
        response = self.client.chat.completions.create(**params)
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    async def astream(
        self,
        prompt: str,
        system: str = "你是一个有帮助的AI助手。",
        **kwargs
    ):
        """异步流式调用"""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
        
        params = {**self.config.to_dict(), **kwargs, "messages": messages, "stream": True}
        response = await self.async_client.chat.completions.create(**params)
        
        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def batch(
        self,
        prompts: List[str],
        max_workers: int = 5,
        **kwargs
    ) -> List[str]:
        """批量并发调用(线程池)"""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.invoke, prompt, **kwargs) for prompt in prompts]
            return [future.result() for future in as_completed(futures)]
    
    async def abatch(
        self,
        prompts: List[str],
        concurrency: int = 5,
        **kwargs
    ) -> List[str]:
        """批量并发调用(异步)"""
        semaphore = asyncio.Semaphore(concurrency)
        
        async def _invoke_with_sem(prompt: str):
            async with semaphore:
                return await self.ainvoke(prompt, **kwargs)
        
        tasks = [_invoke_with_sem(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)


class OutputParser:
    """输出解析器集合"""
    
    @staticmethod
    def json_parser(response: str) -> Dict[str, Any]:
        """解析JSON输出"""
        import json
        # 提取JSON内容(支持markdown代码块)
        match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if match:
            response = match.group(1)
        return json.loads(response)
    
    @staticmethod
    def list_parser(response: str, delimiter: str = "\n") -> List[str]:
        """解析列表输出"""
        return [line.strip() for line in response.split(delimiter) if line.strip()]
    
    @staticmethod
    def number_parser(response: str) -> float:
        """解析数字输出"""
        match = re.search(r'-?\d+\.?\d*', response)
        if match:
            return float(match.group())
        raise ValueError(f"无法从响应中提取数字: {response}")


# 便捷函数
def create_chain(
    llm: VLLMClient,
    template: str,
    output_parser: Optional[Callable] = None
) -> LLMChain:
    """快速创建链"""
    prompt = PromptTemplate(template)
    return LLMChain(llm, prompt, output_parser)


if __name__ == "__main__":
    # 同步示例
    print("=== 同步示例 ===")
    config = ModelConfig(model="your-model-name")
    llm = VLLMClient(config)
    
    # 基础调用
    response = llm.invoke("你好,介绍一下你自己")
    print(response[:100] + "...\n")
    
    # 使用链
    translate_chain = create_chain(
        llm,
        "请将以下文本翻译成{target_lang}:\n{text}"
    )
    result = translate_chain.run(target_lang="英文", text="你好世界")
    print(f"翻译结果: {result}\n")
    
    # 异步示例
    print("=== 异步示例 ===")
    async def async_example():
        # 单个异步调用
        response = await llm.ainvoke("什么是机器学习?")
        print(response[:100] + "...\n")
        
        # 批量异步调用
        prompts = [
            "Python的主要特点是什么?",
            "什么是Docker?",
            "解释一下REST API"
        ]
        results = await llm.abatch(prompts, concurrency=3)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result[:50]}...\n")
    
    asyncio.run(async_example())
