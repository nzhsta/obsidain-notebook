"""
提示词组装示例 - LangChain风格
"""

import asyncio
from langchain_vllm import (
    VLLMClient, ModelConfig, PromptTemplate,
    LLMChain, create_chain, OutputParser
)


def example_1_basic_prompt():
    """示例1: 基础提示词模板"""
    print("=" * 60)
    print("示例1: 基础提示词模板")
    print("=" * 60)
    
    # 创建LLM客户端
    config = ModelConfig(
        model="Qwen2.5-7B-Instruct",  # 替换为你的模型
        temperature=0.3
    )
    llm = VLLMClient(config)
    
    # 定义提示词模板
    template = PromptTemplate(
        template="""请扮演一个{role},回答以下问题:

问题: {question}

要求:
- 专业性强
- 简洁明了
- 不超过{max_words}字"""
    )
    
    # 创建链
    chain = LLMChain(llm, template)
    
    # 运行
    result = chain.run(
        role="Python专家",
        question="什么是装饰器?",
        max_words="100"
    )
    print(result)
    print()


def example_2_multi_step():
    """示例2: 多步骤提示词组装"""
    print("=" * 60)
    print("示例2: 多步骤处理")
    print("=" * 60)
    
    llm = VLLMClient(ModelConfig(model="your-model-name"))
    
    # 步骤1: 生成大纲
    outline_prompt = PromptTemplate(
        """请为以下主题生成文章大纲:
        
主题: {topic}
要求: {requirements}

只返回大纲,每行一个要点"""
    )
    
    # 步骤2: 扩展内容
    expand_prompt = PromptTemplate(
        """基于以下大纲,扩展成完整文章:

{outline}

要求: 每个要点扩展成一段话,总共约{words}字"""
    )
    
    # 创建链
    outline_chain = LLMChain(llm, outline_prompt)
    expand_chain = LLMChain(llm, expand_prompt)
    
    # 执行多步骤
    outline = outline_chain.run(
        topic="Python异步编程",
        requirements="包含3-5个要点"
    )
    print("大纲:\n", outline, "\n")
    
    article = expand_chain.run(
        outline=outline,
        words="300"
    )
    print("文章:\n", article)
    print()


def example_3_output_parser():
    """示例3: 使用输出解析器"""
    print("=" * 60)
    print("示例3: 输出解析器")
    print("=" * 60)
    
    llm = VLLMClient(ModelConfig(model="your-model-name"))
    
    # JSON解析示例
    json_template = PromptTemplate(
        """分析以下文本的情感,以JSON格式返回:

文本: {text}

返回格式:
```json
{{
    "sentiment": "positive/negative/neutral",
    "confidence": 0.0-1.0,
    "keywords": ["关键词1", "关键词2"]
}}
```"""
    )
    
    json_chain = LLMChain(
        llm,
        json_template,
        output_parser=OutputParser.json_parser
    )
    
    result = json_chain.run(text="这个产品真的太好用了!")
    print("JSON解析结果:", result)
    print(f"情感: {result['sentiment']}")
    print(f"置信度: {result['confidence']}")
    print()
    
    # 列表解析示例
    list_template = PromptTemplate(
        """列出{topic}的{count}个优点,每行一个,不要编号:"""
    )
    
    list_chain = LLMChain(
        llm,
        list_template,
        output_parser=OutputParser.list_parser
    )
    
    advantages = list_chain.run(topic="Python", count="5")
    print("列表解析结果:")
    for i, adv in enumerate(advantages, 1):
        print(f"{i}. {adv}")
    print()


async def example_4_concurrent():
    """示例4: 并发处理多个提示词"""
    print("=" * 60)
    print("示例4: 并发处理")
    print("=" * 60)
    
    llm = VLLMClient(ModelConfig(model="your-model-name"))
    
    # 定义多个任务模板
    code_review_template = PromptTemplate(
        "请审查以下{language}代码并给出建议:\n```{language}\n{code}\n```"
    )
    
    code_explain_template = PromptTemplate(
        "请解释以下{language}代码的功能:\n```{language}\n{code}\n```"
    )
    
    code_optimize_template = PromptTemplate(
        "请优化以下{language}代码:\n```{language}\n{code}\n```"
    )
    
    # 创建链
    review_chain = LLMChain(llm, code_review_template)
    explain_chain = LLMChain(llm, code_explain_template)
    optimize_chain = LLMChain(llm, code_optimize_template)
    
    # 并发执行
    code_sample = "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)"
    
    tasks = [
        review_chain.arun(language="python", code=code_sample),
        explain_chain.arun(language="python", code=code_sample),
        optimize_chain.arun(language="python", code=code_sample)
    ]
    
    results = await asyncio.gather(*tasks)
    
    print("审查结果:\n", results[0][:100], "...\n")
    print("解释结果:\n", results[1][:100], "...\n")
    print("优化结果:\n", results[2][:100], "...\n")


async def example_5_batch_processing():
    """示例5: 批量处理不同问题"""
    print("=" * 60)
    print("示例5: 批量问答")
    print("=" * 60)
    
    llm = VLLMClient(ModelConfig(model="your-model-name"))
    
    # 问答模板
    qa_template = PromptTemplate("用一句话回答: {question}")
    qa_chain = LLMChain(llm, qa_template)
    
    # 批量问题
    questions = [
        {"question": "什么是Docker?"},
        {"question": "什么是Kubernetes?"},
        {"question": "什么是微服务?"},
        {"question": "什么是REST API?"},
        {"question": "什么是GraphQL?"}
    ]
    
    # 并发处理
    tasks = [qa_chain.arun(**q) for q in questions]
    answers = await asyncio.gather(*tasks)
    
    # 输出结果
    for q, a in zip(questions, answers):
        print(f"Q: {q['question']}")
        print(f"A: {a}\n")


def example_6_complex_assembly():
    """示例6: 复杂提示词组装"""
    print("=" * 60)
    print("示例6: 复杂提示词组装")
    print("=" * 60)
    
    llm = VLLMClient(ModelConfig(model="your-model-name"))
    
    # 系统角色模板
    system_role = """你是一个{role},具有以下特点:
{characteristics}

请遵循以下原则:
{principles}"""
    
    # 任务模板
    task_template = """## 任务背景
{background}

## 具体要求
{requirements}

## 输入内容
{content}

## 输出格式
{output_format}"""
    
    # 组装完整提示词
    full_template = PromptTemplate(
        f"{system_role}\n\n{task_template}"
    )
    
    chain = LLMChain(llm, full_template)
    
    # 执行
    result = chain.run(
        role="高级Python开发工程师",
        characteristics="- 10年开发经验\n- 精通设计模式\n- 注重代码质量",
        principles="- 代码简洁\n- 性能优先\n- 可维护性强",
        background="需要重构一段遗留代码",
        requirements="1. 提高可读性\n2. 优化性能\n3. 添加类型注解",
        content="def calc(a,b,c): return a+b*c",
        output_format="返回重构后的代码和改进说明"
    )
    
    print(result)
    print()


def main():
    """运行所有示例"""
    # 同步示例
    example_1_basic_prompt()
    # example_2_multi_step()  # 取消注释以运行
    # example_3_output_parser()  # 取消注释以运行
    # example_6_complex_assembly()  # 取消注释以运行
    
    # 异步示例
    # asyncio.run(example_4_concurrent())  # 取消注释以运行
    # asyncio.run(example_5_batch_processing())  # 取消注释以运行


if __name__ == "__main__":
    main()
