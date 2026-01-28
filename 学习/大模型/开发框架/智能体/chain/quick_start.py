"""
快速开始 - 3分钟上手LangChain风格的vLLM框架
"""

import asyncio
from langchain_vllm import (
    VLLMClient,
    ModelConfig,
    PromptTemplate,
    LLMChain,
    create_chain
)


# ============ 1. 基础使用 ============
def quick_start_basic():
    """最简单的使用方式"""
    print("1️⃣  基础使用")
    print("-" * 40)
    
    # 创建客户端
    llm = VLLMClient(ModelConfig(
        model="Qwen2.5-7B-Instruct",  # 改成你的模型名
        temperature=0.7
    ))
    
    # 直接调用
    response = llm.invoke("什么是Python?")
    print(response[:100], "...\n")


# ============ 2. 提示词模板 ============
def quick_start_template():
    """使用提示词模板"""
    print("2️⃣  提示词模板")
    print("-" * 40)
    
    llm = VLLMClient(ModelConfig(model="your-model-name"))
    
    # 方式1: 直接创建链
    chain = create_chain(
        llm,
        template="将'{text}'翻译成{lang}"
    )
    result = chain.run(text="Hello World", lang="中文")
    print(f"翻译: {result}\n")


# ============ 3. 流式输出 ============
def quick_start_stream():
    """流式输出 - 实时显示"""
    print("3️⃣  流式输出")
    print("-" * 40)
    
    llm = VLLMClient(ModelConfig(model="your-model-name"))
    
    print("AI: ", end="", flush=True)
    for chunk in llm.stream("写一首关于代码的诗"):
        print(chunk, end="", flush=True)
    print("\n")


# ============ 4. 并发调用 ============
async def quick_start_concurrent():
    """并发处理多个请求"""
    print("4️⃣  并发调用")
    print("-" * 40)
    
    llm = VLLMClient(ModelConfig(model="your-model-name"))
    
    # 批量问题
    questions = [
        "什么是Docker?",
        "什么是Kubernetes?",
        "什么是微服务?"
    ]
    
    # 并发调用(最多3个并发)
    results = await llm.abatch(questions, concurrency=3)
    
    for q, a in zip(questions, results):
        print(f"Q: {q}")
        print(f"A: {a[:50]}...\n")


# ============ 5. 完整示例 ============
async def complete_example():
    """完整工作流示例"""
    print("5️⃣  完整示例: 代码审查助手")
    print("-" * 40)
    
    llm = VLLMClient(ModelConfig(model="your-model-name"))
    
    # 定义审查模板
    review_template = PromptTemplate("""
作为代码审查专家,请审查以下{language}代码:

```{language}
{code}
```

重点关注: {focus}

请提供:
1. 主要问题
2. 改进建议
3. 最佳实践
""")
    
    # 创建审查链
    review_chain = LLMChain(llm, review_template)
    
    # 并发审查多段代码
    code_samples = [
        {
            "language": "python",
            "code": "def calc(a,b): return a+b",
            "focus": "可读性和类型注解"
        },
        {
            "language": "python", 
            "code": "x = [i for i in range(1000000)]",
            "focus": "性能和内存使用"
        }
    ]
    
    # 并发执行
    tasks = [review_chain.arun(**sample) for sample in code_samples]
    results = await asyncio.gather(*tasks)
    
    for i, result in enumerate(results, 1):
        print(f"\n--- 代码 {i} 审查结果 ---")
        print(result[:150], "...\n")


# ============ 主函数 ============
def main():
    """运行所有示例"""
    # 同步示例
    quick_start_basic()
    quick_start_template()
    quick_start_stream()
    
    # 异步示例
    asyncio.run(quick_start_concurrent())
    asyncio.run(complete_example())
    
    print("=" * 60)
    print("🎉 全部完成! 现在你可以:")
    print("  📖 查看 examples.py 了解更多用法")
    print("  🚀 查看 concurrent_examples.py 学习并发技巧")
    print("  📚 阅读 README.md 获取完整文档")
    print("=" * 60)


if __name__ == "__main__":
    main()
