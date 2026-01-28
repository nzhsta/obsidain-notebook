"""
并发调用示例 - 展示高性能批量处理
"""

import asyncio
import time
from langchain_vllm import VLLMClient, ModelConfig, PromptTemplate, LLMChain


async def example_async_batch():
    """异步批量调用 - 推荐方式"""
    print("=" * 60)
    print("异步批量调用 (推荐)")
    print("=" * 60)
    
    llm = VLLMClient(ModelConfig(model="your-model-name"))
    
    # 准备100个问题
    prompts = [f"用一句话解释: 什么是编程概念#{i}?" for i in range(20)]
    
    # 计时
    start = time.time()
    
    # 并发调用 (最多5个并发)
    results = await llm.abatch(prompts, concurrency=5)
    
    elapsed = time.time() - start
    
    print(f"✓ 处理 {len(prompts)} 个请求")
    print(f"✓ 总耗时: {elapsed:.2f}秒")
    print(f"✓ 平均: {elapsed/len(prompts):.2f}秒/个")
    print(f"\n前3个结果:")
    for i, result in enumerate(results[:3], 1):
        print(f"{i}. {result[:60]}...")
    print()


def example_thread_batch():
    """线程池批量调用 - 同步方式"""
    print("=" * 60)
    print("线程池批量调用 (同步方式)")
    print("=" * 60)
    
    llm = VLLMClient(ModelConfig(model="your-model-name"))
    
    prompts = [f"简单解释: 概念{i}" for i in range(10)]
    
    start = time.time()
    results = llm.batch(prompts, max_workers=3)
    elapsed = time.time() - start
    
    print(f"✓ 处理 {len(prompts)} 个请求")
    print(f"✓ 总耗时: {elapsed:.2f}秒")
    print(f"✓ 使用线程池: 3个worker")
    print()


async def example_chain_concurrent():
    """链式并发 - 多个不同任务并发"""
    print("=" * 60)
    print("链式并发处理")
    print("=" * 60)
    
    llm = VLLMClient(ModelConfig(model="your-model-name"))
    
    # 定义不同的任务链
    tasks = [
        LLMChain(llm, PromptTemplate("总结Python的特点")).arun(),
        LLMChain(llm, PromptTemplate("总结JavaScript的特点")).arun(),
        LLMChain(llm, PromptTemplate("总结Go的特点")).arun(),
        LLMChain(llm, PromptTemplate("总结Rust的特点")).arun(),
    ]
    
    start = time.time()
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start
    
    print(f"✓ 并发完成 {len(tasks)} 个不同任务")
    print(f"✓ 总耗时: {elapsed:.2f}秒")
    
    languages = ["Python", "JavaScript", "Go", "Rust"]
    for lang, result in zip(languages, results):
        print(f"\n{lang}: {result[:80]}...")
    print()


async def example_dynamic_concurrency():
    """动态并发控制"""
    print("=" * 60)
    print("动态并发控制")
    print("=" * 60)
    
    llm = VLLMClient(ModelConfig(model="your-model-name"))
    
    # 测试不同并发度
    prompts = [f"问题{i}" for i in range(15)]
    
    for concurrency in [1, 3, 5]:
        start = time.time()
        await llm.abatch(prompts, concurrency=concurrency)
        elapsed = time.time() - start
        
        print(f"并发度={concurrency}: {elapsed:.2f}秒")
    print()


async def example_streaming_concurrent():
    """流式输出 + 并发"""
    print("=" * 60)
    print("流式输出并发")
    print("=" * 60)
    
    llm = VLLMClient(ModelConfig(model="your-model-name"))
    
    async def stream_and_collect(prompt: str, label: str):
        """流式收集结果"""
        print(f"\n[{label}] 开始生成...")
        chunks = []
        async for chunk in llm.astream(prompt):
            chunks.append(chunk)
        result = "".join(chunks)
        print(f"[{label}] 完成! 长度: {len(result)}")
        return result
    
    # 并发流式调用
    tasks = [
        stream_and_collect("写一个Python快速排序", "任务1"),
        stream_and_collect("写一个Python二分查找", "任务2"),
        stream_and_collect("写一个Python深度优先搜索", "任务3")
    ]
    
    results = await asyncio.gather(*tasks)
    print(f"\n✓ 所有流式任务完成")
    print()


async def example_error_handling():
    """并发错误处理"""
    print("=" * 60)
    print("并发错误处理")
    print("=" * 60)
    
    llm = VLLMClient(ModelConfig(model="your-model-name"))
    
    async def safe_invoke(prompt: str, index: int):
        """安全调用,捕获错误"""
        try:
            result = await llm.ainvoke(prompt)
            return {"index": index, "success": True, "result": result}
        except Exception as e:
            return {"index": index, "success": False, "error": str(e)}
    
    # 批量调用(包含可能失败的请求)
    prompts = [f"问题{i}" for i in range(5)]
    tasks = [safe_invoke(p, i) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
    
    # 统计结果
    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count
    
    print(f"✓ 成功: {success_count}")
    print(f"✗ 失败: {fail_count}")
    print()


async def benchmark_comparison():
    """性能对比: 串行 vs 并发"""
    print("=" * 60)
    print("性能对比")
    print("=" * 60)
    
    llm = VLLMClient(ModelConfig(model="your-model-name"))
    prompts = [f"简答{i}" for i in range(10)]
    
    # 串行执行
    print("串行执行...")
    start = time.time()
    for prompt in prompts:
        await llm.ainvoke(prompt)
    serial_time = time.time() - start
    
    # 并发执行
    print("并发执行...")
    start = time.time()
    await llm.abatch(prompts, concurrency=5)
    concurrent_time = time.time() - start
    
    # 结果
    speedup = serial_time / concurrent_time
    print(f"\n串行耗时: {serial_time:.2f}秒")
    print(f"并发耗时: {concurrent_time:.2f}秒")
    print(f"加速比: {speedup:.2f}x")
    print()


async def main():
    """运行所有示例"""
    # 基础示例
    await example_async_batch()
    
    # 更多示例(取消注释运行)
    # example_thread_batch()
    # await example_chain_concurrent()
    # await example_dynamic_concurrency()
    # await example_streaming_concurrent()
    # await example_error_handling()
    # await benchmark_comparison()


if __name__ == "__main__":
    asyncio.run(main())
