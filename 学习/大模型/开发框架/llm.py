from dataclasses import dataclass
from typing import List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI, AsyncOpenAI
from langchain.prompts import PromptTemplate
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm


@dataclass
class LLMConfig:
    base_url: str = 'http://10.31.16.32:1994/v1'
    model: str = '32b-base'
    max_tokens: int = 10024
    temperature: float = 0.3
    top_p: float = 0.3
    stream: bool = False
    thinking: bool = False
    system_prompt: str = "你是一个有用的帮手"
    timeout: float = 360.0
    max_retries: int = 3

    def get_api_params(self) -> dict:
        """返回 API 调用所需的参数（不含 base_url）"""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }


class VLLMClient:
    def __init__(self, config: LLMConfig, prompt_template: PromptTemplate):
        self.config = config
        self.prompt_template = prompt_template
        self._sync_client: Optional[OpenAI] = None
        self._async_client: Optional[AsyncOpenAI] = None

    @property
    def sync_client(self) -> OpenAI:
        """懒加载同步客户端"""
        if self._sync_client is None:
            self._sync_client = OpenAI(
                base_url=self.config.base_url,
                api_key='EMPTY',
                timeout=self.config.timeout
            )
        return self._sync_client

    @property
    def async_client(self) -> AsyncOpenAI:
        """懒加载异步客户端"""
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                base_url=self.config.base_url,
                api_key='EMPTY',
                timeout=self.config.timeout
            )
        return self._async_client


    def _build_messages(self, prompt: str) -> List[dict]:
        """构建消息列表"""
        return [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": prompt}
        ]

    def _get_extra_body(self) -> dict:
        """获取额外请求参数"""
        return {"chat_template_kwargs": {"enable_thinking": self.config.thinking}}

    def invoke(self, **kwargs) -> str:
        """同步调用"""
        prompt = self.prompt_template.format(**kwargs)
        messages = self._build_messages(prompt)

        response = self.sync_client.chat.completions.create(
            **self.config.get_api_params(),
            messages=messages,
            extra_body=self._get_extra_body()
        )
        return response.choices[0].message.content

    async def ainvoke(self, **kwargs) -> str:
        """异步调用"""
        prompt = self.prompt_template.format(**kwargs)
        messages = self._build_messages(prompt)

        response = await self.async_client.chat.completions.create(
            **self.config.get_api_params(),
            messages=messages,
            extra_body=self._get_extra_body()
        )
        return response.choices[0].message.content

    def concurrent_call(
            self,
            query_list: List[str],
            max_workers: int = 5,
            show_progress: bool = True,
            desc: str = "Processing",
            **kwargs
    ) -> List[str]:
        """
        批量并发调用（线程池实现）

        Args:
            query_list: 查询列表
            max_workers: 最大线程数
            show_progress: 是否显示进度条
            desc: 进度条描述
            **kwargs: 传递给 prompt_template 的其他参数

        Returns:
            结果列表，顺序与输入一致
        """
        results = [None] * len(query_list)
        success_count = 0
        error_count = 0

        def _invoke_single(index: int, query: str) -> tuple:
            try:
                params = dict(kwargs)
                params['input'] = query
                result = self.invoke(**params)
                return index, result, True
            except Exception as e:
                return index, f"[ERROR] {type(e).__name__}: {e}", False

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_invoke_single, i, q): i
                for i, q in enumerate(query_list)
            }

            # 初始化结果列表（保持顺序）
            results = [None] * len(query_list)
            success_count = error_count = 0

            # 使用 tqdm 包装 as_completed
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(
                    iterator,
                    total=len(query_list),
                    desc="🚀 LLM 推理中",
                    unit="req",
                    ncols=100,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
                )

            for future in iterator:
                idx = futures[future]  # 获取原始索引
                result, is_success = future.result()
                results[idx] = result

                if is_success:
                    success_count += 1
                else:
                    error_count += 1

                # 关键修复：使用 set_postfix 字典形式，避免格式混乱
                if show_progress:
                    iterator.set_postfix({
                        "✓ 成功": success_count,
                        "✗ 失败": error_count,
                        "成功率": f"{success_count / (success_count + error_count) * 100:.1f}%"
                    }, refresh=True)

        return results

    async def concurrent_acall(
            self,
            query_list: List[str],
            max_concurrency: int = 5,
            show_progress: bool = True,
            desc: str = "Processing",
            **kwargs
    ) -> List[str]:
        """
        异步并发调用

        Args:
            query_list: 查询列表
            max_concurrency: 最大并发数
            show_progress: 是否显示进度条
            desc: 进度条描述
            **kwargs: 传递给 prompt_template 的其他参数

        Returns:
            结果列表，顺序与输入一致
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        results = [None] * len(query_list)

        # 用于统计成功/失败数量
        stats = {"success": 0, "error": 0}
        stats_lock = asyncio.Lock()

        # 创建进度条
        pbar = None
        if show_progress:
            pbar = tqdm(
                total=len(query_list),
                desc=desc,
                unit="req",
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            )

        async def _invoke_with_sem(index: int, query: str):
            async with semaphore:
                try:
                    params = dict(kwargs)
                    params['input'] = query
                    result = await asyncio.wait_for(
                        self.ainvoke(**params),
                        timeout=self.config.timeout
                    )
                    is_success = True
                except asyncio.TimeoutError:
                    result = "[ERROR] TimeoutError"
                    is_success = False
                except Exception as e:
                    result = f"[ERROR] {type(e).__name__}: {e}"
                    is_success = False

                # 更新结果
                results[index] = result

                # 更新统计和进度条
                async with stats_lock:
                    if is_success:
                        stats["success"] += 1
                    else:
                        stats["error"] += 1

                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix_str(f"✔-{stats['success']} ❌-{stats['error']}")

        # 创建所有任务
        tasks = [_invoke_with_sem(i, q) for i, q in enumerate(query_list)]

        # 并发执行
        await asyncio.gather(*tasks)

        # 关闭进度条
        if pbar:
            pbar.close()

        return results

    def close(self):
        """关闭客户端连接"""
        if self._sync_client:
            self._sync_client.close()

    async def aclose(self):
        """异步关闭客户端连接"""
        if self._async_client:
            await self._async_client.close()


# ============ 使用示例 ============

async def main():
    template_clean = """
        你是个专业的文本总结助手，请根据输入，进行内容提炼，并格式化输出
        ### 任务目标
        任务的目标是什么
        ### 
        简单描述用户的查询目的
        ### 决策步骤
        <基于观察进行逻辑分析，最多3个递进论点，每个论点以(1)(2)(3)标记>
        ### 推理过程
        <连接分析到结论的逻辑桥梁，使用"→"符号表示推导关系>
        ### 结论
        <给出最终判断>
         **要求**：
        - 不允许出现子标题或嵌套列表
        - 总字数控制在300字以内

        输出格式：
        任务目标：
        用户查询目的：
        决策步骤：
            (1) ...
            (2) ...
            ...
        推理过程：
        结论：
        
       

        输入：{{input}}
        """

    clean_prompt = PromptTemplate.from_template(
        template=template_clean,
        template_format="jinja2"
    )

    config = LLMConfig(
        timeout=360.0,
        system_prompt="你是一个专业的文本处理助手"
    )

    client = VLLMClient(config=config, prompt_template=clean_prompt)

    try:
        data = pd.read_excel(
            '/home/jarven/workspace/qwen3MindieTest/output/32b-base/batch/full/'
            '20260128/自验/NVIDIA/4B-50_detailed_20260128_192000_fix.xlsx'
        )
        query_list = data['think'].tolist()

        print(f"共 {len(query_list)} 条数据待处理\n")

        # 异步并发调用（带进度条）
        results = await client.concurrent_acall(
            query_list=query_list,
            max_concurrency=50,
            show_progress=True,
            desc="🚀 LLM 推理中"
        )

        # 统计结果
        success = sum(1 for r in results if not r.startswith("[ERROR]"))
        failed = len(results) - success

        print(f"\n处理完成: 成功 {success} 条, 失败 {failed} 条")

        # 保存结果
        data['summary'] = results
        output_path = '/home/jarven/workspace/output_with_summary.xlsx'
        data.to_excel(output_path, index=False)
        print(f"结果已保存至: {output_path}")

    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())

