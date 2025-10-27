#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：llm-train 
@File    ：async_parallel_chain.py
@IDE     ：PyCharm 
@Author  ：ZhangHan
@Number  ： 413266
@Date    ：2025/5/30 16:41 
'''



import asyncio
import time
from asyncio import Semaphore
import random
from typing import Dict, List, Tuple

from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables import ConfigurableField

from model_hub.llm_client import ChatClient, BaseLLMClient

# Control concurrency level
CONCURRENT_LIMIT = 40

class ParallelChain(LLMChain):
    """
    对Chain进行包装，增加异步并发调用不同prompt的功能。
    """

    def parallel_run(self, combinations, semaphore: Semaphore = None,**llm_kwargs):
        llm_temperature = llm_kwargs.get("llm_temperature",None)
        llm_seed = llm_kwargs.get("llm_seed", None)
        # # 修改温度和随机种子
        # if llm_temperature or llm_seed:
        #     self.llm = self.llm.with_config(configurable={"llm_temperature": llm_temperature or self.llm.temperature , "llm_seed": llm_seed or self.llm.seed})
        async def async_generate(llm_temperature,llm_seed,**kwargs):
            if llm_temperature:
                if isinstance(llm_temperature, Tuple):
                    assert llm_temperature[0] >= 0 and llm_temperature[
                        1] <= 1, "Temperature Value should be between 0.0 and 1.0!"
                    llm_temperature = random.uniform(*llm_temperature)
                elif isinstance(llm_temperature, List):
                    llm_temperature = random.choice(llm_temperature)
                    assert 0 <= llm_temperature <= 1, "Temperature Value should be between 0.0 and 1.0!"
            seed = llm_seed or random.randint(1, 200)
            async with semaphore:
                self.llm = self.llm.with_config(
                    configurable={"llm_temperature": llm_temperature,
                                  "llm_seed": seed})
                resp = await self.arun(**kwargs)
                return resp

        async def generate_concurrently(llm_temperature,llm_seed,combinations):
            tasks = [async_generate(llm_temperature,llm_seed,**combo) for combo in combinations]
            return await asyncio.gather(*tasks)

        return asyncio.run(generate_concurrently(llm_temperature,llm_seed,combinations))



if __name__ == "__main__":
    llm = ChatClient.from_model_type('qwen').configurable_fields(
        temperature=ConfigurableField(
            id="llm_temperature",
            name="LLM Temperature",
            description="The temperature of the LLM",
        ),
        seed=ConfigurableField(
            id="llm_seed",
            name="LLM seed",
            description="The seed of the LLM",
        ),
    )
    prompt = PromptTemplate(
        input_variables=["var"],
        template="{var}的8次方是多少?",
    )
    # chain_0 = LLMChain(llm=llm, prompt=prompt)
    # resp = chain_0.run(var="2")
    # chain_0.llm = chain_0.llm.with_config(configurable={"llm_temperature": 1.0,"llm_seed": 888})
    # resp2 = chain_0.run(var="2")
    chain = ParallelChain(llm=llm, prompt=prompt)
    chain.llm = chain.llm.with_config(configurable={"llm_temperature": 0.9, "llm_seed": 888})
    results = chain.parallel_run([{"var":"2"},{"var":"3"}],semaphore=Semaphore(CONCURRENT_LIMIT))
    print(results)
