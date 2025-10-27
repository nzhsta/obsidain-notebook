#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：llm-train 
@File    ：Inference_KPDD.py
@IDE     ：PyCharm 
@Author  ：ZhangHan
@Number  ： 413266
@Date    ：2025/7/2 10:48 
'''
import copy
from collections import defaultdict

from chains.rlhf_generation.config.generation_config import CONCURRENT_LIMIT

from asyncio import Semaphore
import random
from json import JSONDecodeError
from typing import List, Union, Tuple, Optional, Any

import pandas as pd
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.example_selectors import BaseExampleSelector
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.outputs import Generation
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_core.utils.json import parse_json_markdown
from tqdm import tqdm

from chains.rlhf_generation.generation_prompt.generate_query_prompts import \
    KPDDS_prompt_for_response_generation_template, KPDDS_prompt_for_training_template, \
    kpdd_complex_tasks_inference_prompt_template, kpdd_complex_tasks_inference_prompt_debug_template, \
    kpdd_complex_tasks_inference_only_da_prompt_template
from llms.fastchat_client import FastChatClient
from model_hub.async_parallel_chain_wrapper import ParallelChain
from model_hub.llm_client import ChatClient

import os

file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)


class KPDD_Inferencer:
    def __init__(self, chain: ParallelChain, semaphore: Semaphore, **kwargs):
        self.chain = chain
        self.semaphore = semaphore
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def __call__(self, kpdd_complex_questions: List[str], epochs: int = 1, temperatures: Union[Tuple | List] = None):

        num_questions = len(kpdd_complex_questions)
        chunk_size = CONCURRENT_LIMIT
        num_chunks = num_questions // chunk_size + int(num_questions % chunk_size > 0)
        llm_seed = self.chain.llm.seed
        llm_temperature = self.chain.llm.temperature
        responses = []

        for _ in tqdm(range(epochs), desc=f"一共需要进行{epochs}轮响应生成"):
            if temperatures:
                if isinstance(temperatures, Tuple):
                    assert temperatures[0] >= 0 and temperatures[1] <= 1, "Temperature Value should be between 0.0 and 1.0!"
                    llm_temperature = random.uniform(*temperatures)
                elif isinstance(temperatures, List):
                    llm_temperature = random.choice(temperatures)
                    assert 0 <= llm_temperature <= 1, "Temperature Value should be between 0.0 and 1.0!"
            for i in tqdm(range(num_chunks), desc=f"并发调用大模型中，一共{num_questions}个请求,并发数{chunk_size}"):
                # 并发批处理
                batched_questions = kpdd_complex_questions[i * chunk_size:(i + 1) * chunk_size]
                batched_prompts = self.prepare_input_prompt(batched_questions)
                batched_response = self.chain.parallel_run(batched_prompts, semaphore=self.semaphore,
                                                           llm_temperature=llm_temperature, llm_seed=llm_seed)
                responses.extend(batched_response)
        return responses

    # def save(self,kpdd_complex_questions):
    #     res = []
    #     for kpdds_complex in kpdd_complex_questions:
    #         res.append(kpdds_complex.to_dict)
    #     df = pd.DataFrame(res)
    #     df.to_excel(os.path.join(dir_path, self.output_filepath), index=False)
    def prepare_input_prompt(self, batched_questions: List[str]) -> List:
        """
        根据自己的prompt设计组装并发的请求prompt
        """
        batched_prompts = []
        for question in batched_questions:
            input_variables = {}
            input_variables["question"] = question
            batched_prompts.append(input_variables)
        return batched_prompts

    @classmethod
    def from_llm_type(
            cls,
            from_llm_type: str,
            fewshot_prompt: Optional[BasePromptTemplate] = KPDDS_prompt_for_response_generation_template,
            example_selector: BaseExampleSelector = None,
            **kwargs,
    ):
        if from_llm_type == "fastchat":
            LLM_CONFIG = {
                "model_name": "NingBoComplexModel-14B",
                "template_name": "qwen-7b-chat",
                "model_kwargs": {
                    "temperature": 0.3,
                    "top_p": 0.95,
                    "do_sample": True,
                    "max_new_tokens": 2048,
                    "repetition_penalty": 1.0,
                },
                "service_url": "http://10.31.24.3:1978/worker_generate"
            }

            llm = FastChatClient(
                service_url=LLM_CONFIG["service_url"], streaming=False,
                model_name=LLM_CONFIG["model_name"], template_name=LLM_CONFIG["template_name"],
                model_kwargs=LLM_CONFIG["model_kwargs"],
                callbacks=[StdOutCallbackHandler()]
            )

        else:
            assert from_llm_type in ["qwen", "deepseek", "qwen3", "glm", "dpo", "sft"], "指定的模型类型不在范围内"
            llm = ChatClient.from_model_type(from_llm_type).configurable_fields(
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
        semaphore = Semaphore(CONCURRENT_LIMIT)

        class CustomOutputParser(JsonOutputParser):
            def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
                """Parse the result of an LLM call to a JSON object.

                Args:
                    result: The result of the LLM call.
                    partial: Whether to parse partial JSON objects.
                        If True, the output will be a JSON object containing
                        all the keys that have been returned so far.
                        If False, the output will be the full JSON object.
                        Default is False.

                Returns:
                    The parsed JSON object.

                Raises:
                    OutputParserException: If the output is not valid JSON.
                """
                text = result[0].text
                text = text.strip()
                if partial:
                    try:
                        return parse_json_markdown(text)
                    except JSONDecodeError:
                        return None
                else:
                    try:
                        return parse_json_markdown(text)
                    except JSONDecodeError as e:
                        msg = f"Invalid json output: {text}"
                        return msg

        output_parser = CustomOutputParser()
        fewshot_chain = ParallelChain(llm=llm, prompt=fewshot_prompt, output_parser=output_parser)
        return cls(chain=fewshot_chain,
                   semaphore=semaphore,
                   **kwargs)


import json
import argparse


def main(
        model_type="dpo",
        input_file=None,
        output_file=None,
        output_column=None,
        prompt_template=None,
        max_samples=None,
        fewshot_prompt=None,
        output_filepath=None
):
    """
    执行模型推理并保存结果到Excel文件。

    Args:
        model_type (str): 模型类型，如 'dpo', 'sft' 等。
        input_file (str): 输入Excel文件路径。
        output_file (str): 输出Excel文件路径。
        output_column (str): 要写入结果的列名。
        prompt_template (str or None): 提示模板内容（可选）。
        max_samples (int): 最大样本数用于推理。
        fewshot_prompt: 少样本提示模板对象。
        output_filepath: 推理器输出保存路径。
    """
    # 初始化推理器
    inferencer = KPDD_Inferencer.from_llm_type(
        model_type,
        fewshot_prompt=fewshot_prompt or kpdd_complex_tasks_inference_prompt_template,
        output_filepath=output_filepath
    )
    tmp_list = []
    if isinstance(input_file, str):
        test_data = pd.read_excel(input_file)
    else:
        for f in input_file:
            df = pd.read_excel(f)
            df['source_file'] = os.path.basename(f)
            # tmp_list = [pd.read_excel(f) for f in input_file]
            tmp_list.append(df)
        test_data = pd.concat(tmp_list, axis=0, ignore_index=True)
    # 读取测试数据
    # 执行推理（仅前max_samples条）
    # questions = test_data["question"].head(max_samples).tolist()
    questions = test_data["question"].tolist()
    responses = inferencer(questions)

    # 填充剩余行（未推理部分

    # 写入JSON格式结果
    test_data[output_column] = [
        json.dumps(response, ensure_ascii=False, indent=4)
        for response in responses
    ]
    test_data['initial_value'] = test_data[output_column].copy()
    for i in range(len(test_data[output_column])):
        rows = json.loads(test_data[output_column][i])
        parsed = []
        for row in rows:
            try:
                tool = row['tools']
            except KeyError:
                pass
            for t in tool:
                if '--' in t:
                    parsed.append(t.split("--", 1)[1])
                if '-' in t and '--' not in t:
                    parsed.append(t.split("-", 1)[1])
                if '-' not in t and '--' not in t:
                    parsed.append(t)
        result = ",".join([item for item in parsed if item])
        import re
        if "事故信息" in result:
            # result = result.replace("事故信息", "交通事故信息")
            result = re.sub(r'(?<!交通)事故信息', '交通事故信息', result)
        test_data.loc[i, output_column] = result
        if test_data["gold_table"][i] == "车辆和驾驶员基础信息" and test_data[output_column][i] in ["驾驶员基础信息",
                                                                                                    "机动车基础信息"]:
            test_data.loc[i, output_column] = "车辆和驾驶员基础信息"
        if str(test_data.loc[i, "gold_table"]) in str(test_data.loc[i, output_column]):
            test_data.loc[i, output_column + "得分"] = 1
        else:
            test_data.loc[i, output_column + "得分"] = 0
    accuracy = test_data[output_column + "得分"].sum() / len(test_data)
    # 保存结果
    test_data.to_excel(output_filepath, index=False)
    print(f'样本准确率为{accuracy}')
    print(f"推理完成，结果已保存至: {output_filepath}")


if __name__ == "__main__":
    from pathlib import Path

    parser = argparse.ArgumentParser(description="运行任务规划推理流程")
    parser.add_argument("--model_type", type=str, default="sft", help="模型类型 (如 dpo, sft)")
    parser.add_argument(
        "--input_file",
        type=str,
        default=("/data1/home/yi_yx/chains/rlhf_generation/output/new/宁波500测试用例.xlsx",
                 "/data1/home/yi_yx/chains/rlhf_generation/output/new/20250730-版本转测测试用例-单独测试.xlsx"),
        nargs="+",  # 接收 1..N 个路径
        help="输入一个或多个 Excel 文件路径（空格分隔）"
    )

    parser.add_argument("--output_column", type=str, default='9.2_da_lambda0.5', help="结果写入的列名")
    parser.add_argument("--max_samples", type=int, help="推理样本数量")
    parser.add_argument("--output_filepath", type=str,
                        default="/data1/home/zhang_han/chains/rlhf_generation/output/base/综合_0917_only_da_qwen3.xlsx",
                        help="推理输出保存路径")

    args = parser.parse_args()
    # main(
    #     model_type=args.model_type,
    #     input_file=args.input_file,
    #     # output_file=args.output_file,
    #     output_column=args.output_column,
    #     max_samples=args.max_samples,
    #     output_filepath=args.output_filepath,
    #     fewshot_prompt=kpdd_complex_tasks_inference_only_da_prompt_template
    #     # 假设 prompt_template 已在全局定义，如 kpdd_complex_tasks_inference_prompt_template
    # )

    import json

    file_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(file_path)
    inferencer = KPDD_Inferencer.from_llm_type(
        "sft", fewshot_prompt=kpdd_complex_tasks_inference_only_da_prompt_template
        , output_filepath="output/jiaoguan_base/complex_task_test_data_20250718_result.xlsx"
    )
    ## 单个问题
    df = pd.read_excel("/data1/home/yi_yx/chains/rlhf_generation/output/new/ci_test_data_20250807_result.xlsx")
    queries = df["query"].to_list()
    queries = ["根据处理机关代码，统计现场违法数量的周环比、年同比",
               "根据处理机关代码，统计4月份现场违法数量，与上一个月环比，和去年同比",
               "统计4月份现场一般事故数量，与上一个月环比，和去年同比",
               "查询本月事故路段TOP5并展示详情",
               "查询一个月出现天数≥20天且出现时间在0：00~6：00的面包车，且该车近3个月存在4次闯红灯违法记录的车辆",
               "按照时间段、区域名称分组，道路上被抓拍到的车辆数量降序取前TOPN",
               "根据车牌种类和车牌号码，统计各道路的车辆数，并给出对应的车牌明细",
               "找出近一个月存在严重违法或累积记分大于12分的车辆",
               "找出近一个月存在严重违法的且累积记分大于12分的车辆",
               "查询3年内初次考取驾照且名下没有车辆的驾驶员",
               "查询近一个月有车检记录但是车检前2天和车检当天没有被抓拍到的车辆",
               ]
    responses = inferencer(queries)
    df["result"] = [json.dumps(ele, ensure_ascii=False, indent=4) for ele in responses]
    df.to_excel("/data1/home/yi_yx/chains/rlhf_generation/output/new/ci_test_data_20250828_result.xlsx", index=False)

    #
    inferencer = KPDD_Inferencer.from_llm_type(
        "qwen", fewshot_prompt=kpdd_complex_tasks_inference_prompt_template
        , output_filepath="output/jiaoguan_base/complex_task_sft_val_20250804.xlsx"
    )
    test_data = pd.read_excel(os.path.join(dir_path, "output/jiaoguan_base/新增测试用例_250801_result.xlsx"))


    def extract_tools(row):
        response = eval(row["response_sft"])
        return response[0]["tools"]


    test_data["tools"] = test_data.apply(extract_tools, axis=1)
    responses = inferencer(test_data["question"].tolist())
    #
    # test_data["response_sft"] = [ json.dumps(response,ensure_ascii=False,indent = 4)   for response in responses]
    test_data.to_excel(os.path.join(dir_path, "output/jiaoguan_base/新增测试用例_250801_result.xlsx"), index=False)
    print()
    # 　找出现有提示词中，存在问题的用例
    single_step_list = ["./output/jiaoguan_base/data_analysis_query_20250806_drop_duplicate.xlsx",
                        "./output/jiaoguan_base/DS_query_20250716_drop_duplicate.xlsx",
                        "./output/jiaoguan_base/knowledge_database_query_20250716_drop_duplicate.xlsx",
                        "./output/jiaoguan_base/text2image_query_20250716_drop_duplicate.xlsx"]
    multi_steps_list = ["./output/jiaoguan_base/generated_complex_task_response_20250806_cleaned_filtered2.xlsx"]

    inferencer = KPDD_Inferencer.from_llm_type(
        "sft", fewshot_prompt=kpdd_complex_tasks_inference_prompt_template
        , output_filepath="output/jiaoguan_base/complex_task_test_data_20250718_result.xlsx"
    )
    # def is_correct(row):
    #     responses = row["response_sft"]
    #     try:
    #         response = json.loads(responses)
    #         if len(response) == 1 and response[0]["planId"] == "1" :
    #             tools = response[0]["tools"]
    #             if f"{row['topic']}--{row['child_topic']}" in tools or f"{row['topic']}" in tools:
    #                 return True
    #     except Exception as e:
    #         pass
    #     return False
    #
    # single_step_df = []
    # rollout_epoch = 1
    # for _ in range(rollout_epoch):
    #     for file_name in single_step_list:
    #         test_df = pd.read_excel(os.path.join(dir_path,file_name))
    #         responses = inferencer(test_df["question"].tolist())
    #         test_df["response_sft"] = [json.dumps(response, ensure_ascii=False, indent=4) for response in responses]
    #         test_df["is_correct"] = False
    #         test_df["is_correct"] = test_df.apply(is_correct,axis = 1)
    #         test_df = test_df.loc[ test_df["is_correct"] == False,["question","topic","child_topic","grandson_topic","solution_key_points","used_fields","response_key_information","out_join_fields","response_sft"]]
    #         single_step_df.append(test_df)
    # problem_df = pd.concat(single_step_df,axis= 0).drop_duplicates("question",keep='first')
    # problem_df.to_excel(os.path.join(dir_path,"./output/jiaoguan_base/single_step_problem_cases_0807.xlsx"),index = False)

    mapping = dict(
        [
            ("交通警情", "知识库"),
            ("管理制度", "知识库"),
            ("交通导则规范", "知识库"),
            ("通知公告", "知识库"),
            ("智慧交通与技术创新", "DeepSeek"),
            ("应急预案与突发事件处理", "DeepSeek"),
            ("通用问题", "DeepSeek"),
            ("交通事故处理", "DeepSeek"),
            ("交通管理与便民服务", "DeepSeek"),
            ("交通法规与政策", "DeepSeek"),
            ("交通安全教育", "DeepSeek"),
            ("技术支持与系统维护", "DeepSeek"),
        ]
    )


    def is_correct_multi_step(row):
        responses = row["response_sft"]
        try:
            response = json.loads(responses)
            subtopics = eval(row['subtopics'])
            subtopics = [mapping.get(ele, ele) for ele in subtopics if ele != "同环比计算"]
            subtopics = set(subtopics)
            response_topics = sum([copy.deepcopy(t["tools"]) for t in response], [])
            response_topics = set([ele.split("--")[-1] for ele in response_topics])
            if subtopics - response_topics:
                return False
            else:
                return True
        except Exception as e:
            pass
        return False


    rollout_epoch = 1
    single_step_df = []
    multi_steps_list = ["./output/jiaoguan_base/新增测试用例_250801.xlsx"]
    for _ in range(rollout_epoch):
        for file_name in multi_steps_list:
            test_df = pd.read_excel(os.path.join(dir_path, file_name))
            responses = inferencer(test_df["question"].tolist())
            test_df["response_sft"] = [json.dumps(response, ensure_ascii=False, indent=4) for response in responses]
            test_df["is_correct"] = False
            test_df["is_correct"] = test_df.apply(is_correct_multi_step, axis=1)
            test_df = test_df.loc[test_df["is_correct"] == False, :]
            single_step_df.append(test_df)
    problem_df = pd.concat(single_step_df, axis=0).drop_duplicates("question", keep='first')
    problem_df.to_excel(os.path.join(dir_path, "./output/jiaoguan_base/multi_step_problem_cases_0808_added.xlsx"),
                        index=False)
