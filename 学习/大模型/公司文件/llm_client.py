#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：llm-train 
@File    ：llm_client.py
@IDE     ：PyCharm 
@Author  ：ZhangHan
@Number  ： 413266
@Date    ：2025/5/30 10:40 
'''
import ast
import json
import time
import yaml

from openai import OpenAI
from typing import Dict
from typing import Optional, List, Any

from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.schema.callbacks.manager import CallbackManagerForLLMRun
from langchain.utils import get_colored_text
from langchain_core.callbacks import StdOutCallbackHandler
from pydantic import Extra

DEEPSEEK_CONTEXT_EXCEPTION = "模型服务上下文能力受限"
RESPONSE_DEEPSEEK_EXCEPTION = "很抱歉，模型服务异常，请稍候再试"
DEEPSEEK_CONTEXT_LEN = 8192
from utils.logger import logger_init

logger = logger_init(__name__)

# 大模型对话模板，这种情况当前只有是使用FastChat部署的形式，才需要使用LLM对话模板
LLM_TEMPLATE_NAME = "DH-BDM-Chat"

class BaseLLMClient(LLM):
    """
        大模型客户端调用基类
    """

    # 模型服务地址
    service_url: Optional[str] = None
    # 是否开启流式
    streaming: bool = False
    # 模型调用参数
    model_kwargs: Optional[dict] = {
        "temperature": 0.2,
        "repetition_penalty": 1.0,
        "top_p": 0.9,
        "top_k": 50,
        "do_sample": True,
        "num_beams": 1,
        "max_new_tokens": 4096,
    }

    # 发送的Prompt是否修改为特殊模板
    is_using_template: bool = False
    # 对话模板
    template_name: Optional[str] = None
    # 模型名称，单实例部署的时候对应的通用模型
    model_name: Optional[str] = None
    #模型温度
    temperature: float = 0.9
    # 模型种子
    seed: int = 123
    # 对话模板
    api_key: Optional[str] = None
    enable_thinking:Optional[bool] = False
    thinking_budget: Optional[int] = -1

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
        # 如果采用对话模板，但没有送入对话模板名称，则设置默认对话模板
        if self.is_using_template is True and \
                self.template_name is None:
            self.template_name = LLM_TEMPLATE_NAME

        # 如果已经设置了对话模板名称，但是对应标志位为False，则改为True
        if self.template_name is not None:
            if self.is_using_template is False:
                self.is_using_template = True

    @property
    def _llm_type(self) -> str:
        return "base_llm_client"

    @staticmethod
    def get_response(data,
                     service_url,
                     timeout=100,
                     **kwargs) -> Dict:
        raise Exception("必须实现get_response")

    def _data_field(self) -> str:
        # 非Chat模式，对应的结果从Text获取
        raise Exception("必须指定choices获取数据的字段")

    @staticmethod
    def get_streaming_response(data,
                               service_url,
                               timeout=100,
                               **kwargs) -> Dict:
        raise Exception("必须实现get_streaming_response")

    def create_params(self,
                      prompt: str,
                      stop: Optional[List[str]] = None,
                      seed: Optional[int] = 42,
                      **kwargs: Any, ) -> Dict:
        raise Exception("必须实现create_params")


    def _call_using_llm(self,
                        params: dict,
                        run_manager: Optional[CallbackManagerForLLMRun] = None,
                        ) -> Dict:
        """
            具体调用大模型
        """
        pass

    def _call_using_think_llm(self,
                    params: dict,
                    run_manager: Optional[CallbackManagerForLLMRun] = None,
                    ) -> Dict:
        """
            具体调用大模型
        """
        pass

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              seed: Optional[int] = 42,
              **kwargs: Any,
              ) -> str:
        _run_manager = run_manager or CallbackManagerForLLMRun.get_noop_manager()

        # 1. 参数处理
        params = self.create_params(prompt=prompt,
                                    stop=self.model_kwargs.pop("stop",None),
                                    seed=self.seed,
                                    temperature= self.temperature,
                                    **self.model_kwargs)

        logger.info(
            f"进行大模型调用, 使用的大模型框架形式： {self._llm_type},  "
            f"调用接口：{self.service_url} ....")
        # 2. 调用大模型
        if run_manager.handlers and getattr(run_manager.handlers[0],"is_think",0) == 1:
            result = self._call_using_think_llm(params=params,
                                        run_manager=_run_manager)
        else:
            result = self._call_using_llm(params=params,
                                        run_manager=_run_manager)

        completion_text = result.get("completion_text", "")
        completion_tokens = result.get("completion_tokens", -1)
        prompt_tokens = result.get("prompt_tokens", -1)
        elapsed_time = result.get("elapsed_time", -1)
        if run_manager.handlers and getattr(run_manager.handlers[0],"is_think",0) == 1:
            completion_tokens = 0
            prompt_tokens = 0

        # 3. 输出截断兜底处理，根据 Stop, 进行输出截断
        if stop is not None:
            completion_text = self.enforce_stop_tokens(completion_text,
                                                       stop)

        # 4. 打印调用信息
        self.print_call_information(prompt_tokens=prompt_tokens,
                                    completion_tokens=completion_tokens,
                                    elapsed_time=elapsed_time,
                                    prompt=prompt,
                                    completion_text=completion_text,
                                    params=params,
                                    run_manager=_run_manager)

        return completion_text

    def print_call_information(self,
                               prompt_tokens: int,
                               completion_tokens: int,
                               elapsed_time: float,
                               prompt: str,
                               completion_text: str,
                               params: dict,
                               run_manager: Optional[CallbackManagerForLLMRun] = None,
                               ):
        """
            打印大模型调用的核心信息
        """
        # 打印调用信息

        # 调用参数
        # 去除Prompt, 避免重复打印
        if "prompt" in params:
            params.pop("prompt")

        if "messages" in params:
            params.pop("messages")

        # 2024.02.20 将下面的2行打印合并为一行打印，这样在前端页面展示上更加友好
        # 打印Params
        run_manager.on_text(
            get_colored_text(f"\n[{'.'.join(self.lc_id())}] "
                             f"Params: \n", "green") +
            get_colored_text(json.dumps(params, ensure_ascii=False), "yellow"), verbose=self.verbose)

        # 打印输入的Prompt信息
        run_manager.on_text(
            get_colored_text(f"\n[{'.'.join(self.lc_id())}] "
                             f"Prompt(#Tokens: {prompt_tokens}): \n", "green") +
            get_colored_text(prompt, "yellow"), verbose=self.verbose)

        # 打印输出的信息
        speed = completion_tokens / elapsed_time
        run_manager.on_text(
            get_colored_text(f"\n[{'.'.join(self.lc_id())}] "
                             f"Completion(#Tokens: {completion_tokens},"
                             f"Speed: {speed:.2f}tokens/s,"
                             f"ElapsedTime: {elapsed_time:.4f}s):\n", "green") +
            get_colored_text(completion_text, "yellow"), end="\n\n", verbose=self.verbose)

    @staticmethod
    def is_chinese_char(cp):
        """
            校验是否包含中文字符
        Args:
            cp:

        Returns:

        """
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
                (cp >= 0x4E00 and cp <= 0x9FFF)
                or (cp >= 0x3400 and cp <= 0x4DBF)  #
                or (cp >= 0x20000 and cp <= 0x2A6DF)  #
                or (cp >= 0x2A700 and cp <= 0x2B73F)  #
                or (cp >= 0x2B740 and cp <= 0x2B81F)  #
                or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
                or (cp >= 0xF900 and cp <= 0xFAFF)
                or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

    def enforce_stop_tokens(self, text: str, stop: List[str]) -> str:
        """Cut off the text as soon as any stop words occur."""
        new_stop = []
        for stop_str in stop:
            stop_str = stop_str.replace("|", "\|")  # 将|进行转义表达
            new_stop.append(stop_str)
        text = enforce_stop_tokens(text, new_stop)

        return text

    def remove_unsupported_params(self, params: dict) -> dict:
        """
            去除不支持的参数
        """
        # TODO：最新版本的MindIE有些参数已经对齐，这里需要改为获取MindIE的版本进行参数控制判断
        # FastChat worker_generate/worker_generate_stream 模式，对标 OpenAI Completions 模式
        # 支持参数(VLL模式)：https://gitee.com/deepeye/FastChat/blob/main/fastchat/serve/vllm_worker.py
        # FastChat底层VLLM支持参数: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        # model
        # prompt
        # temperature
        # top_p
        # top_k
        # max_new_tokens
        # repetition_penalty
        # seed
        # echo
        # 未找到 do_sample 参数

        # OpenAI Completions 支持参数：https://platform.openai.com/docs/api-reference/completions/create
        # model
        # prompt
        # echo
        # frequency_penalty
        # max_tokens
        # presence_penalty
        # seed
        # stop
        # stream
        # temperature
        # top_p

        # OpenAI Chat 支持参数：https://platform.openai.com/docs/api-reference/chat/completions//create
        # model
        # messages
        # frequency_penalty
        # max_tokens
        # max_completion_tokens 等效为 max_new_tokens 参数
        # presence_penalty
        # seed
        # stop
        # stream
        # temperature
        # top_p

        # MindIE RC2 Chat 支持参数：https://www.hiascend.com/document/detail/zh/mindie/10RC2/mindieservice/servicedev/mindie_service0192.html
        # model
        # messages
        # max_tokens
        # presence_penalty
        # frequency_penalty
        # seed
        # temperature
        # top_p
        # stream

        # MindIE RC3 支持参数（当前使用参数）： https://www.hiascend.com/document/detail/zh/mindie/10RC3/mindieservice/servicedev/mindie_service0232.html
        # model
        # messages
        # max_tokens
        # presence_penalty
        # frequency_penalty
        # repetition_penalty
        # stop
        # stream
        # temperature
        # top_p
        # top_k

        # 去除OpenAI调用不支持的参数
        if not self._llm_type == "fastchat_llm_client":
            if "echo" in params:
                # Chat模式没有 echo 参数，华为 MindIE不支持 echo参数
                # 所以除了FastChat框架，其他都默认删除echo参数
                # 对OpenAI的 Completions 模式无法设置，但是默认值也是False，暂不产生影响
                params.pop("echo")
            if "top_k" in params:
                params.pop("top_k")
            if "do_sample" in params:
                params.pop("do_sample")
            if "repetition_penalty" in params:
                params.pop("repetition_penalty")
            # 建议统一使用 max_tokens 参数进行设置
            if "max_new_tokens" in params:
                params.pop("max_new_tokens")
        else:
            # 使用fastchat框架
            if "max_tokens" in params:
                # 转换为 FastChat的参数
                # max_tokens => max_new_tokens
                params["max_new_tokens"] = params.get("max_tokens")
                params.pop("max_tokens")

        # fixme: 后续优化, llm_hub 会多传入的参数
        if "model_name" in params:
            params.pop("model_name")
        if "model_type" in params:
            params.pop("model_type")

        return params


class ChatClient(BaseLLMClient):
    """
        LLM对话模式的 Client
    """

    user_role: str = "user"
    ai_role: str= "assistant"

    @property
    def _llm_type(self) -> str:
        return "chat_client"

    def _data_field(self) -> str:
        # Chat模式，对应的结果从content获取
        return "content"

    def assemble_messages(self, prompt: str):
        """
            组装messages
        """
        # 当前主要是送入一个Prompt的形式，非真正的对话模式
        # 因此只有一轮的对话
        messages = [{
            "role": self.user_role,
            "content": prompt
        }]
        return messages
    def remove_markdown_dividers(self, md_text, split_key):
        index = md_text.find(split_key)
        if index == 0:
            md_text = md_text[len(split_key):]
        elif index > 0:
            md_text = md_text[0:index]
        return md_text
    def create_params(self,
                      prompt: str,
                      stop: Optional[List[str]] = None,
                      seed: Optional[int] = 42,
                      **kwargs: Any, ):
        """
            生成参数
        """
        # 1. 组装params
        kwargs["stop"] = stop
        kwargs["seed"] = seed
        _model_kwargs = self.model_kwargs or {}
        params = {**_model_kwargs, **kwargs}
        #    添加流式参数
        params["stream"] = self.streaming

        # 组装message
        # TODO: 暂时按单轮进行组装
        messages = self.assemble_messages(prompt=prompt)

        # 2. 多Lora/基础+融合模型 部署形式，判断是否特别指定调用的模型
        params.update(dict(
            model=self.model_name,
            messages=messages,
            echo=False,
        ))

        # 3. 去除不支持的params
        self.remove_unsupported_params(params=params)

        return params

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True
        extra_fields_behavior = False

    def _call_using_llm(self,
                        params: dict,
                        run_manager: Optional[CallbackManagerForLLMRun] = None,
                        ):
        """
            调用大模型的，实际获取结果
        """
        # 默认输出
        prompt_tokens = -1
        completion_tokens = -1
        completion_text = ""
        elapsed_time = -1

        try:
            start_time = time.time()
            if self.streaming:
                final_data = None
                for data in self.get_streaming_response(params, self.service_url,
                                                        stream=self.streaming):
                    if completion_text == "":
                        # 表示流式的第一个Token输出
                        streaming_start_elapsed_time = time.time() - start_time
                        logger.info(f"大模型流式输出第一个Token耗时：{streaming_start_elapsed_time:.4f}")
                    # 暂时假设回复只有一轮结果
                    choices = data["choices"]
                    if len(choices) == 0:
                        new_text = "(LLM流式调用失败, 停止输出)"
                        completion_text += new_text
                        if run_manager and new_text:
                            run_manager.on_llm_new_token(
                                completion_text,
                                verbose=self.verbose,
                            )
                        break
                    # 暂时假设回复只有一轮结果
                    new_text = choices[0]["delta"][self._data_field()]
                    completion_text += new_text
                    if run_manager and new_text:
                        run_manager.on_llm_new_token(
                            new_text,
                            verbose=self.verbose,
                        )

                    # 输出最后的输出
                    final_data = data
                if final_data:
                    prompt_tokens = final_data["usage"]["prompt_tokens"]
                    completion_tokens = final_data["usage"]["completion_tokens"]

                # 流式输出结束
                streaming_end_elapsed_time = time.time() - start_time
                elapsed_time = streaming_end_elapsed_time
                logger.info(f"大模型流式输出耗时：{streaming_end_elapsed_time}")
            else:
                data = self.get_response(params, self.service_url, stream=self.streaming,api_key=self.api_key,
                                         enable_thinking=self.enable_thinking,
                                         thinking_budget=self.thinking_budget)
                # 输出结束
                end_elapsed_time = time.time() - start_time
                logger.info(f"大模型式输出耗时：{end_elapsed_time}")
                # 暂时假设回复只有一轮结果
                choices = data["choices"]
                prompt_tokens = data["usage"]["prompt_tokens"]
                completion_tokens = data["usage"]["completion_tokens"]
                elapsed_time = end_elapsed_time
                if len(choices) > 0:
                    # 暂时假设回复只有一轮结果
                    completion_text = choices[0]["message"][self._data_field()]

            return {"completion_text": completion_text,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "elapsed_time":elapsed_time}

        except Exception as exc:
            import traceback
            traceback.print_exc()
            logger.exception(exc)
            run_manager.on_text(f"The Error is: {exc}, it may have exceeded the token limit.")
            return {"completion_text": completion_text,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens}

    def _call_using_think_llm(self,
                        params: dict,
                        run_manager: Optional[CallbackManagerForLLMRun] = None,
                        ):
        """
            调用大模型的，实际获取结果
        """
        # 默认输出
        prompt_tokens = -1
        completion_tokens = -1
        completion_text = ""
        completion_content = ""
        elapsed_time = -1
        reasoning_key = '<think>'
        content_key = '</think>'
        im_start_key = '<|im_start|>'
        assistant_ky = 'assistant'
        end_sentence_key = '<｜end▁of▁sentence｜>'
        end_key = '<|im_end|>'
        split_key = '---'   # prompt消除分割线效果一般，使用后处理进行规避
        deepseek_exceed_limit_tag = False
        try:
            start_time = time.time()
            if self.streaming:
                final_data = None
                if run_manager.handlers[0].is_think == 1:
                    # 使用 ast.literal_eval 将字符串转换为列表
                    converted_list = ast.literal_eval(params["messages"][0]["content"])
                    msg_len = len(params["messages"][0]["content"])
                    if msg_len > DEEPSEEK_CONTEXT_LEN:
                        converted_list = [converted_list[-1]]
                        logger.info(f"精简content:{converted_list[-1]}")
                        if len(converted_list[0]["content"]) > DEEPSEEK_CONTEXT_LEN:
                            deepseek_exceed_limit_tag = True
                            raise Exception
                    think_params = {}
                    think_params["model"] = params["model"]
                    think_params["messages"] = converted_list
                    think_params["stream"] = params["stream"]
                    think_params['temperature'] = params['temperature']
                    think_params['top_p'] =  params['top_p']
                    think_params['seed'] =  params['seed']
                    think_params['max_tokens'] = params['max_tokens']
                try:
                    for data in self.get_streaming_think_response(think_params, self.service_url,
                                                            stream=self.streaming):
                        if completion_text == "":
                            # 表示流式的第一个Token输出
                            streaming_start_elapsed_time = time.time() - start_time
                            logger.info(f"大模型流式输出第一个Token耗时：{streaming_start_elapsed_time},url:{self.service_url}")
                        # 暂时假设回复只有一轮结果
                        choices = data["choices"]
                        if len(choices) == 0:
                            new_text = "(LLM流式调用失败, 停止输出)"
                            completion_text += new_text
                            if run_manager and new_text:
                                run_manager.on_llm_new_token(
                                    completion_text,
                                    verbose=self.verbose,
                                )
                            break
                        # 暂时假设回复只有一轮结果
                        new_text = choices[0]["delta"][self._data_field()]
                        #> 非空校验
                        if run_manager and new_text is None:
                            continue
                        #> bak0
                        # completion_text += new_text
                        if reasoning_key in new_text:
                            new_text = new_text[len(reasoning_key):]
                            # completion_text = new_text
                        elif im_start_key in new_text:
                            new_text = new_text[len(im_start_key):]
                            # completion_text = new_text
                        elif assistant_ky in new_text:
                            new_text = new_text[len(assistant_ky):]
                            # completion_text = new_text
                        #! 处理<|im_end|>
                        index = new_text.find(end_key)
                        im_end_tag = False
                        if index == 0:
                            new_text = new_text[len(end_key):]
                            im_end_tag = True
                        elif index > 0:
                            new_text = new_text[0:index]
                            im_end_tag = True
                        completion_text += new_text
                        parts = completion_text.split(content_key, 1)
                        if run_manager and new_text and len(parts) <= 1:
                            # if '<|im_end|>' in completion_text:
                            #     new_text = new_text.split('<|im_end|>')[0]
                            #> deepseek思维链模式
                            run_manager.on_llm_new_token(
                                new_text,
                                verbose=self.verbose,
                                type="think"
                            )
                            if im_end_tag == True:
                                run_manager.on_llm_new_token(
                                    completion_text,
                                    verbose=self.verbose
                                )
                        # else:
                            # print(140,completion_text,"len(parts):", len(parts))

                        if run_manager and new_text and len(parts) == 2:
                            if new_text:
                                new_text = self.remove_markdown_dividers(new_text, split_key)
                                #! 处理<｜end▁of▁sentence｜>
                                # index = new_text.find(end_sentence_key)
                                # if index == 0:
                                #     # new_text = new_text[0:index]
                                #     new_text = new_text[len(end_sentence_key):]
                                # elif index > 0:
                                #     new_text = new_text[0:index]
                                #> 方法1:修复'ttt<｜end▁of▁sentence｜>'时思维链可能会缺失问题
                                new_text = self.remove_markdown_dividers(new_text, end_sentence_key)

                                # #! 处理<|im_end|>
                                # index = new_text.find(end_key)
                                # if index == 0:
                                #     # new_text = new_text[0:index]
                                #     new_text = new_text[len(end_key):]
                                # elif index > 0:
                                #     new_text = new_text[0:index]

                                #!
                                #> 方法1
                                # if '</think>' in new_text:
                                # # new_text = new_text.split('<think>' )[0]
                                #     new_text = new_text[len('</think>'):]

                                #> 方法2:修复'ttt</think>'时思维链可能会缺失问题
                                index = new_text.find(content_key)
                                if index == 0:
                                    new_text = new_text[len(content_key):]
                                elif index > 0:
                                    new_text = new_text[index+len(content_key):]

                                run_manager.on_llm_new_token(
                                    new_text,
                                    verbose=self.verbose,
                                    type="normal"
                                )
                        # 输出最后的输出
                        final_data = data
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    completion_text = RESPONSE_DEEPSEEK_EXCEPTION
                    run_manager.on_llm_new_token(
                        completion_text,
                        verbose=self.verbose
                    )
                if final_data:
                    if im_end_tag == True:
                        run_manager.on_llm_new_token(
                            completion_text,
                            verbose=self.verbose
                        )
                    # vllm openai流式协议无此字段，宁波现场部署的也无此字段进行特殊处理。
                    # prompt_tokens = final_data["usage"]["prompt_tokens"]
                    # completion_tokens = final_data["usage"]["completion_tokens"]
                    prompt_tokens = 0
                    completion_tokens = 0

                # 流式输出结束
                streaming_end_elapsed_time = time.time() - start_time
                logger.info(f"大模型流式输出耗时：{streaming_end_elapsed_time}")
                elapsed_time = streaming_end_elapsed_time
            else:
                data = self.get_response(params, self.service_url, stream=self.streaming,api_key= self.api_key )
                # 输出结束
                end_elapsed_time = time.time() - start_time
                logger.info(f"大模型非流式输出耗时：{end_elapsed_time}")
                # 暂时假设回复只有一轮结果
                choices = data["choices"]
                prompt_tokens = data["usage"]["prompt_tokens"]
                completion_tokens = data["usage"]["completion_tokens"]
                if len(choices) > 0:
                    # 暂时假设回复只有一轮结果
                    completion_text = choices[0]["message"][self._data_field()]
                elapsed_time = end_elapsed_time

            return {"completion_text": completion_text,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "elapsed_time": elapsed_time}

        except Exception as exc:
            import traceback
            traceback.print_exc()
            logger.exception(exc)
            run_manager.on_text(f"The Error is: {exc}, it may have exceeded the token limit.")
            completion_text = RESPONSE_DEEPSEEK_EXCEPTION
            if deepseek_exceed_limit_tag == True:
                completion_text = DEEPSEEK_CONTEXT_EXCEPTION
            run_manager.on_llm_new_token(
                completion_text,
                verbose=self.verbose
            )
            return {"completion_text": completion_text,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "elapsed_time": elapsed_time}

    @staticmethod
    def get_response(data,
                     service_url,
                     timeout=100,
                     **kwargs):
        client = OpenAI(base_url=service_url.split('v1/')[0] + "v1/", api_key=kwargs.get("api_key",'na') if kwargs.get("api_key",'na') else 'na')
        extra_body = {"chat_template_kwargs": {}}
        if kwargs.get("enable_thinking", None) is not None:
            extra_body["chat_template_kwargs"]["enable_thinking"] = kwargs.get("enable_thinking", None)
        if kwargs.get("thinking_budget", None) is not None:
            extra_body["chat_template_kwargs"]["thinking_budget"] = kwargs.get("thinking_budget", None)
        if extra_body["chat_template_kwargs"]:
            chat_completion = client.chat.completions.create(
                extra_body=extra_body,**data
            )
        else:
            chat_completion = client.chat.completions.create(**data
            )
        return chat_completion.dict()

    @staticmethod
    def get_streaming_think_response(data,
                               service_url,
                               timeout=100,
                               **kwargs):
        tmp_url = service_url.split('v1/')[0] + "v1/"
        client = OpenAI(base_url=tmp_url, api_key='na')
        # model="qwen1half-14b-chat"
        model = data["model"]
        messages = data["messages"]
        seed = data["seed"]
        max_tokens = data["max_tokens"]
        top_p = data["top_p"]
        temperature = data["temperature"]
        # messages=[
        #     {
        #         "role":"user",
        #         "content": "1+1等于几",
        #         # "content":"你是谁？"
        #     }
        # ]
        chat_completion = client.chat.completions.create(
            model=model,
            messages=messages,
            top_p=top_p,         # 设置 top_p 参数
            temperature=temperature,   # 设置 temperature 参数
            # max_tokens=max_tokens,
            seed=seed,
            stream=True
        )
        # todo 诡异问题：使用透传参数无法复现思维链
        # chat_completion = client.chat.completions.create(
        #     **data
        # )
        for chunk in chat_completion:
            item = chunk.dict()
            yield item

    @staticmethod
    def get_streaming_response(data,
                               service_url,
                               timeout=100,
                               **kwargs):
        client = OpenAI(base_url=service_url.split('v1/')[0] + "v1/", api_key='na')
        chat_completion = client.chat.completions.create(
            **data
        )
        for chunk in chat_completion:
            item = chunk.dict()
            yield item


    @classmethod
    def from_model_type(cls,model_type:str) -> LLM:
        import os
        yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),f"model_type_config/{model_type}.yaml")
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        LLM_DEFAULT_MODEL_API_KWARGS = config["model_api_kwargs"]
        mode_service_ip = config["mode_service_ip"]
        mode_service_port = config["mode_service_port"]
        model_name = config["model_name"]
        service_url = f"http://{mode_service_ip}:{mode_service_port}/v1/chat/completions"
        return cls(model_name=model_name,
                     service_url=service_url,
                     temperature = LLM_DEFAULT_MODEL_API_KWARGS.pop("temperature",0.8),
                     seed=LLM_DEFAULT_MODEL_API_KWARGS.pop("seed", 111),
                     model_kwargs=LLM_DEFAULT_MODEL_API_KWARGS,
                     streaming=False,
                     api_key = config.get("model_api_key",None),
                     enable_thinking =config.get("enable_thinking",None),
                     thinking_budget = config.get("thinking_budget",-1),
                     callbacks=[StdOutCallbackHandler(color="green")])





if __name__ == "__main__":

    llm = ChatClient.from_model_type('qwen2.5-14B')
    answer = llm.abatch(["1+2+3+..+100等于多少?","2的8次方是多少"])
    print(answer)


