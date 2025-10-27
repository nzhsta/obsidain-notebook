## 1. 大语言模型开发框架的价值是什么？

_SDK：Software Development Kit，它是一组软件工具和资源的集合，旨在帮助开发者创建、测试、部署和维护应用程序或软件。_

所有开发框架（SDK）的核心价值，都是降低开发、维护成本。
大语言模型开发框架的价值，是让开发者可以更方便地开发基于大语言模型的应用。主要提供两类帮助：
1. 第三方能力抽象。比如 LLM、向量数据库、搜索接口等
2. 常用工具、方案封装
3. 底层实现封装。比如流式接口、超时重连、异步与并行等
    

好的开发框架，需要具备以下特点：
1. 可靠性、鲁棒性高
2. 可维护性高
3. 可扩展性高
4. 学习成本低
    

举些通俗的例子：
- 与外部功能解依赖
    - 比如可以随意更换 LLM 而不用大量重构代码
    - 更换三方工具也同理
        
- 经常变的部分要在外部维护而不是放在代码里
    - 比如 Prompt 模板
        
- 各种环境下都适用
    - 比如线程安全
        
- 方便调试和测试
    - 至少要能感觉到用了比不用方便吧
    - 合法的输入不会引发框架内部的报错

<div class="alert alert-success">
<b>划重点：</b>选对了框架，事半功倍；反之，事倍功半。
</div>


**什么是 SDK?** [https://aws.amazon.com/cn/what-is/sdk/](https://aws.amazon.com/cn/what-is/sdk/) 
**SDK 和 API 的区别是什么?** [https://aws.amazon.com/cn/compare/the-difference-between-sdk-and-api/](https://aws.amazon.com/cn/compare/the-difference-between-sdk-and-api/) 

 举个例子：使用 SDK，4 行代码实现一个简易的 RAG 系统：
 LlamaIndex 默认的 Embedding 模型是 `OpenAIEmbedding(model="text-embedding-ada-002")`
```python
# !pip install --upgrade llama-index

# !pip install llama-index-llms-dashscope
# !pip install llama-index-llms-openai-like
# !pip install llama-index-embeddings-dashscope
```

```python
import os
from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels

# LlamaIndex默认使用的大模型被替换为百炼
# Settings.llm = OpenAILike(
#     model="qwen-max",
#     api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     is_chat_model=True
# )

Settings.llm = DashScope(model_name=DashScopeGenerationModels.QWEN_MAX, api_key=os.getenv("DASHSCOPE_API_KEY"))

# LlamaIndex默认使用的Embedding模型被替换为百炼的Embedding模型
Settings.embed_model = DashScopeEmbedding(
    # model_name="text-embedding-v1"
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V1,
    # api_key=os.getenv("DASHSCOPE_API_KEY")
)
```

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("deepseek v3有多少参数？")

# print(response)
```


# 2. LlamaIndex 介绍

官网标题：_「 Build AI Knowledge Assistants over your enterprise data 」_

LlamaIndex 是一个为开发「**知识增强**」的大语言模型应用的框架（也就是 SDK）。**知识增强**，泛指任何在私有或特定领域数据基础上应用大语言模型的情况。例如：
![basic_rag.png](https://learning-1316972768.cos.ap-nanjing.myqcloud.com/%E5%90%8E%E6%9C%9F/202408/basic_rag.png)
- Question-Answering Chatbots (也就是 RAG)
- Document Understanding and Extraction （文档理解与信息抽取）
- Autonomous Agents that can perform research and take actions （智能体应用）
- Workflow orchestrating single and multi-agent (编排单个或多个智能体形成工作流）
LlamaIndex 有 Python 和 Typescript 两个版本，Python 版的文档相对更完善。
- Python 文档地址：[https://docs.llamaindex.ai/en/stable/](https://docs.llamaindex.ai/en/stable/)
- Python API 接口文档：[https://docs.llamaindex.ai/en/stable/api_reference/](https://docs.llamaindex.ai/en/stable/api_reference/
- TS 文档地址：[https://ts.llamaindex.ai/](https://ts.llamaindex.ai/)
    
LlamaIndex 是一个开源框架，Github 链接：[https://github.com/run-llama](https://github.com/run-llama)

#  3  LlamaIndex 的核心模块
![](https://learning-1316972768.cos.ap-nanjing.myqcloud.com/%E5%90%8E%E6%9C%9F/202408/llamaindex.png)

```python
# ！pip install llama-index
```

