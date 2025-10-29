- LangChain 是一套面向大模型的开发框架（SDK）
- LangChain 是 AGI 时代软件工程的一个探索和原型
- 学习 LangChain 要关注接口变更

高度抽象封装，扩展性很强，灵活

# 1 LangChain 的核心组件
1. **模型 I/O 封装**
    - Chat Models：对语言模型接口的封装
    - PromptTemple：提示词模板
    - OutputParser：解析输出
        
2. 数据连接封装（弱于 LlamaIndex）
    - Document Loaders：各种格式文件的加载器
    - Document Transformers：对文档的常用操作，如：split, filter, translate, extract metadata, etc
    - Text Embedding Models：文本向量化表示，用于检索等操作
    - Verctorstores & Retrievers：向量数据库与向量检索
        
3. **架构封装**
    - Chain/LCEL：实现一个功能或者一系列顺序功能组合
    - Agent：根据用户输入，自动规划执行步骤，自动选择每步需要的工具，最终完成用户指定的功能
        - Tools：调用外部功能的函数，例如：调 google 搜索、文件 I/O、Linux Shell 等等
    - LangGraph：工作流开发框架
        
4. LangSmith：过程监控与调试框架
![image.png](https://learning-1316972768.cos.ap-nanjing.myqcloud.com/%E5%90%8E%E6%9C%9F/202408/20251029213236.png)
文档（以 Python 版为例）
- 功能模块：[https://python.langchain.com/docs/tutorials](https://python.langchain.com/docs/tutorials)
- API 文档：[https://python.langchain.com/api_reference/](https://python.langchain.com/api_reference/)
- 三方组件集成：[https://python.langchain.com/docs/integrations/providers/](https://python.langchain.com/docs/integrations/providers/)
- 更多 HowTo：[https://python.langchain.com/docs/how_to/](https://python.langchain.com/docs/how_to/)
 
- LangChain 是开源项目
  项目地址：[https://github.com/langchain-ai](https://github.com/langchain-ai)


# 2 模型 I/O 封装
把不同的模型，统一封装成一个接口，方便更换模型而不用重构代码
## 2.1 模型 API: ChatModel

### 2.1.1 OpenAI 模型封装
```python
# !pip install -U langchain  
# !pip install -U langchain-openai
from langchain.chat_models import init_chat_model  
model = init_chat_model("gpt-4o-mini", model_provider="openai")  
response = model.invoke("你是谁")  
print(response.content)
```

我是一个AI助手，旨在回答问题和提供信息。如果你有什么想问的，或者需要帮助的，请告诉我！

### 2.1.2 多轮对话 Session 封装
```python
from langchain.schema import (  
    AIMessage,  # 等价于OpenAI接口中的assistant role  
    HumanMessage,  # 等价于OpenAI接口中的user role  
    SystemMessage  # 等价于OpenAI接口中的system role  
)  
​  
messages = [  
    SystemMessage(content="你是聚客AI大模型课的课程助理。"),  
    HumanMessage(content="我是学员，我叫小聚。"),  
    AIMessage(content="欢迎！"),  
    HumanMessage(content="我是谁？")  
]  
​  
ret = model.invoke(messages)  
​  
print(ret.content)
```

你是学员小聚。很高兴与你交流！有什么问题或者需要帮助的地方吗？


**划重点：通过模型封装，实现不同模型的统一接口调用

### 2.1.3 换个国产模型
​ 
```python
# !pip install -U langchain-deepseek
from langchain.chat_models import init_chat_model

model = init_chat_model(model="deepseek-chat", model_provider="deepseek")

response = model.invoke("你是谁")
print(response.content)
```
### 2.1.4 流式输出
```python
for token in model.stream("你是谁"):
	print(token.content, end="")
```


## 2.2 模型的输入与输出
![image.png](https://learning-1316972768.cos.ap-nanjing.myqcloud.com/%E5%90%8E%E6%9C%9F/202408/20251029213820.png)
### 2.2.1 Prompt 模板封装
1. PromptTemplate 可以在模板中自定义变量
	```python
	from langchain.prompts import PromptTemplate
	
	template = PromptTemplate.from_template("给我讲个关于{subject}的笑话")
	print("===Template===")
	print(template)
	print("===Prompt===")
	print(template.format(subject='小明'))
	```

	```output
	===Template===
	input_variables=['subject'] input_types={} partial_variables={} template='给我讲个关于{subject}的笑话'
	===Prompt===
	给我讲个关于小明的笑话
	```

	```python
	from langchain.chat_models import init_chat_model
	
	# 定义 LLM
	llm = init_chat_model("deepseek-chat", model_provider="deepseek")
	# 通过 Prompt 调用 LLM
	ret = llm.invoke(template.format(subject='小明'))
	# 打印输出
	print(ret.content)
	```

	```output
	好的！这里有一个关于小明的经典笑话：
	
	---
	
	**老师**：小明，用“果然”造句。  
	**小明**：我先吃苹果，然后喝酸奶。  
	**老师**：……这不对，“果然”要连在一起用！  
	**小明**（淡定）：没错啊，我先吃“苹”，果然喝“酸奶”！  
	
	（谐音梗：苹果的“果”+“然”=果然）  
	
	---
	
	希望这个笑话能让你会心一笑！ 😄
	```
2.  ChatPromptTemplate 用模板表示的对话上下文
	````python
	from langchain.prompts import (
	    ChatPromptTemplate,
	    HumanMessagePromptTemplate,
	    SystemMessagePromptTemplate,
	)
	from langchain.chat_models import init_chat_model
	llm = init_chat_model("gpt-4o-mini", model_provider="openai")
	
	template = ChatPromptTemplate.from_messages(
	    [
	        SystemMessagePromptTemplate.from_template("你是{product}的客服助手。你的名字叫{name}"),
	        HumanMessagePromptTemplate.from_template("{query}")
	    ]
	)
	
	prompt = template.format_messages(
	    product="聚客AI大模型课程",
	    name="小聚",
	    query="你是谁"
	)
	
	print(prompt)
	
	ret = llm.invoke(prompt)
	
	print(ret.content)
	````

	```output
	[SystemMessage(content='你是聚客AI大模型课程的客服助手。你的名字叫小聚', additional_kwargs={}, response_metadata={}), HumanMessage(content='你是谁', additional_kwargs={}, response_metadata={})]
	我是小聚，聚客AI大模型课程的客服助手。如果你有任何问题或者需要帮助，随时可以问我！
	```

3. MessagesPlaceholder 把多轮对话变成模板
	```python
	from langchain.prompts import (
	    ChatPromptTemplate,
	    HumanMessagePromptTemplate,
	    MessagesPlaceholder,
	)
	
	human_prompt = "Translate your answer to {language}."
	human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)
	
	chat_prompt = ChatPromptTemplate.from_messages(
	    # variable_name 是 message placeholder 在模板中的变量名
	    # 用于在赋值时使用
	    [MessagesPlaceholder("history"), human_message_template]
	)
	```

	```python
	from langchain_core.messages import AIMessage, HumanMessage
	
	human_message = HumanMessage(content="Who is Elon Musk?")
	ai_message = AIMessage(
	    content="Elon Musk is a billionaire entrepreneur, inventor, and industrial designer"
	)
	
	messages = chat_prompt.format_prompt(
	    # 对 "history" 和 "language" 赋值
	    history=[human_message, ai_message], language="中文"
	)
	
	print(messages.to_messages())
	```

	```output
	[HumanMessage(content='Who is Elon Musk?', additional_kwargs={}, response_metadata={}), AIMessage(content='Elon Musk is a billionaire entrepreneur, inventor, and industrial designer', additional_kwargs={}, response_metadata={}), HumanMessage(content='Translate your answer to 中文.', additional_kwargs={}, response_metadata={})]
	```

	```python
	result = llm.invoke(messages)
	print(result.content)
	```

	```output
	埃隆·马斯克是一位亿万富翁企业家、发明家和工业设计师。
	```


**划重点：把Prompt模板看作带有参数的函数**

### 2.2.2 从文件加载 Prompt 模板
```python
from langchain.prompts import PromptTemplate

template = PromptTemplate.from_file("example_prompt_template.txt")
print("===Template===")
print(template)
print("===Prompt===")
print(template.format(topic='黑色幽默'))
```

```output
===Template===
input_variables=['topic'] input_types={} partial_variables={} template='举一个关于{topic}的例子'
===Prompt===
举一个关于黑色幽默的例子
```

### 2.2.3 结构化输出
#### 2.2.3.1 直接输出 Pydantic 对象
```python
from pydantic import BaseModel, Field

# 定义你的输出对象
class Date(BaseModel):
    year: int = Field(description="Year")
    month: int = Field(description="Month")
    day: int = Field(description="Day")
    era: str = Field(description="BC or AD")
```

```python
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from langchain.chat_models import init_chat_model
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# 定义结构化输出的模型
structured_llm = llm.with_structured_output(Date)

template = """提取用户输入中的日期。
用户输入:
{query}"""

prompt = PromptTemplate(
    template=template,
)

query = "2023年四月6日天气晴..."
input_prompt = prompt.format_prompt(query=query)

structured_llm.invoke(input_prompt)
```

```output
Date(year=2023, month=4, day=6, era='AD')
```

#### 2.2.3.2 输出指定格式的 JSON
```python
# OpenAI 模型的JSON格式
json_schema = {
    "title": "Date",
    "description": "Formated date expression",
    "type": "object",
    "properties": {
        "year": {
            "type": "integer",
            "description": "year, YYYY",
        },
        "month": {
            "type": "integer",
            "description": "month, MM",
        },
        "day": {
            "type": "integer",
            "description": "day, DD",
        },
        "era": {
            "type": "string",
            "description": "BC or AD",
        },
    },
}
structured_llm = llm.with_structured_output(json_schema)

structured_llm.invoke(input_prompt)
```

```output
{'year': 2023, 'month': 4, 'day': 6, 'era': 'AD'}
```

#### 2.2.3.3 使用 OutputParser
[`OutputParser`](https://python.langchain.com/v0.2/docs/concepts/#output-parsers) 可以按指定格式解析模型的输出
```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser(pydantic_object=Date)

prompt = PromptTemplate(
    template="提取用户输入中的日期。\n用户输入:{query}\n{format_instructions}",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

input_prompt = prompt.format_prompt(query=query)
output = llm.invoke(input_prompt)
print("原始输出:\n"+output.content)

print("\n解析后:")
parser.invoke(output)
```

```output
原始输出:
{"year": 2023, "month": 4, "day": 6, "era": "AD"}
解析后:
{'year': 2023, 'month': 4, 'day': 6, 'era': 'AD'}
```
也可以用 `PydanticOutputParser`
```python
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=Date)

input_prompt = prompt.format_prompt(query=query)
output = llm.invoke(input_prompt)
print("原始输出:\n"+output.content)

print("\n解析后:")
parser.invoke(output)
```

```output
原始输出:
{"year": 2023, "month": 4, "day": 6, "era": "AD"}
解析后:
Date(year=2023, month=4, day=6, era='AD')
```


`OutputFixingParser` 利用大模型做格式自动纠错
```python
from langchain.output_parsers import OutputFixingParser
from langchain.chat_models import init_chat_model

llm = init_chat_model(model="deepseek-chat", model_provider="deepseek")

# 纠错能力与大模型能力相关
new_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

bad_output = output.content.replace("4","四")
print("PydanticOutputParser:")
try:
    parser.invoke(bad_output)
except Exception as e:
    print(e)

print("OutputFixingParser:")
new_parser.invoke(bad_output)
```

```output
PydanticOutputParser:
Invalid json output: ```json
{"year": 2023, "month": 四, "day": 6, "era": "AD"}
For troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/OUTPUT_PARSING_FAILURE 
OutputFixingParser:
Date(year=2023, month=4, day=6, era='AD')
```

## 2.3  Function Calling
```python
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a + b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a * b
```

```python
import json

llm_with_tools = llm.bind_tools([add, multiply])

query = "3.5的4倍是多少?"
messages = [HumanMessage(query)]

output = llm_with_tools.invoke(messages)

print(json.dumps(output.tool_calls, indent=4))
```

```python
[
    {
        "name": "multiply",
        "args": {
            "a": 3.5,
            "b": 4
        },
        "id": "call_0_5d27b732-349e-4803-8fac-27f52cf5a263",
        "type": "tool_call"
    }
]
```

回传 Funtion Call 的结果
```python
messages.append(output)

available_tools = {"add": add, "multiply": multiply}

for tool_call in output.tool_calls:
    selected_tool = available_tools[tool_call["name"].lower()]
    tool_msg = selected_tool.invoke(tool_call)
    messages.append(tool_msg)

new_output = llm_with_tools.invoke(messages)
for message in messages:
    print(json.dumps(message.model_dump(), indent=4, ensure_ascii=False))
print(new_output.content)
```

```output
{
    "content": "3.5的4倍是多少?",
    "additional_kwargs": {},
    "response_metadata": {},
    "type": "human",
    "name": null,
    "id": null,
    "example": false
}
{
    "content": "",
    "additional_kwargs": {
        "tool_calls": [
            {
                "id": "call_0_5d27b732-349e-4803-8fac-27f52cf5a263",
                "function": {
                    "arguments": "{\"a\":3.5,\"b\":4}",
                    "name": "multiply"
                },
                "type": "function",
                "index": 0
            }
        ],
        "refusal": null
    },
    "response_metadata": {
        "token_usage": {
            "completion_tokens": 27,
            "prompt_tokens": 250,
            "total_tokens": 277,
            "completion_tokens_details": null,
            "prompt_tokens_details": {
                "audio_tokens": null,
                "cached_tokens": 192
            },
            "prompt_cache_hit_tokens": 192,
            "prompt_cache_miss_tokens": 58
        },
        "model_name": "deepseek-chat",
        "system_fingerprint": "fp_8802369eaa_prod0425fp8",
        "id": "31dd676f-2bd5-4baf-a1bd-a79052128ed6",
        "service_tier": null,
        "finish_reason": "tool_calls",
        "logprobs": null
    },
    "type": "ai",
    "name": null,
    "id": "run--74e25843-2fdc-4de7-beda-605ab0ea4883-0",
    "example": false,
    "tool_calls": [
        {
            "name": "multiply",
            "args": {
                "a": 3.5,
                "b": 4
            },
            "id": "call_0_5d27b732-349e-4803-8fac-27f52cf5a263",
            "type": "tool_call"
        }
    ],
    "invalid_tool_calls": [],
    "usage_metadata": {
        "input_tokens": 250,
        "output_tokens": 27,
        "total_tokens": 277,
        "input_token_details": {
            "cache_read": 192
        },
        "output_token_details": {}
    }
}
{
    "content": "14.0",
    "additional_kwargs": {},
    "response_metadata": {},
    "type": "tool",
    "name": "multiply",
    "id": null,
    "tool_call_id": "call_0_5d27b732-349e-4803-8fac-27f52cf5a263",
    "artifact": null,
    "status": "success"
}
3.5的4倍是14.0。
```

## 2.4 小结

1. LangChain 统一封装了各种模型的调用接口，包括补全型和对话型两种
2. LangChain 提供了 PromptTemplate 类，可以自定义带变量的模板
3. LangChain 提供了一些列输出解析器，用于将大模型的输出解析成结构化对象
4. LangChain 提供了 Function Calling 的封装
5. 上述模型属于 LangChain 中较为实用的部分

# 3  Chain 和 LangChain Expression Language (LCEL)
LangChain Expression Language（LCEL）是一种声明式语言，可轻松组合不同的调用顺序构成 Chain。LCEL 自创立之初就被设计为能够支持将原型投入生产环境，**无需代码更改**，从最简单的“提示+LLM”链到最复杂的链（已有用户成功在生产环境中运行包含数百个步骤的 LCEL Chain）。

LCEL 的一些亮点包括：

1. **流支持**：使用 LCEL 构建 Chain 时，你可以获得最佳的首个令牌时间（即从输出开始到首批输出生成的时间）。对于某些 Chain，这意味着可以直接从 LLM 流式传输令牌到流输出解析器，从而以与 LLM 提供商输出原始令牌相同的速率获得解析后的、增量的输出。
    
2. **异步支持**：任何使用 LCEL 构建的链条都可以通过同步 API（例如，在 Jupyter 笔记本中进行原型设计时）和异步 API（例如，在 LangServe 服务器中）调用。这使得相同的代码可用于原型设计和生产环境，具有出色的性能，并能够在同一服务器中处理多个并发请求。
    
3. **优化的并行执行**：当你的 LCEL 链条有可以并行执行的步骤时（例如，从多个检索器中获取文档），我们会自动执行，无论是在同步还是异步接口中，以实现最小的延迟。
    
4. **重试和回退**：为 LCEL 链的任何部分配置重试和回退。这是使链在规模上更可靠的绝佳方式。目前我们正在添加重试/回退的流媒体支持，因此你可以在不增加任何延迟成本的情况下获得增加的可靠性。
    
5. **访问中间结果**：对于更复杂的链条，访问在最终输出产生之前的中间步骤的结果通常非常有用。这可以用于让最终用户知道正在发生一些事情，甚至仅用于调试链条。你可以流式传输中间结果，并且在每个 LangServe 服务器上都可用。
    
6. **输入和输出模式**：输入和输出模式为每个 LCEL 链提供了从链的结构推断出的 Pydantic 和 JSONSchema 模式。这可以用于输入和输出的验证，是 LangServe 的一个组成部分。
    
7. **无缝 LangSmith 跟踪集成**：随着链条变得越来越复杂，理解每一步发生了什么变得越来越重要。通过 LCEL，所有步骤都自动记录到 LangSmith，以实现最大的可观察性和可调试性。
    
8. **无缝 LangServe 部署集成**：任何使用 LCEL 创建的链都可以轻松地使用 LangServe 进行部署。

原文：[https://python.langchain.com/docs/expression_language/](https://python.langchain.com/docs/expression_language/)


## 3.1 Pipeline 式调用 PromptTemplate, LLM 和 OutputParser
```python
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum
import json
from langchain.chat_models import init_chat_model
```

```python
# 输出结构
class SortEnum(str, Enum):
    data = 'data'
    price = 'price'


class OrderingEnum(str, Enum):
    ascend = 'ascend'
    descend = 'descend'


class Semantics(BaseModel):
    name: Optional[str] = Field(description="流量包名称", default=None)
    price_lower: Optional[int] = Field(description="价格下限", default=None)
    price_upper: Optional[int] = Field(description="价格上限", default=None)
    data_lower: Optional[int] = Field(description="流量下限", default=None)
    data_upper: Optional[int] = Field(description="流量上限", default=None)
    sort_by: Optional[SortEnum] = Field(description="按价格或流量排序", default=None)
    ordering: Optional[OrderingEnum] = Field(
description="升序或降序排列", default=None)


# Prompt 模板
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个语义解析器。你的任务是将用户的输入解析成JSON表示。不要回答用户的问题。"),
        ("human", "{text}"),
    ]
)

# 模型
llm = init_chat_model("deepseek-chat", model_provider="deepseek")

structured_llm = llm.with_structured_output(Semantics)

# LCEL 表达式
runnable = (
    {"text": RunnablePassthrough()} | prompt | structured_llm
)

# 直接运行
ret = runnable.invoke("不超过100元的流量大的套餐有哪些")
print(
    json.dumps(
        ret.model_dump(),
        indent = 4,
        ensure_ascii=False
    )
)
```

```output
{
    "name": null,
    "price_lower": null,
    "price_upper": 100,
    "data_lower": null,
    "data_upper": null,
    "sort_by": "data",
    "ordering": "descend"
}
```

使用 LCEL 的价值，也就是 LangChain 的核心价值。</b> <br />
官方从不同角度给出了举例说明：https://python.langchain.com/docs/concepts/lcel/


## 3.2 用 LCEL 实现 RAG
```python
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings.dashscope import DashScopeEmbeddings

# 加载文档
loader = PyMuPDFLoader("./data/deepseek-v3-1-4.pdf")
pages = loader.load_and_split()

# 文档切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=200,
    length_function=len,
    add_start_index=True,
)

texts = text_splitter.create_documents(
    [page.page_content for page in pages[:1]]
)

# 灌库
embeddings = DashScopeEmbeddings(
    model="text-embedding-v1", dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)
db = FAISS.from_documents(texts, embeddings)

# 检索 top-5 结果
retriever = db.as_retriever(search_kwargs={"k": 5})
```

```python
docs = retriever.invoke("deepseek v3有多少参数")

for doc in docs:
    print(doc.page_content)
    print("----")
```

```python
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# Prompt模板
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Chain
rag_chain = (
    {"question": RunnablePassthrough(), "context": retriever}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("deepseek V3有多少参数")
```


## 3.3 用 LCEL 实现模型切换（工厂模式）
```python
from langchain_core.runnables.utils import ConfigurableField
from langchain_community.chat_models import QianfanChatEndpoint
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage
import os

# 模型1
ds_model = init_chat_model("deepseek-chat", model_provider="deepseek")

# 模型2
gpt_model = init_chat_model("gpt-4o-mini", model_provider="openai")


# 通过 configurable_alternatives 按指定字段选择模型
model = gpt_model.configurable_alternatives(
    ConfigurableField(id="llm"), 
    default_key="gpt", 
    deepseek=ds_model,
    # claude=claude_model,
)

# Prompt 模板
prompt = ChatPromptTemplate.from_messages(
    [
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)

# LCEL
chain = (
    {"query": RunnablePassthrough()} 
    | prompt
    | model 
    | StrOutputParser()
)

# 运行时指定模型 "gpt" or "deepseek"
ret = chain.with_config(configurable={"llm": "gpt"}).invoke("请自我介绍")

print(ret)
```


扩展阅读：什么是[**工厂模式**](https://www.runoob.com/design-pattern/factory-pattern.html)；[**设计模式**](https://www.runoob.com/design-pattern/design-pattern-intro.html)概览。


**思考：**从模块间解依赖角度，LCEL的意义是什么？

​