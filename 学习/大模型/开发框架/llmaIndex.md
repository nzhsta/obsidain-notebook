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

# 3 数据加载（loading）
## 3.1 加载本地数据
`SimpleDirectoryReader` 是一个简单的本地文件加载器。它会遍历指定目录，并根据文件扩展名自动加载文件（**文本内容**）。

支持的文件类型：
- `.csv` - comma-separated values
- `.docx` - Microsoft Word
- `.epub` - EPUB ebook format
- `.hwp` - Hangul Word Processor
- `.ipynb` - Jupyter Notebook
- `.jpeg`, `.jpg` - JPEG image
- `.mbox` - MBOX email archive
- `.md` - Markdown
- `.mp3`, `.mp4` - audio and video
- `.pdf` - Portable Document Format
- `.png` - Portable Network Graphics
- `.ppt`, `.pptm`, `.pptx` - Microsoft PowerPoint

```python
import json
from pydantic.v1 import BaseModel

def show_json(data):
    """用于展示json数据"""
    if isinstance(data, str):
        obj = json.loads(data)
        print(json.dumps(obj, indent=4, ensure_ascii=False))
    elif isinstance(data, dict) or isinstance(data, list):
        print(json.dumps(data, indent=4, ensure_ascii=False))
    elif issubclass(type(data), BaseModel):
        print(json.dumps(data.dict(), indent=4, ensure_ascii=False))

def show_list_obj(data):
    """用于展示一组对象"""
    if isinstance(data, list):
        for item in data:
            show_json(item)
    else:
        raise ValueError("Input is not a list")
```

```python
from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(
        input_dir="./data", # 目标目录
        recursive=False, # 是否递归遍历子目录
        required_exts=[".pdf"] # (可选)只读取指定后缀的文件
    )
documents = reader.load_data()
print(documents[0].text)
show_json(documents[0].json())
```

**注意：对图像、视频、语音类文件，默认不会自动提取其中文字。如需提取，参考下面介绍的 `Data Connectors`。

默认的 `PDFReader` 效果并不理想，我们可以更换文件加载器
- **LlamaParse**
  首先，登录并从 [https://cloud.llamaindex.ai](https://cloud.llamaindex.ai) ↗ 注册并获取 api-key 。然后，安装该包：
	```python
	# !pip install llama-cloud-services
	# 在系统环境变量里配置 LLAMA_CLOUD_API_KEY=XXX
	
	from llama_cloud_services import LlamaParse
	from llama_index.core import SimpleDirectoryReader
	import nest_asyncio
	nest_asyncio.apply() # 只在Jupyter笔记环境中需要此操作，否则会报错
	
	# set up parser
	parser = LlamaParse(
	    result_type="markdown"  # "markdown" and "text" are available
	)
	file_extractor = {".pdf": parser}
	
	documents = SimpleDirectoryReader(input_dir="./data", required_exts=[".pdf"], file_extractor=file_extractor).load_data()
	print(documents[0].text)
	```

## 3.2 Data Connectors
用于处理更丰富的数据类型，并将其读取为 `Document` 的形式。

例如：直接读取网页
```python
# !pip install llama-index-readers-web
from llama_index.readers.web import SimpleWebPageReader

documents = SimpleWebPageReader(html_to_text=True).load_data(
    ["https://edu.guangjuke.com/tx/"]
)

print(documents[0].text)
```

**更多 Data Connectors**
- 内置的[文件加载器](https://llamahub.ai/l/readers/llama-index-readers-file)
- 连接三方服务的[数据加载器](https://docs.llamaindex.ai/en/stable/module_guides/loading/connector/modules/)，例如数据库
- 更多加载器可以在 [LlamaHub](https://llamahub.ai/) 上找到

# 4 文本切分与解析（chunking）
为方便检索，我们通常把 `Document` 切分为 `Node`。
在 LlamaIndex 中，`Node` 被定义为一个文本的「chunk」。
## 4.1 使用 TextSplitters 对文本做切分
例如：`TokenTextSplitter` 按指定 token 数切分文本
```python
from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter

node_parser = TokenTextSplitter(
    chunk_size=512,  # 每个 chunk 的最大长度
    chunk_overlap=200  # chunk 之间重叠长度
)

nodes = node_parser.get_nodes_from_documents(
    documents, show_progress=False
)
show_json(nodes[1].json())
show_json(nodes[2].json())
```

LlamaIndex 提供了丰富的 `TextSplitter`，例如：
- [`SentenceSplitter`](https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/sentence_splitter/)：在切分指定长度的 chunk 同时尽量保证**句子边界**不被切断；
- [`CodeSplitter`](https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/code/)：根据 AST（编译器的抽象句法树）**切分代码**，保证代码功能片段完整；
- [`SemanticSplitterNodeParser`](https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/semantic_splitter/)：根据**语义相关性**对将文本切分为片段——根据 transformer 模型。

## 4.2 使用 NodeParsers 对有结构的文档做解析
例如：`HTMLNodeParser` 解析 HTML 文档
```python
from llama_index.core.node_parser import HTMLNodeParser
from llama_index.readers.web import SimpleWebPageReader

documents = SimpleWebPageReader(html_to_text=False).load_data(
    ["https://edu.guangjuke.com/tx/"]
)

# 默认解析 ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "b", "i", "u", "section"]
parser = HTMLNodeParser(tags=["span"])  # 可以自定义解析哪些标签
nodes = parser.get_nodes_from_documents(documents)

for node in nodes:
    print(node.text+"\n")
```

更多的 `NodeParser` 包括 [`MarkdownNodeParser`](https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/markdown/)，[`JSONNodeParser`](https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/json/) 等等。

# 5. 索引（Indexing）与检索（Retrieval）
**基础概念**：在「检索」相关的上下文中，「索引」即 `index`，通常是指为了实现快速检索而设计的特定「数据结构」。
索引的具体原理与实现不是本课程的教学重点，感兴趣的同学可以参考：[传统索引](https://en.wikipedia.org/wiki/Search_engine_indexing)、[向量索引](https://medium.com/kx-systems/vector-indexing-a-roadmap-for-vector-databases-65866f07daf5)
## 5.1 向量检索
1. `VectorStoreIndex` 直接在内存中构建一个 Vector Store 并建索引
	```python
	from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
	from llama_index.core.node_parser import TokenTextSplitter, SentenceSplitter
	
	# 加载 pdf 文档
	documents = SimpleDirectoryReader(
	    "./data", 
	    required_exts=[".pdf"],
	).load_data()
	
	# 定义 Node Parser
	node_parser = TokenTextSplitter(chunk_size=512, chunk_overlap=200)
	
	# 切分文档
	nodes = node_parser.get_nodes_from_documents(documents)
	
	# 构建 index，默认是在内存中
	index = VectorStoreIndex(nodes)
	
	# 另外一种实现方式
	# index = VectorStoreIndex.from_documents(documents=documents, transformations=[SentenceSplitter(chunk_size=512)])
	
	# 写入本地文件
	# index.storage_context.persist(persist_dir="./doc_emb")
	
	# 获取 retriever
	vector_retriever = index.as_retriever(
	    similarity_top_k=2 # 返回2个结果
	)
	
	# 检索
	results = vector_retriever.retrieve("deepseek v3数学能力怎么样？")
	
	print(results[0].text)
	```
2. 使用自定义的 Vector Store，以 `Qdrant` 为例：
```python
# !pip install llama-index-vector-stores-qdrant
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(location=":memory:")
collection_name = "demo"
collection = client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
# storage: 指定存储空间
storage_context = StorageContext.from_defaults(vector_store=vector_store)


# 创建 index：通过 Storage Context 关联到自定义的 Vector Store
index = VectorStoreIndex(nodes, storage_context=storage_context)

# 获取 retriever
vector_retriever = index.as_retriever(similarity_top_k=1)

# 检索
results = vector_retriever.retrieve("deepseek v3数学能力怎么样")

print(results[0])
```
## 5.2 更多索引与检索方式

LlamaIndex 内置了丰富的检索机制，例如：

- 关键字检索
    - [`BM25Retriever`](https://docs.llamaindex.ai/en/stable/api_reference/retrievers/bm25/)：基于 tokenizer 实现的 BM25 经典检索算法
    - [`KeywordTableGPTRetriever`](https://docs.llamaindex.ai/en/stable/api_reference/retrievers/keyword/#llama_index.core.indices.keyword_table.retrievers.KeywordTableGPTRetriever)：使用 GPT 提取检索关键字
    - [`KeywordTableSimpleRetriever`](https://docs.llamaindex.ai/en/stable/api_reference/retrievers/keyword/#llama_index.core.indices.keyword_table.retrievers.KeywordTableSimpleRetriever)：使用正则表达式提取检索关键字
    - [`KeywordTableRAKERetriever`](https://docs.llamaindex.ai/en/stable/api_reference/retrievers/keyword/#llama_index.core.indices.keyword_table.retrievers.KeywordTableRAKERetriever)：使用[`RAKE`](https://pypi.org/project/rake-nltk/)算法提取检索关键字（有语言限制）
        
- RAG-Fusion [`QueryFusionRetriever`](https://docs.llamaindex.ai/en/stable/api_reference/retrievers/query_fusion/)
- 还支持 [KnowledgeGraph](https://docs.llamaindex.ai/en/stable/api_reference/retrievers/knowledge_graph/)、[SQL](https://docs.llamaindex.ai/en/stable/api_reference/retrievers/sql/#llama_index.core.retrievers.SQLRetriever)、[Text-to-SQL](https://docs.llamaindex.ai/en/stable/api_reference/retrievers/sql/#llama_index.core.retrievers.NLSQLRetriever) 等等

## 5.3 索引后处理
LlamaIndex 的 `Node Postprocessors` 提供了一系列检索后处理模块。

例如：我们可以用不同模型对检索后的 `Nodes` 做重排序
```python
# 获取 retriever
vector_retriever = index.as_retriever(similarity_top_k=5)

# 检索
nodes = vector_retriever.retrieve("deepseek v3有多少参数?")

for i, node in enumerate(nodes):
    print(f"[{i}] {node.text}\n")
```

```python
from llama_index.core.postprocessor import LLMRerank

postprocessor = LLMRerank(top_n=2) # 走一遍transformer，比较耗时
nodes = postprocessor.postprocess_nodes(nodes, query_str="deepseek v3有多少参数?")
for i, node in enumerate(nodes):
    print(f"[{i}] {node.text}")
```

更多的 Rerank 及其它后处理方法，参考官方文档：[Node Postprocessor Modules](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/)

# 6 生成回复（QA & Chat）
## 6.1 单轮问答（Query Engine）

```python
qa_engine = index.as_query_engine()  
response = qa_engine.query("deepseek v3数学能力怎么样?")  
print(response)
```

```output
DeepSeek-V3在数学相关基准测试中表现出色，特别是在非长链思维开放源码和闭源模型中处于领先地位。它在某些特定的基准测试如MATH-500上甚至超过了o1-preview，这表明它具有强大的数学推理能力。
```

### 流式输出

```python
qa_engine = index.as_query_engine(streaming=True)  
response = qa_engine.query("deepseek v3数学能力怎么样?")  
response.print_response_stream()
```

```output
DeepSeek-V3在数学相关基准测试中表现出色，特别是在非长链思维开放源码和闭源模型中处于领先地位。它在某些特定的基准测试，如MATH-500上，甚至超过了o1-preview的表现，这证明了其强大的数学推理能力。
```

## 6.2 多轮对话（Chat Engine）

```python
chat_engine = index.as_chat_engine()  
response = chat_engine.chat("deepseek v3数学能力怎么样?")  
print(response)
```

```output
DeepSeek-V3在数学相关基准测试中表现出色，特别是在非长链思维开放源码和闭源模型中处于领先地位。它在特定的基准测试如MATH-500上甚至超过了o1-preview的表现，这证明了其强大的数学推理能力。
```

```python
response = chat_engine.chat("代码能力呢?")  
print(response)
```

```output
DeepSeek-V3在编码相关的任务上表现非常出色，特别是在编程竞赛的基准测试中，如LiveCodeBench，它被认为是该领域中的领先模型。对于工程相关的任务，虽然DeepSeek-V3的表现略低于Claude-Sonnet-3.5，但它仍然显著优于其他所有模型，显示了其在多种技术基准测试中的竞争力。
```

### 流式输出
```python
chat_engine = index.as_chat_engine()  
streaming_response = chat_engine.stream_chat("deepseek v3数学能力怎么样?")  
# streaming_response.print_response_stream()  
for token in streaming_response.response_gen:  
    print(token, end="", flush=True)

```

```output

DeepSeek-V3在数学相关基准测试中表现出色，特别是在非长链思维开放源码和闭源模型中处于领先地位。它在某些特定的基准测试，如MATH-500上，甚至超过了o1-preview的表现，这证明了其强大的数学推理能力。
```


# 7. 底层接口：Prompt、LLM 与 Embedding

## 7.1 Prompt 模板

### 7.1.1  `PromptTemplate` 定义提示词模板
```python
from llama_index.core import PromptTemplate
prompt = PromptTemplate("写一个关于{topic}的笑话")
prompt.format(topic="小明")
```

### 7.1.2  `ChatPromptTemplate` 定义多轮消息模板
```python
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate

chat_text_qa_msgs = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content="你叫{name}，你必须根据用户提供的上下文回答问题。",
    ),
    ChatMessage(
        role=MessageRole.USER, 
        content=(
            "已知上下文：\n" \
            "{context}\n\n" \
            "问题：{question}"
        )
    ),
]
text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)

print(
    text_qa_template.format(
        name="小明",
        context="这是一个测试",
        question="这是什么"
    )
)
```

```output
system: 你叫小明，你必须根据用户提供的上下文回答问题。
user: 已知上下文：
这是一个测试

问题：这是什么
assistant: 
```

## 7.2  语言模型
```python
from llama_index.llms.openai import OpenAI
llm = OpenAI(temperature=0, model="gpt-4o")

response = llm.complete(prompt.format(topic="小明"))
print(response.text)

```

```python
response = llm.complete(
    text_qa_template.format(
        name="小明",
        context="这是一个测试",
        question="你是谁，我们在干嘛"
    )
)

print(response.text)
```

### 7.2.1 连接DeepSeek
```python
# !pip install llama-index-llms-deepseek
import os
from llama_index.llms.deepseek import DeepSeek

llm = DeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"), temperature=1.5)

response = llm.complete("写个笑话")
print(response)
```

### 7.2.2 设置全局使用的语言模型

```python
from llama_index.core import Settings

Settings.llm = DeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"), temperature=1.5)
```

除 OpenAI 外，LlamaIndex 已集成多个大语言模型，包括云服务 API 和本地部署 API，详见官方文档：[Available LLM integrations](https://docs.llamaindex.ai/en/stable/module_guides/models/llms/modules/)


## 7.3 Embedding 模型


```python
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

# 全局设定
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=512)
```

LlamaIndex 同样集成了多种 Embedding 模型，包括云服务 API 和开源模型（HuggingFace）等，详见[官方文档](https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings/)。



# 8. 基于 LlamaIndex 实现一个功能较完整的 RAG 系统
功能要求：
- 加载指定目录的文件
- 支持 RAG-Fusion
- 使用 Qdrant 向量数据库，并持久化到本地
- 支持检索后排序
- 支持多轮对话

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

EMBEDDING_DIM = 1536
COLLECTION_NAME = "full_demo"
PATH = "./qdrant_db"

client = QdrantClient(path=PATH)
```


```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core.postprocessor import LLMRerank, SimilarityPostprocessor
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels

# 1. 指定全局llm与embedding模型
Settings.llm = DashScope(model_name=DashScopeGenerationModels.QWEN_MAX,api_key=os.getenv("DASHSCOPE_API_KEY"))
Settings.embed_model = DashScopeEmbedding(model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V1)

# 2. 指定全局文档处理的 Ingestion Pipeline
Settings.transformations = [SentenceSplitter(chunk_size=512, chunk_overlap=200)]

# 3. 加载本地文档
documents = SimpleDirectoryReader("./data").load_data()

if client.collection_exists(collection_name=COLLECTION_NAME):
    client.delete_collection(collection_name=COLLECTION_NAME)

# 4. 创建 collection
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
)

# 5. 创建 Vector Store
vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)

# 6. 指定 Vector Store 的 Storage 用于 index
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# 7. 定义检索后排序模型
reranker = LLMRerank(top_n=2)
# 最终打分低于0.6的文档被过滤掉
sp = SimilarityPostprocessor(similarity_cutoff=0.6)

# 8. 定义 RAG Fusion 检索器
fusion_retriever = QueryFusionRetriever(
    [index.as_retriever()],
    similarity_top_k=5, # 检索召回 top k 结果
    num_queries=3,  # 生成 query 数
    use_async=False,
    # query_gen_prompt="",  # 可以自定义 query 生成的 prompt 模板
)

# 9. 构建单轮 query engine
query_engine = RetrieverQueryEngine.from_args(
    fusion_retriever,
    node_postprocessors=[reranker],
    response_synthesizer=get_response_synthesizer(
        response_mode = ResponseMode.REFINE
    )
)

# 10. 对话引擎
chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine, 
    # condense_question_prompt="" # 可以自定义 chat message prompt 模板
)
```


```python
# 测试多轮对话
# User: deepseek v3有多少参数
# User: 每次激活多少

while True:
    question=input("User:")
    if question.strip() == "":
        break
    response = chat_engine.chat(question)
    print(f"AI: {response}")
```

    AI: DeepSeek V3总共有671亿个参数。
    AI: DeepSeek V3每次激活370亿个参数。


# 9. 工作流（Workflow）简介

工作流顾名思义是对一些列工作步骤的抽象。

LlamaIndex 的工作流是事件（`event`）驱动的：

- 工作流由 `step` 组成
- 每个 `step` 处理特定的事件
- `step` 也会产生新的事件（交由后继的 `step` 进行处理）
- 直到产生 `StopEvent` 整个工作流结束

举个具体的例子：用自然语言查询数据库，数据库中包含多张表

工作流：

![workflow.png](https://learning-1316972768.cos.ap-nanjing.myqcloud.com/%E5%90%8E%E6%9C%9F/202408/workflow.png)


分步说明：

1. 用户输入自然语言查询
2. 系统先去检索跟查询相关的表
3. 根据表的 Schema 让大模型生成 SQL
4. 用生成的 SQL 查询数据库
5. 根据查询结果，调用大模型生成自然语言回复

### 9.1 数据准备


```python
# 下载 WikiTableQuestions
# WikiTableQuestions 是一个为表格问答设计的数据集。其中包含 2,108 个从维基百科提取的 HTML 表格

# !wget "https://github.com/ppasupat/WikiTableQuestions/releases/download/v1.0.2/WikiTableQuestions-1.0.2-compact.zip" -O wiki_data.zip
# !unzip wiki_data.zip
```

1. 遍历目录加载表格
```python
import pandas as pd
from pathlib import Path

data_dir = Path("./WikiTableQuestions/csv/200-csv")
csv_files = sorted([f for f in data_dir.glob("*.csv")])
dfs = []
for csv_file in csv_files:
    print(f"processing file: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
        dfs.append(df)
    except Exception as e:
        print(f"Error parsing {csv_file}: {str(e)}")
```

2. 为每个表生成一段文字表述（用于检索），保存在 `WikiTableQuestions_TableInfo` 目录


```python
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage


class TableInfo(BaseModel):
    """Information regarding a structured table."""

    table_name: str = Field(
        ..., description="table name (must be underscores and NO spaces)"
    )
    table_summary: str = Field(
        ..., description="short, concise summary/caption of the table"
    )


prompt_str = """\
Give me a summary of the table with the following JSON format.

- The table name must be unique to the table and describe it while being concise. 
- Do NOT output a generic table name (e.g. table, my_table).

Do NOT make the table name one of the following: {exclude_table_name_list}

Table:
{table_str}

Summary: """
prompt_tmpl = ChatPromptTemplate(
    message_templates=[ChatMessage.from_str(prompt_str, role="user")]
)

llm = DashScope(model_name=DashScopeGenerationModels.QWEN_MAX, api_key=os.getenv("DASHSCOPE_API_KEY"))
```


```python
tableinfo_dir = "WikiTableQuestions_TableInfo"
# !mkdir {tableinfo_dir}
```


```python
import json


def _get_tableinfo_with_index(idx: int) -> str:
    results_gen = Path(tableinfo_dir).glob(f"{idx}_*")
    results_list = list(results_gen)
    if len(results_list) == 0:
        return None
    elif len(results_list) == 1:
        path = results_list[0]
        with open(path, 'r') as file:
            data = json.load(file) 
            return TableInfo.model_validate(data)
    else:
        raise ValueError(
            f"More than one file matching index: {list(results_gen)}"
        )
    

table_names = set()
table_infos = []
for idx, df in enumerate(dfs):
    table_info = _get_tableinfo_with_index(idx)
    if table_info:
        table_infos.append(table_info)
    else:
        while True:
            df_str = df.head(10).to_csv()
            table_info = llm.structured_predict(
                TableInfo,
                prompt_tmpl,
                table_str=df_str,
                exclude_table_name_list=str(list(table_names)),
            )
            table_name = table_info.table_name
            print(f"Processed table: {table_name}")
            if table_name not in table_names:
                table_names.add(table_name)
                break
            else:
                # try again
                print(f"Table name {table_name} already exists, trying again.")
                pass

        out_file = f"{tableinfo_dir}/{idx}_{table_name}.json"
        json.dump(table_info.dict(), open(out_file, "w"))
    table_infos.append(table_info)
```



3. 将上述表格存入 SQLite 数据库


```python
# put data into sqlite db
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
)
import re


# Function to create a sanitized column name
def sanitize_column_name(col_name):
    # Remove special characters and replace spaces with underscores
    return re.sub(r"\W+", "_", col_name)


# Function to create a table from a DataFrame using SQLAlchemy
def create_table_from_dataframe(
    df: pd.DataFrame, table_name: str, engine, metadata_obj
):
    # Sanitize column names
    sanitized_columns = {col: sanitize_column_name(col) for col in df.columns}
    df = df.rename(columns=sanitized_columns)

    # Dynamically create columns based on DataFrame columns and data types
    columns = [
        Column(col, String if dtype == "object" else Integer)
        for col, dtype in zip(df.columns, df.dtypes)
    ]

    # Create a table with the defined columns
    table = Table(table_name, metadata_obj, *columns)

    # Create the table in the database
    metadata_obj.create_all(engine)

    # Insert data from DataFrame into the table
    with engine.connect() as conn:
        for _, row in df.iterrows():
            insert_stmt = table.insert().values(**row.to_dict())
            conn.execute(insert_stmt)
        conn.commit()


# engine = create_engine("sqlite:///:memory:")
engine = create_engine("sqlite:///wiki_table_questions.db")
metadata_obj = MetaData()
for idx, df in enumerate(dfs):
    tableinfo = _get_tableinfo_with_index(idx)
    print(f"Creating table: {tableinfo.table_name}")
    create_table_from_dataframe(df, tableinfo.table_name, engine, metadata_obj)
```


## 9.2 构建基础工具

1. 创建基于表的描述的向量索引

- `ObjectIndex` 是一个 LlamaIndex 内置的模块，通过索引 (Index）检索任意 Python 对象
- 这里我们使用 `VectorStoreIndex` 也就是向量检索，并通过 `SQLTableNodeMapping` 将文本描述的 `node` 和数据库的表形成映射
- 相关文档：https://docs.llamaindex.ai/en/stable/examples/objects/object_index/#the-objectindex-class


```python
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index.core import SQLDatabase, VectorStoreIndex

sql_database = SQLDatabase(engine)

table_node_mapping = SQLTableNodeMapping(sql_database)
table_schema_objs = [
    SQLTableSchema(table_name=t.table_name, context_str=t.table_summary)
    for t in table_infos
]  # add a SQLTableSchema for each table

obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex,
)
obj_retriever = obj_index.as_retriever(similarity_top_k=3)
```

2. 创建 SQL 查询器


```python
from llama_index.core.retrievers import SQLRetriever
from typing import List

sql_retriever = SQLRetriever(sql_database)


def get_table_context_str(table_schema_objs: List[SQLTableSchema]):
    """Get table context string."""
    context_strs = []
    for table_schema_obj in table_schema_objs:
        table_info = sql_database.get_single_table_info(
            table_schema_obj.table_name
        )
        if table_schema_obj.context_str:
            table_opt_context = " The table description is: "
            table_opt_context += table_schema_obj.context_str
            table_info += table_opt_context

        context_strs.append(table_info)
    return "\n\n".join(context_strs)
```

3. 创建 Text 2 SQL 的提示词（系统默认模板），和输出结果解析器（从生成的文本中抽取 SQL）


```python
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatResponse

def parse_response_to_sql(chat_response: ChatResponse) -> str:
    """Parse response to SQL."""
    response = chat_response.message.content
    sql_query_start = response.find("SQLQuery:")
    if sql_query_start != -1:
        response = response[sql_query_start:]
        # TODO: move to removeprefix after Python 3.9+
        if response.startswith("SQLQuery:"):
            response = response[len("SQLQuery:") :]
    sql_result_start = response.find("SQLResult:")
    if sql_result_start != -1:
        response = response[:sql_result_start]
    return response.strip().strip("```").strip()


text2sql_prompt = DEFAULT_TEXT_TO_SQL_PROMPT.partial_format(
    dialect=engine.dialect.name
)
print(text2sql_prompt.template)
```

4. 创建自然语言回复生成模板


```python
response_synthesis_prompt_str = (
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n"
    "SQL: {sql_query}\n"
    "SQL Response: {context_str}\n"
    "Response: "
)
response_synthesis_prompt = PromptTemplate(
    response_synthesis_prompt_str,
)
```


```python
llm = DashScope(model_name=DashScopeGenerationModels.QWEN_MAX, api_key=os.getenv("DASHSCOPE_API_KEY"))
```


## 9.3 定义工作流


```python
from llama_index.core.workflow import (
    Workflow,
    StartEvent,
    StopEvent,
    step,
    Context,
    Event,
)

# 事件：找到了数据库中相关的表
class TableRetrieveEvent(Event):
    """Result of running table retrieval."""

    table_context_str: str
    query: str

# 事件：文本转为了 SQL
class TextToSQLEvent(Event):
    """Text-to-SQL event."""

    sql: str
    query: str


class TextToSQLWorkflow1(Workflow):
    """Text-to-SQL Workflow that does query-time table retrieval."""

    def __init__(
        self,
        obj_retriever,
        text2sql_prompt,
        sql_retriever,
        response_synthesis_prompt,
        llm,
        *args,
        **kwargs
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self.obj_retriever = obj_retriever
        self.text2sql_prompt = text2sql_prompt
        self.sql_retriever = sql_retriever
        self.response_synthesis_prompt = response_synthesis_prompt
        self.llm = llm

    @step
    def retrieve_tables(
        self, ctx: Context, ev: StartEvent
    ) -> TableRetrieveEvent:
        """Retrieve tables."""
        table_schema_objs = self.obj_retriever.retrieve(ev.query)
        table_context_str = get_table_context_str(table_schema_objs)
        print("====\n"+table_context_str+"\n====")
        return TableRetrieveEvent(
            table_context_str=table_context_str, query=ev.query
        )

    @step
    def generate_sql(
        self, ctx: Context, ev: TableRetrieveEvent
    ) -> TextToSQLEvent:
        """Generate SQL statement."""
        fmt_messages = self.text2sql_prompt.format_messages(
            query_str=ev.query, schema=ev.table_context_str
        )
        chat_response = self.llm.chat(fmt_messages)
        sql = parse_response_to_sql(chat_response)
        print("====\n"+sql+"\n====")
        return TextToSQLEvent(sql=sql, query=ev.query)

    @step
    def generate_response(self, ctx: Context, ev: TextToSQLEvent) -> StopEvent:
        """Run SQL retrieval and generate response."""
        retrieved_rows = self.sql_retriever.retrieve(ev.sql)
        print("====\n"+str(retrieved_rows)+"\n====")
        fmt_messages = self.response_synthesis_prompt.format_messages(
            sql_query=ev.sql,
            context_str=str(retrieved_rows),
            query_str=ev.query,
        )
        chat_response = llm.chat(fmt_messages)
        return StopEvent(result=chat_response)
```


```python
workflow = TextToSQLWorkflow1(
    obj_retriever,
    text2sql_prompt,
    sql_retriever,
    response_synthesis_prompt,
    llm,
    verbose=True,
)
```


```python
response = await workflow.run(
    query="What was the year that The Notorious B.I.G was signed to Bad Boy?"
)
print(str(response))
```

## 9.4 可视化工作流


```python
# !pip install llama-index-utils-workflow
```


```python
from llama_index.utils.workflow import draw_all_possible_flows

draw_all_possible_flows(
    TextToSQLWorkflow1, filename="text_to_sql_table_retrieval.html"
)
```

## 9.5 工作流管理框架意义是什么

思考以下情况：

- `step` 的执行顺序有逻辑分支
- `step` 的执行有循环
- `step` 的执行可以并行
- 一个 `step` 的触发条件依赖前面若干 `step` 的结果，且它们之间可能有循环或者并行

![workflow2.png](https://learning-1316972768.cos.ap-nanjing.myqcloud.com/%E5%90%8E%E6%9C%9F/202408/workflow2.png)


所以，工作流管理框架的意思是便于将单个事件的处理逻辑和事件之间的执行顺序独立开

关于 LlamaIndex 工作流的更详细文档：https://docs.llamaindex.ai/en/stable/examples/workflow/workflows_cookbook/



# 10. LlamaIndex 的更多功能

- 智能体（Agent）开发框架：https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/
- RAG 的评测：https://docs.llamaindex.ai/en/stable/module_guides/evaluating/
- 过程监控：https://docs.llamaindex.ai/en/stable/module_guides/observability/

以上内容涉及较多背景知识，暂时不在本课展开，相关知识会在后面课程中逐一详细讲解。

此外，LlamaIndex 针对生产级的 RAG 系统中遇到的各个方面的细节问题，总结了很多高端技巧（[Advanced Topics](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)），对实战很有参考价值，非常推荐有能力的同学阅读。