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

postprocessor = LLMRerank(top_n=2) # 走一遍transformer
nodes = postprocessor.postprocess_nodes(nodes, query_str="deepseek v3有多少参数?")
for i, node in enumerate(nodes):
    print(f"[{i}] {node.text}")
```

更多的 Rerank 及其它后处理方法，参考官方文档：[Node Postprocessor Modules](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/)

