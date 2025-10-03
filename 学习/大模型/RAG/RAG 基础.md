

# 1 大模型应用开发的三种模式

> [!note] Thinking
> Prompt vs RAG vs Fine-tuning 什么时候使用？


![image.png](https://learning-1316972768.cos.ap-nanjing.myqcloud.com/%E5%AD%A6%E4%B9%A0/largeModel/20251002202115899.png)

# 2 什么是 RAG ？
RAG（Retrieval-Augmented Generation）

- 检索增强生成，是一种结合<font color="#ff0000">信息检索</font>（Retrieval）和<font color="#ff0000">文本生成</font>（Generation）的技术
- RAG技术通过实时检索相关文档或信息，并**将其作为上下文输入到生成模型**中，从而提高生成结果的时效性和准确性。
    
![image.png](https://learning-1316972768.cos.ap-nanjing.myqcloud.com/%E5%AD%A6%E4%B9%A0/largeModel/20251002202516826.png)

## 2.1 RAG 的优势是什么？

- **解决知识时效性问题：** 大模型的训练数据通常是静态的，无法涵盖最新信息，而RAG可以检索外部知识库实时更新信息
    
- **减少模型幻觉：** 通过引入外部知识，RAG能够减少模型生成虚假或不准确内容的可能性
    
- **提升专业领域回答质量：** RAG能够结合垂直领域的专业知识库，生成更具专业深度的回答
    
- **生成内容的溯源（可解释性）**
    

## 2.2 RAG 的核心原理与流程
![image.png](https://learning-1316972768.cos.ap-nanjing.myqcloud.com/%E5%AD%A6%E4%B9%A0/largeModel/20251002202933136.png)

**Step1：数据预处理，构建索引库**

- 知识库构建：收集并整理文档、网页、数据库等多源数据，构建外部知识库
    
- 文档分块：将文档切分为适当大小的片段（chunks），以便后续检索。分块策略需要在语义完整性与检索效率之间取得平衡
    
- 向量化处理：使用嵌入模型（如BGE、M3E、Chinese-Alpaca-2等）将文本块转换为向量，并存储在向量数据库中
    

**Step2：检索阶段**

- 查询处理：将用户**输入的问题转换为向量**，并在向量数据库中进行相似度检索，找到**最相关的文本片段**
    
- **重排序（较为复杂）**：对检索结果进行相关性排序，选择最相关的片段作为生成阶段的输入
    

**Step3：生成阶段**

- <span style="background:#ff4d4f">上下文组装——prompt 重写</span>：将**检索到的文本片段与用户问题结合**，形成增强的上下文输入
    
- 生成回答：大语言模型基于增强的上下文生成最终回答
    ​

**划重点： RAG 本质上就是重构了一个新的 Prompt！**

​

# 3 NativeRAG
![image.png](https://learning-1316972768.cos.ap-nanjing.myqcloud.com/%E5%AD%A6%E4%B9%A0/largeModel/20251002203451746.png)

NativeRAG的步骤：
- Indexing => 如何更好地把知识存起来。

- Retrieval => 如何在大量的知识中，找到一小部分有用的，给到模型参考。
    
- Generation => 如何结合用户的提问和检索到的知识，让模型生成有用的答案。
    
> [!attention] 划重点
> 上面三个步骤虽然看似简单，但在 RAG 应用从构建到落地实施的整个过程中，涉及较多复杂的工作内容！



​

# 4 LangChain快速搭建本地知识库检索

## 4.1 环境准备
1. 本地安装好 Conda 环境
2. 推荐使用阿里大模型平台百炼：[https://bailian.console.aliyun.com/](https://bailian.console.aliyun.com/)
3. 百炼平台使用
    - 注册登录
    - 申请api key


## 4.2 搭建流程
1. 文档加载，并按一定条件**切割**成片段

2. 将切割的文本片段灌入**检索引擎**
    
3. 封装**检索接口**
    
4. 构建**调用流程**：Query -> 检索 -> Prompt -> LLM -> 回复
    
### 4.2.1 知识库构建——准备阶段

```python
# !pip install pypdf2 读取pdf 
# !pip install dashscope  
# !pip install langchain  
# !pip install langchain-openai  
# !pip install langchain-community  
# !pip install faiss-cpu

import os  
import logging  
import pickle  
from PyPDF2 import PdfReader  
from langchain.chains.question_answering import load_qa_chain  
from langchain_openai import OpenAI, ChatOpenAI  
from langchain_openai import OpenAIEmbeddings  
from langchain_community.embeddings import DashScopeEmbeddings  
from langchain_community.callbacks.manager import get_openai_callback  
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.vectorstores import FAISS  
from typing import List, Tuple  
​  
def extract_text_with_page_numbers(pdf) -> Tuple[str, List[int]]:  
    """  
    从PDF中提取文本并记录每行文本对应的页码  
      
    参数:  
        pdf: PDF文件对象  
      
    返回:  
        text: 提取的文本内容  
        page_numbers: 每行文本对应的页码列表  
    """  
    text = ""  
    page_numbers = []  
​  
    for page_number, page in enumerate(pdf.pages, start=1):  
        extracted_text = page.extract_text()  
        if extracted_text:  
            text += extracted_text  
            page_numbers.extend([page_number] * len(extracted_text.split("\n")))  
        else:  
            logging.warning(f"No text found on page {page_number}.")  
​  
    return text, page_numbers
```  
​  

```python
def process_text_with_splitter(text: str, page_numbers: List[int], save_path: str = None) -> FAISS:  
    """  
    处理文本并创建向量存储  
      
    参数:  
        text: 提取的文本内容  
        page_numbers: 每行文本对应的页码列表  
        save_path: 可选，保存向量数据库的路径  
      
    返回:  
        knowledgeBase: 基于FAISS的向量存储对象  
    """  
    # 创建文本分割器，用于将长文本分割成小块，根据512token分割——bert编码器
    # 但是各个chunk之间有128个字符的重叠，保证语义的完整性
    text_splitter = RecursiveCharacterTextSplitter(  
        separators = ["\n\n", "\n", ".", " ", ""],  
        chunk_size = 512,  
        chunk_overlap = 128,  
        length_function = len,  
    )  
​  
    # 分割文本  
    chunks = text_splitter.split_text(text)  
    # logging.debug(f"Text split into {len(chunks)} chunks.")  
    print(f"文本被分割成 {len(chunks)} 个块。")  
          
    # 创建嵌入模型，OpenAI嵌入模型，配置环境变量 OPENAI_API_KEY  
    # embeddings = OpenAIEmbeddings()  
​  
    # 调用阿里百炼平台文本嵌入模型，配置环境变量 DASHSCOPE_API_KEY  
    embeddings = DashScopeEmbeddings(  
        model = "text-embedding-v2"  
    )  
    # 从文本块创建知识库  
    knowledgeBase = FAISS.from_texts(chunks, embeddings)  
    print("已从文本块创建知识库...")  
      
    # 存储每个文本块对应的页码信息  
    page_info = {chunk: page_numbers[i] for i, chunk in enumerate(chunks)}  
    knowledgeBase.page_info = page_info  
​  
    # 如果提供了保存路径，则保存向量数据库和页码信息  
    if save_path:  
        # 确保目录存在  
        os.makedirs(save_path, exist_ok=True)  
          
        # 保存FAISS向量数据库  
        knowledgeBase.save_local(save_path)  
        print(f"向量数据库已保存到: {save_path}")  
          
        # 保存页码信息到同一目录  
        with open(os.path.join(save_path, "page_info.pkl"), "wb") as f:  
            pickle.dump(page_info, f)  
        print(f"页码信息已保存到: {os.path.join(save_path, 'page_info.pkl')}")  
      
    return knowledgeBase 

def load_knowledge_base(load_path: str, embeddings = None) -> FAISS:  
    """  
    从磁盘加载向量数据库和页码信息  
      
    参数:  
        load_path: 向量数据库的保存路径  
        embeddings: 可选，嵌入模型。如果为None，将创建一个新的DashScopeEmbeddings实例  
      
    返回:  
        knowledgeBase: 加载的FAISS向量数据库对象  
    """  
    # 如果没有提供嵌入模型，则创建一个新的  
    if embeddings is None:  
        embeddings = DashScopeEmbeddings(  
            model="text-embedding-v2"  
        )  
      
    # 加载FAISS向量数据库，添加allow_dangerous_deserialization=True参数以允许反序列化  
    knowledgeBase = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)  
    print(f"向量数据库已从 {load_path} 加载。")  
      
    # 加载页码信息  
    page_info_path = os.path.join(load_path, "page_info.pkl")  
    if os.path.exists(page_info_path):  
        with open(page_info_path, "rb") as f:  
            page_info = pickle.load(f)  
        knowledgeBase.page_info = page_info  
        print("页码信息已加载。")  
    else:  
        print("警告: 未找到页码信息文件。")  
      
    return knowledgeBase     
``` 
​  
### 4.2.2 知识库构建
 ```python
 # 读取PDF文件  
 pdf_reader = PdfReader('./浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf')  
 # 提取文本和页码信息  
 text, page_numbers = extract_text_with_page_numbers(pdf_reader) 
 print(f"提取的文本长度: {len(text)} 个字符。") 
 
 # 处理文本并创建知识库，同时保存到磁盘  
 save_dir = "./vector_db"  
 knowledgeBase = process_text_with_splitter(text, page_numbers, save_path=save_dir)  \  
 ```
​  

```output
提取的文本长度: 3881 个字符。  
文本被分割成 10 个块。  
已从文本块创建知识库...  
向量数据库已保存到: ./vector_db  
页码信息已保存到: ./vector_db\page_info.pkl  
文本被分割成 10 个块。  
已从文本块创建知识库...

<langchain_community.vectorstores.faiss.FAISS at 0x1d0ffedb220>
```

### 4.2.3 设置查询问题
<span style="background:rgba(2, 170, 11, 0.55)">检索时，需要和构建数据库的 embeddingModel 一致</span>
- ChatOpenAI 方式
```python
# query = "客户经理被投诉了，投诉一次扣多少分"  
query = "客户经理每年评聘申报时间是怎样的？"  
if query:  
    # 执行相似度搜索，找到与查询相关的文档  
    docs = knowledgeBase.similarity_search(query)  
      
    # 初始化对话大模型  
    chatLLM  = ChatOpenAI(  
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",  
        api_key = os.getenv("DASHSCOPE_API_KEY"),  
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",  
        model = "deepseek-v3"  
    )  
      
    # 加载问答链  
    chain = load_qa_chain(chatLLM, chain_type="stuff")  
    # 准备输入数据  
    input_data = {"input_documents": docs, "question": query}  
​  
    # 使用回调函数跟踪API调用成本  
    with get_openai_callback() as cost:  
        # 执行问答链  
        response = chain.invoke(input=input_data)  
        print(f"查询已处理。成本: {cost}")  
        print(response["output_text"])  
        print("来源:")  
​  
    # 记录唯一的页码  
    unique_pages = set()  
​  
    # 显示每个文档块的来源页码  
    for doc in docs:  
        text_content = getattr(doc, "page_content", "")  
        source_page = knowledgeBase.page_info.get(  
            text_content.strip(), "未知"  
        )  
​  
        if source_page not in unique_pages:  
            unique_pages.add(source_page)  
            print(f"文本块页码: {source_page}")
```

```outpot
查询已处理。成本: Tokens Used: 1289
	Prompt Tokens: 1240
		Prompt Tokens Cached: 0
	Completion Tokens: 49
		Reasoning Tokens: 0
Successful Requests: 1
Total Cost (USD): $0.0
根据第十一条的规定，客户经理每年评聘的申报时间是每年的1月份。由分行人力资源部和个人业务部在每年的2月份组织统一的资格考试。考试合格者由分行颁发个金客户经理资格证书，其有效期为一年。
来源:
文本块页码: 1
```

- Tongyi 方式
```python
from langchain_community.llms import Tongyi

# 设置查询问题
# query = "客户经理被投诉了，投诉一次扣多少分？"
query = "客户经理每年评聘申报时间是怎样的？"
if query:
    # 示例：如何加载已保存的向量数据库
    # 注释掉以下代码以避免在当前运行中重复加载
    # 创建嵌入模型，embeddingModle需要和构建向量数据库时保持一致
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v2"
    )
    # 从磁盘加载向量数据库
    loaded_knowledgeBase = load_knowledge_base("./vector_db", embeddings)
    # 使用加载的知识库进行查询
    docs = loaded_knowledgeBase.similarity_search(query)
    
    # 初始化对话大模型
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY"),
    llm = Tongyi(model_name="deepseek-v3", dashscope_api_key=DASHSCOPE_API_KEY)
    
    # 加载问答链
    chain = load_qa_chain(llm, chain_type="stuff")

    # 准备输入数据
    input_data = {"input_documents": docs, "question": query}

    # 使用回调函数跟踪API调用成本
    with get_openai_callback() as cost:
        # 执行问答链
        response = chain.invoke(input=input_data)
        print(f"查询已处理。成本: {cost}")
        print(response["output_text"])
        print("来源:")

    # 记录唯一的页码
    unique_pages = set()

    # 显示每个文档块的来源页码
    for doc in docs:
        text_content = getattr(doc, "page_content", "")
        source_page = knowledgeBase.page_info.get(
            text_content.strip(), "未知"
        )

        if source_page not in unique_pages:
            unique_pages.add(source_page)
            print(f"文本块页码: {source_page}")

```

```output
向量数据库已从 ./vector_db 加载。
页码信息已加载。
查询已处理。成本: Tokens Used: 0
	Prompt Tokens: 0
		Prompt Tokens Cached: 0
	Completion Tokens: 0
		Reasoning Tokens: 0
Successful Requests: 1
Total Cost (USD): $0.0
客户经理每年的评聘申报时间是每年一月份。由分行人力资源部、个人业务部每年二月份组织统一的资格考试。考试合格者由分行颁发个金客户经理资格证书，其有效期为一年。
来源:
文本块页码: 1
```


> [!attention] 小结
> 

**1. PDF文本提取与处理**
- 使用PyPDF2库的PdfReader从PDF文件中提取文本在提取过程中记录每行文本对应的页码，便于后续溯源
- 使用RecursiveCharacterTextSplitter将长文本分割成小块，便于向量化处理
    
**2. 向量数据库构建**
- 使用OpenAIEmbeddings / DashScopeEmbeddings将文本块转换为向量表示
- 使用FAISS向量数据库存储文本向量，支持高效的相似度搜索为每个文本块保存对应的页码信息，实现查询结果溯源

**3. 语义搜索与问答链**
- 基于用户查询，使用similarity_search在向量数据库中检索相关文本块
- 使用文本语言模型和load_qa_chain构建问答链将检索到的文档和用户问题作为输入，生成回答
    
**4. 成本跟踪与结果展示**
- 使用get_openai_callback跟踪API调用成本
- 展示问答结果和来源页码，方便用户验证信息
    

# 5 三大阶段有效提升RAG质量方法
数据准备阶段==>知识检索阶段==>答案生成阶段

## 5.1 数据准备阶段（影响最大）
### 5.1.1 常见问题
- **数据质量差：** 企业大部分数据（尤其是非结构化数据）缺乏良好的数据治理，未经标记/评估的非结构化数据可能包含敏感、过时、矛盾或不正确的信息。
    
- **多模态信息：** 提取、定义和理解文档中的不同内容元素，如标题、配色方案、图像和标签等存在挑战。
    
- **复杂的PDF提取：** PDF是为人类阅读而设计的，机器解析起来非常复杂。
    

### 5.1.2 如何提升数据准备阶段的质量？
- 构建完整的数据准备流程
- 采用智能文档技术


- **（1）构建完整的数据准备流程**
	1. 数据评估与分类
		- 数据审计：全面审查现有数据，识别敏感、过时、矛盾或不准确的信息。
		- 数据分类：按类型、来源、敏感性和重要性对数据进行分类，便于后续处理。
	    
	2. 数据清洗
		- 去重：删除重复数据
		- 纠错：修正格式错误、拼写错误等
		- 更新：替换过时信息，确保数据时效性
		- 一致性检查：解决数据矛盾，确保逻辑一致
	    
	3. 敏感信息处理
		- 识别敏感数据：使用工具或正则表达式识别敏感信息，如个人身份信息
		- 脱敏或加密：对敏感数据进行脱敏处理，确保合规。
	    
	4. 数据标记与标注
		- 元数据标记：为数据添加元数据，如来源、创建时间等
		- 内容标注：对非结构化数据进行标注，便于后续检索和分析
	    
	5. 数据治理框架
		- 制定政策：明确数据管理、访问控制和更新流程
		- 责任分配：指定数据治理负责人，确保政策执行
		- 监控与审计：定期监控数据质量，进行审计
    

- **（2）智能文档技术**
  对于复杂的 pdf 处理，目前没有完美的解决方案，推荐使用智能文档技术
  ![image.png](https://learning-1316972768.cos.ap-nanjing.myqcloud.com/%E5%AD%A6%E4%B9%A0/largeModel/20251003112335218.png)

	- 阿里文档智能：[https://www.aliyun.com/product/ai/docmind?spm=a2c4g.11174283.0.0.bfe667a8tIVMdG](https://www.aliyun.com/product/ai/docmind?spm=a2c4g.11174283.0.0.bfe667a8tIVMdG)
	- 微软 LayoutLMv3：[https://www.microsoft.com/en-us/research/articles/layoutlmv3/](https://www.microsoft.com/en-us/research/articles/layoutlmv3/)

**其他方向：可以尝试使用 RagFlow 技术，不论是原声的 pdf 文档还是扫描文档，使用 ocr 的技术都可以解决**


## 5.2 知识检索阶段
### 5.2.1 常见问题
- **内容缺失：** 当检索过程缺少关键内容时，系统会提供不完整、碎片化的答案 => 降低RAG的质量

- **错过排名靠前的文档：** 用户查询相关的文档时被检索到，但相关性极低，导致答案不能满足用户需求，这是因为在检索过程中，用户通过主观判断决定检索“文档数量”。**理论上所有文档都要被排序并考虑进一步处理，但在实践中，通常只有排名top k的文档才会被召回，而k值需要根据经验确定。**
    
- **不在上下文中：** 从数据库中检索出包含答案的文档，但未能包含在生成答案的上下文中。这种情况通常发生在**返回大量文件时，需要进行整合**以选择最相关的信息。
    

### 5.2.2 如何提升知识检索阶段的质量？
- 通过查询转换澄清用户意图：**明确用户意图**，提高检索准确性——<span style="background:#ff4d4f">查询改写</span>
    
- 采用**混合检索**和**重排**策略：不单纯使用向量检索(如加上BM 2.5 关键字检索)，确保最相关的文档被优先处理，生成更准确的答案。
    

- **（1）通过查询转换澄清用户意图**
	- 场景：用户询问 “如何申请信用卡？”
	- 问题：用户意图可能模糊，例如不清楚是申请流程、所需材料还是资格条件。
	
	- 解决方法：通过查询转换明确用户意图。
	
	- 实现步骤：
		- <font color="#00b050">    意图识别+查询拓展</font>
		    - **意图识别**：使用自然语言处理技术识别用户意图。例如，识别用户是想了解流程、材料还是资格。
		    - **查询扩展**：根据识别结果扩展查询。例如：
		        - 如果用户想了解流程，查询扩展为“信用卡申请的具体步骤”
		        - 如果用户想了解材料，查询扩展为“申请信用卡需要哪些材料”
		        - 如果用户想了解资格，查询扩展为“申请信用卡的资格条件”
	            
	    - 检索：使用**扩展后的查询**检索相关文档
	- 示例：
	    1. 用户输入：“如何申请信用卡？”
	    2. 系统识别意图为 `流程`，扩展查询为 `信用卡申请的具体步骤`
	    3. 检索结果包含详细的申请步骤文档，系统生成准确答案

<font color="#00b050">目前 langChain、langIndex 等均支持返回多个 query查询</font>

- **（2）混合检索和重排策略**
	- 场景：用户询问“信用卡年费是多少？”
	- 问题：直接检索可能返回大量文档，部分相关但排名低，导致答案不准确。
	    
	- 解决方法：采用混合检索和重排策略。

	- 步骤：
	    1. 混合检索：结合**关键词检索和语义检索**
		    1. 关键词检索：“信用卡年费”        
		    2. 语义检索：使用嵌入模型检索与“信用卡年费”语义相近的文档
	    2. 重排：对检索结果进行重排。 
	    3. 生成答案：从重排后的文档中生成答案。
	        
	- 示例：
	    1. 用户输入：“信用卡年费是多少？”
	    2. 系统进行混合检索，结合关键词和语义检索。
	    3. 重排后，最相关的文档（如“信用卡年费政策”）排名靠前。
	    4. 系统生成准确答案：“信用卡年费根据卡类型不同，普通卡年费为100元，金卡为300元，白金卡为1000元。”
        

## 5.3 答案生成阶段

### 5.3.1 常见问题

- **未提取：** 答案与所提供的上下文相符，但大语言模型却<font color="#c00000">无法准确提取</font>。这种情况通常发生在上下文中存在过多噪音或相互冲突的信息时==><font color="#00b050">RagSft</font>
    
- **不完整：** 尽管能够利用上下文生成答案，但信息缺失会导致对用户查询的答复不完整。格式错误：当prompt中的附加指令格式不正确时，大语言模型可能误解或曲解这些指令，从而导致错误的答案。
    
- **幻觉：** 大模型可能会产生误导性或虚假性信息。
    

### 5.3.2 如何提升答案生成阶段的质量？
- 改进提示词模板
- 实施动态防护栏
    

**（1）改进提示词模板**

| 场景               | 原始提示词                    | 改进后的提示词                                         |
| ---------------- | ------------------------ | ----------------------------------------------- |
| 用户询问“如何申请信用卡？”   | “根据以下上下文回答问题：如何申请信用卡？”   | “根据以下上下文，提取与申请信用卡相关的具体步骤和所需材料：如何申请信用卡？”         |
| 用户询问“信用卡的年费是多少？” | “根据以下上下文回答问题：信用卡的年费是多少？” | “根据以下上下文，详细列出不同信用卡的年费信息，并说明是否有减免政策：信用卡的年费是多少？”  |
| 用户询问“什么是零存整取？”   | “根据以下上下文回答问题：什么是零存整取？”   | “根据以下上下文，准确解释零存整取的定义、特点和适用人群，确保信息真实可靠：什么是零存整取？” |

- **如何对原有的提示词进行优化？**
  可以通过 `DeepSeek-R1` 或 `QWQ` 的推理链，对<font color="#00b050">提示词进行优化</font>：
	- 信息提取：从原始提示词中提取关键信息。 
	- 需求分析：分析用户的需求，明确用户希望获取的具体信息。
	- 提示词优化：根据需求分析的结果，优化提示词，使其更具体、更符合用户的需求。
    

**（2）实施动态防护栏**

动态防护栏（Dynamic Guardrails）是一种在生成式AI系统中用于**实时监控和调整模型输出的机制**，旨在确保生成的内容符合预期、准确且安全。它通过设置规则、约束和反馈机制，动态地干预模型的生成过程，避免生成错误、不完整、不符合格式要求或含有**虚假信息（幻觉）**的内容。

在RAG系统中，动态防护栏的作用**尤为重要**，因为它可以帮助解决以下问题：

- 未提取：确保模型从上下文中提取了正确的信息。
- 不完整：确保生成的答案覆盖了所有必要的信息。
- 格式错误：确保生成的答案符合指定的格式要求。   
- 幻觉：防止模型生成与上下文无关或虚假的信息。
    

**场景1：防止未提取** ——agent

用户问题：“如何申请信用卡？”
- 上下文：包含申请信用卡的步骤和所需材料。
- 动态防护栏规则：检查生成的答案是否包含“步骤”和“材料”。如果缺失，提示模型重新生成。
    
- 示例：
    - 错误输出：“申请信用卡需要提供一些材料。”
    - 防护栏触发：检测到未提取具体步骤，提示模型补充。
        

**场景2：防止不完整**

用户问题：“信用卡的年费是多少？”
- 上下文：包含不同信用卡的年费信息。
    
- 动态防护栏规则：检查生成的答案是否列出所有信用卡的年费。如果缺失，提示模型补充。
    
- 示例：
    - 错误输出：“信用卡A的年费是100元。”
    - 防护栏触发：检测到未列出所有信用卡的年费，提示模型补充。
        

**场景3：防止幻觉**（**回复后校验**）

用户问题：“什么是零存整取？”

- 上下文：包含零存整取的定义和特点。
    
- 动态防护栏规则：检查**生成的答案是否与上下文一致**。如果不一致，提示模型重新生成。
    
- 示例：
    - 错误输出：“零存整取是一种贷款产品。
    - 防护栏触发：检测到与上下文不一致，提示模型重新生成。
        

- **如何实现动态防护栏技术？**
	- **事实性校验规则**，在生成阶段，设置规则验证生成内容是否与检索到的知识片段一致。例如，可以使用参考文献验证机制，确保生成内容有可靠来源支持，避免输出矛盾或不合理的回答。
		- **如何制定事实性校验规则？**
			- 当业务逻辑明确且规则较为固定时，可以**人为定义一组规则**，比如：
				- 规则1：生成的答案必须包含检索到的知识片段中的**关键实体**（如“年费”、“利率”）。
				- 规则2：生成的答案必须符合**指定的格式**（如步骤列表、表格等）。
			- **实施方法**：
			    - 使用**正则表达式或关键词匹配**来检查生成内容是否符合规则。
			    - 例如，检查生成内容是否包含“年费”这一关键词，或者是否符合步骤格式（如“1. 登录；2. 设置”）

# 6 RAG在不同阶段提升质量的实践

- **数据准备环节**，阿里云考虑到文档具有多层标题属性且不同标题之间存在关联性，提出多粒度知识提取方案，按照不同标题级别对文档进行拆分，然后基于Qwen14b模型和RefGPT训练了一个面向知识提取任务的专属模型，对各个粒度的chunk进行知识提取和组合，并通过去重和降噪的过程保证知识不丢失、不冗余。最终将文档知识提取成多个事实型对话，提升检索效果；
    
- **知识检索环节**，哈啰出行采用**多路召回**的方式，主要是**向量召回**和**搜索召回**。其中，向量召回使用了两类，一类是**大模型的向量**、另一类是传统**深度模型向量**；搜索召回也是多链路的，包括关键词、ngram等。通过多路召回的方式，可以达到较高的召回查全率。
    
- **答案生成环节**，中国移动为了解决事实性不足或逻辑缺失，采用FoRAG两阶段生成策略，首先生成大纲，然后基于大纲扩展生成最终答案。
    

# 7 QA

**如果LLM可以处理无限上下文了，RAG还有意义吗？**

- **效率与成本**：LLM处理长上下文时计算资源消耗大，响应时间增加。RAG通过检索相关片段，**减少输入长度**。
    
- **知识更新**：LLM的知识截止于训练数据，无法实时更新。RAG可以连接外部知识库，增强时效性。
    
- **可解释性**：RAG的检索过程透明，用户可查看来源，增强信任。LLM的生成过程则较难追溯。
    
- 定制化：RAG可针对特定领域定制检索系统，提供更精准的结果，而LLM的通用性可能无法满足特定需求。
    
- 数据隐私：RAG允许在本地或私有数据源上检索，避免敏感数据上传云端，适合隐私要求高的场景。
    
- 结合LLM的生成能力和RAG的检索能力，可以提升整体性能，提供更全面、准确的回答。
    

# 8 学习打卡

## 8.1 结合你的业务场景，创建本地知识检索

- Step1：收集整理知识库
    
- Step2：从PDF中提取文本并记录每行文本对应的页码
    
- Step3：处理文本并创建向量存储
    
- Step4：执行相似度搜索，找到与查询相关的文档
    
- Step5：使用问到链对用户问题进行回答
    
- Step6：显示每个文档块的来源页码
    

## 8.2 理解有效提升RAG质量的方法

- 如何提升数据准备阶段的质量？
    
- 如何提升知识检索阶段的质量？
    
- 如何提升答案生成阶段的质量？