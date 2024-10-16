from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


def qa_agent(openai_api_key, memory, uploaded_file, question):
    model = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key, base_url="https://api.aigc369.com/v1")
    file_content = uploaded_file.read()  # 读取 返回buytes
    temp_file_path = "temp.pdf"   #临时储存pdf数据
    with open(temp_file_path, "wb") as temp_file:  #"wb"模式是wb 写入二进制
        temp_file.write(file_content)
    loader = PyPDFLoader(temp_file_path)   #打开上传的pdf
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(   #分隔器
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n", "。", "！", "？", "，", "、", ""]
    )
    texts = text_splitter.split_documents(docs)

    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large",api_key=openai_api_key,base_url="https://api.aigc369.com/v1")  #使用模型
    db = FAISS.from_documents(texts, embeddings_model)  #导入FAISS向量数据库
    retriever = db.as_retriever()   #检索器
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,  #模型
        retriever=retriever,  #检索器
        memory=memory   #记忆
    )
    response = qa.invoke({"chat_history": memory, "question": question})
    return response
