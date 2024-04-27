from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import HuggingFaceHub, Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import MarkdownTextSplitter
from typing import List

def initChatModel(llm: LLM):
    from langchain.schema import (
        HumanMessage,
        SystemMessage,
    )
    messages = [
        SystemMessage(content="You're a helpful assistant"),
        HumanMessage(
            content="What happens when an unstoppable force meets an immovable object?"
        ),
    ]
    chat_model = ChatHuggingFace(llm=llm)
    res = chat_model.invoke(messages)
    print(res)

def initOllamaModel() -> LLM:
    llm = Ollama(model="llama2")
    return llm

def initHfModel() -> LLM:
    return HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        model_kwargs={
            "max_new_tokens": 512,
            "top_k": 30,
            "temperature": 0.1,
            "repetition_penalty": 1.03,
        },
    )

def initOllamaEmbeddings() -> Embeddings:
    return OllamaEmbeddings(model="llama2")

def initHfEmbeddings() -> Embeddings:
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def loadDocuments(docs_dir):
    from langchain_community.document_loaders import DirectoryLoader
    loader = DirectoryLoader(docs_dir, glob="**/*.md")
    docs = loader.load()
    return docs

def createRetriever(embeddings: Embeddings, directories: List[str]):
    text_splitter = MarkdownTextSplitter()
    documents = [document for directory in directories for document in loadDocuments(directory)]
    texts = text_splitter.split_documents(documents)
    db = FAISS.from_documents(texts, embeddings)
    return db.as_retriever()

def createTestRetriever(embeddings: Embeddings):
    db = FAISS.from_texts(["self-assessment is cool"], embeddings)
    return db.as_retriever()

def createChatPromptTemplate():
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    return ChatPromptTemplate.from_template(template)

llm = initOllamaModel()
#response = llm.invoke("tell me a dad joke")

dirs = ["/home/vikas/.nb/devx","/home/vikas/.nb/janus"]
embeddings = initOllamaEmbeddings()
retriever = createRetriever(embeddings, dirs)
# testRetriever = createTestRetriever(embeddings)
promptWithRetriever = {
        "context": retriever,
        "question": RunnablePassthrough()
    } | createChatPromptTemplate()
prompt = ChatPromptTemplate.from_template("Summarize topic {topic}")
chain = (
    promptWithRetriever
    | llm
    | StrOutputParser()
)
while True:
    question = input("Ask me a question about your docs: ")
    response = chain.invoke({"question": question})
    print(response)
