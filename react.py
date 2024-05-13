#import cProfile
from typing import List

from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.document_loaders.text import TextLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms import HuggingFaceHub, Ollama
#from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.sqlitevss import SQLiteVSS
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import MarkdownHeaderTextSplitter

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
    from langchain_community.document_loaders.directory import DirectoryLoader
    loader = DirectoryLoader(docs_dir, glob="**/*.md", loader_cls=TextLoader)
    docs = loader.load()
    return docs

def createRetriever(embeddings: Embeddings, directories: List[str]):
    documents = [document for directory in directories for document in loadDocuments(directory)]
    markdown_header_text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#","header1"),("##","header2")])
    texts = [ text.page_content for document in documents[:1] for text in markdown_header_text_splitter.split_text(document.page_content)]
    conn = SQLiteVSS.create_connection(':memory:')
    db = SQLiteVSS('langchain', conn, embeddings, 'vss.db')
    db = SQLiteVSS.from_texts(texts, embeddings, check_same_thread=False)
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

def print_stats():
    import pstats
    p = pstats.Stats("react.prof")
    p.sort_stats("cumulative").print_stats(10)

def main():
    llm = initOllamaModel()
    #response = llm.invoke("tell me a dad joke")

    #dirs = ["/home/vikas/.nb/devx","/home/vikas/.nb/janus"]
    embeddings = initOllamaEmbeddings()
    retriever = createRetriever(embeddings, [])
    # testRetriever = createTestRetriever(embeddings)
    promptWithRetriever = {
            "context": retriever,
            "question": RunnablePassthrough()
        } | createChatPromptTemplate()
    #prompt = ChatPromptTemplate.from_template("Summarize topic {topic}")
    chain = (
        promptWithRetriever
        | llm
        | StrOutputParser()
    )
    while True:
        question = input("Ask me a question about your docs: ")
        response = chain.invoke({"question": question})
        print(response)

main()
#with cProfile.Profile() as pr:
    #response = llm.invoke("generate a hello world program in go with no explanation")
    #print(response)
    # pr.dump_stats("react.prof")
    # print_stats()
