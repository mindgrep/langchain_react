from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.llms import HuggingFaceHub, Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

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
    llm = Ollama(model="llama2:7b")
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
    return OllamaEmbeddings(model="llama2:7b")

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

def createRetriever(embeddings: Embeddings, directory: str):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = loadDocuments(directory)
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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
dir = "/home/vikas/.nb/work"
embeddings = initOllamaEmbeddings()
#retriever = createRetriever(embeddings, dir)
testRetriever = createTestRetriever(embeddings)
promptWithRetriever = {
        "context": testRetriever,
        "question": RunnablePassthrough()
    } | createChatPromptTemplate()
prompt = ChatPromptTemplate.from_template("Fun math fact about {country}")
chain = (
    prompt
    | llm
    | StrOutputParser()
)
while True:
    country = input("Give me a country and I will tell you something interesting about it: ")
    response = chain.invoke({"country": country})
    print(response)
