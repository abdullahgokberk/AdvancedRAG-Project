from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]

#İç içe listelerde urller içindeki itemleri doc_list olarak kaydediyoruz.
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000,chunk_overlap=200)

splits = text_splitter.split_documents(docs_list)

embeddings = OpenAIEmbeddings()

#Kayıt kısmı
#Database'in harddisk'e gerçekten kaydedilmesi için persist_directory="./.chroma" yapıyoruz. Bu isimli klasöre yükleyecek.
vectorstore = Chroma.from_documents(
    documents=splits,
    collection_name="rag-chroma",
    embedding=embeddings,
    persist_directory="./.chroma",
)

#Geri çağırdığımız kısım
retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=embeddings
).as_retriever()










