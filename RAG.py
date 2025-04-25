import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter


pdf_dir = "Data"
all_documents = []

for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        path = os.path.join(pdf_dir, filename)
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_documents.extend(docs)  


splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)
split_docs = splitter.split_documents(all_documents)

print(f"Loaded {len(all_documents)} docs")
print(f"Split into {len(split_docs)} chunks")


embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(split_docs, embeddings)

vectorstore.save_local("pdf_index_store")

llm = ChatOpenAI(model="gpt-4.1")

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4})
)

query = "what do lions hunt?"

retrieved_docs = rag_chain.retriever.get_relevant_documents(query)

print("\n--- Retrieved Chunks ---")
for i, doc in enumerate(retrieved_docs):
    print(f"\nChunk {i+1}:\n{doc.page_content}")


response = rag_chain.invoke({"query": query})


print("Answer:", response)
