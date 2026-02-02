import os
import time
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pinecone import Pinecone, ServerlessSpec

def ingest_knowledge_base():
    load_dotenv()
    
    file_path = "KNOWLEDGE_BASE.md"
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    with open(file_path, "r") as f:
        text = f.read()

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(text)

    print(f"Split into {len(md_header_splits)} chunks")

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        print("PINECONE_API_KEY not found in environment variables")
        return

    index_name = "acme-dental-index" 

    pc = Pinecone(api_key=pinecone_api_key)

    # Check if index exists, create if not
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        print(f"Creating index {index_name}...")
        try:
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            while not pc.describe_index(index_name).status["ready"]:
                time.sleep(1)
        except Exception as e:
            print(f"Error creating index: {e}")
            return

    print("Upserting to Pinecone...")
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        PineconeVectorStore.from_documents(
            documents=md_header_splits,
            embedding=embeddings,
            index_name=index_name,
        )
        print("Ingestion complete.")
    except Exception as e:
        print(f"Error during upsert: {e}")

if __name__ == "__main__":
    ingest_knowledge_base()
