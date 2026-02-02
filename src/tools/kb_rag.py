from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


@tool
def retrieve_faq(query: str) -> str:
    """Search the clinic's knowledge base for answers to frequently asked questions."""

    index_name = "acme-dental-index"
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)

    docs = docsearch.similarity_search(query, k=3)
    return "\n\n".join([d.page_content for d in docs])
