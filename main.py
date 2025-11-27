import os
from typing import Any, List, Tuple

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama


# Load the speech file from disk and return it as LangChain Documents
def load_speech(path: str = "speech.txt"):
    if not os.path.exists(path):
        raise FileNotFoundError("speech.txt not found. Place it in the project folder.")

    loader = TextLoader(path, encoding="utf-8")
    return loader.load()


# Split the document into smaller overlapping chunks
# Smaller chunks improve retrieval accuracy
def split_documents(documents) -> List[Any]:
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,     # size of each chunk
        chunk_overlap=100,  # overlap to preserve continuity
        length_function=len,
    )
    return splitter.split_documents(documents)


# Convert each text chunk into an embedding and store it in ChromaDB
def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma.from_documents(chunks, embedding=embeddings)


# Initialize the LLM and connect the retriever to the vector database
def build_qa_components(vectorstore) -> Tuple[Ollama, Any]:
    # Mistral runs locally through Ollama
    llm = Ollama(model="mistral")

    # Retriever fetches the top matching chunks per query
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    return llm, retriever


# Handles one round of question-answering
def answer_question(llm: Ollama, retriever, question: str):
    # Retrieve the most relevant chunks from the vector store
    docs = retriever.invoke(question)

    # Combine retrieved chunks into one context block
    context = "\n\n".join(doc.page_content for doc in docs)

    # Fallback in case retrieval returns nothing useful
    if not context.strip():
        context = "No relevant context found in the text."

    # Prompt instructs the model to answer only from the retrieved context
    prompt = f"""
CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
Answer strictly using the context.
If the answer is not present, say so clearly.

ANSWER:
"""

    # Ask the local LLM to generate an answer
    response = llm.invoke(prompt)

    return str(response).strip(), docs


# Interactive CLI loop
def chat_loop(llm: Ollama, retriever):
    print("\nAmbedkarGPT is ready. Ask a question (type 'exit' to quit).\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        answer, sources = answer_question(llm, retriever, question)

        print("\nAmbedkarGPT:")
        print(answer)

        # Show small text snippets for transparency
        if sources:
            print("\nSource:")
            for i, doc in enumerate(sources, 1):
                print(f"[{i}] {doc.page_content[:180].replace('\n', ' ')}...")
        print()


# Entry point
def main():
    # Load and prepare the document
    docs = load_speech()
    chunks = split_documents(docs)

    # Create vector database
    store = build_vectorstore(chunks)

    # Setup LLM + retriever
    llm, retriever = build_qa_components(store)

    # Launch chat interface
    chat_loop(llm, retriever)


if __name__ == "__main__":
    main()
