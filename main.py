import os
from typing import Any, List, Tuple

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama


def load_speech(path: str = "speech.txt"):
    if not os.path.exists(path):
        raise FileNotFoundError("speech.txt not found. Place it in the project folder.")

    loader = TextLoader(path, encoding="utf-8")
    return loader.load()


def split_documents(documents) -> List[Any]:
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma.from_documents(chunks, embedding=embeddings)


def build_qa_components(vectorstore) -> Tuple[Ollama, Any]:
    llm = Ollama(model="mistral")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return llm, retriever


def answer_question(llm: Ollama, retriever, question: str):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    if not context.strip():
        context = "No relevant context found in the text."

    prompt = f"""
CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
Answer strictly using the context.
If the answer is not in the context, say you do not know.

ANSWER:
"""

    response = llm.invoke(prompt)
    return str(response).strip(), docs


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

        if sources:
            print("\nSource:")
            for i, doc in enumerate(sources, 1):
                print(f"[{i}] {doc.page_content[:180].replace('\n', ' ')}...")
        print()


def main():
    docs = load_speech()
    chunks = split_documents(docs)
    store = build_vectorstore(chunks)
    llm, retriever = build_qa_components(store)
    chat_loop(llm, retriever)


if __name__ == "__main__":
    main()
