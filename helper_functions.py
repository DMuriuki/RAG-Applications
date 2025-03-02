import asyncio
import random
import textwrap
from enum import Enum
from typing import List

import numpy as np
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain import PromptTemplate
from openai import RateLimitError
from rank_bm25 import BM25Okapi
import fitz


def replace_tabs_with_space(documents: List) -> List:
    """
    Replaces all tab characters ('\t') with spaces in the page content of each document.

    Args:
        documents (List): A list of document objects with 'page_content' attributes.

    Returns:
        List: The modified list of documents with tab characters replaced by spaces.
    """
    for doc in documents:
        doc.page_content = doc.page_content.replace('\t', ' ')  # Replace tabs with spaces
    return documents


def wrap_text(text: str, width: int = 120) -> str:
    """
    Wraps the input text to the specified width.

    Args:
        text (str): The input text to wrap.
        width (int): The width at which to wrap the text.

    Returns:
        str: The wrapped text.
    """
    return textwrap.fill(text, width=width)


def encode_pdf_to_vector_store(path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> FAISS:
    """
    Encodes a PDF document into a vector store using OpenAI embeddings.

    Args:
        path (str): Path to the PDF file.
        chunk_size (int): The size of each chunk of text.
        chunk_overlap (int): The overlap between chunks.

    Returns:
        FAISS: A vector store containing the encoded content.
    """
    loader = PyPDFLoader(path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_tabs_with_space(texts)

    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(cleaned_texts, embeddings)


def encode_text_to_vector_store(content: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> FAISS:
    """
    Encodes a string into a vector store using OpenAI embeddings.

    Args:
        content (str): The text content to be encoded.
        chunk_size (int): The size of each chunk of text.
        chunk_overlap (int): The overlap between chunks.

    Returns:
        FAISS: A vector store containing the encoded content.

    Raises:
        ValueError: If the input content is not valid.
        RuntimeError: If there is an error during encoding.
    """
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Content must be a non-empty string.")

    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        raise ValueError("chunk_overlap must be a non-negative integer.")

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.create_documents([content])

        # Assign metadata to each chunk
        for chunk in chunks:
            chunk.metadata['relevance_score'] = 1.0

        embeddings = OpenAIEmbeddings()
        return FAISS.from_documents(chunks, embeddings)

    except Exception as e:
        raise RuntimeError(f"Error during encoding: {str(e)}")


def retrieve_context_for_question(question: str, query_retriever) -> List[str]:
    """
    Retrieves relevant context for a given question using the provided query retriever.

    Args:
        question (str): The question for which to retrieve context.

    Returns:
        List[str]: A list of relevant document contents.
    """
    docs = query_retriever.get_relevant_documents(question)
    return [doc.page_content for doc in docs]


class QuestionAnswerContext(BaseModel):
    """
    Model for generating answers based on the provided context.
    
    Attributes:
        answer_based_on_content (str): The generated answer.
    """
    answer_based_on_content: str = Field(description="Answer based on the provided context.")


def create_question_answer_chain(llm) -> PromptTemplate:
    """
    Creates a chain of reasoning for answering a question based on context.

    Args:
        llm: The language model to be used for generating answers.

    Returns:
        PromptTemplate: A chain combining the prompt template and language model.
    """
    question_answer_prompt_template = """
    For the question below, provide a concise but sufficient answer based ONLY on the provided context:
    {context}
    Question
    {question}
    """

    question_answer_prompt = PromptTemplate(
        template=question_answer_prompt_template,
        input_variables=["context", "question"],
    )

    return question_answer_prompt | llm.with_structured_output(QuestionAnswerContext)


def answer_question_from_context(
    question: str, context: List[str], question_answer_chain: PromptTemplate
) -> dict:
    """
    Answers a question using the provided context.

    Args:
        question (str): The question to be answered.
        context (List[str]): The context to be used for answering the question.
        question_answer_chain (PromptTemplate): The chain for answering the question.

    Returns:
        dict: A dictionary with the answer, context, and question.
    """
    input_data = {"question": question, "context": context}
    output = question_answer_chain.invoke(input_data)
    return {"answer": output.answer_based_on_content, "context": context, "question": question}


def display_context(context: List[str]) -> None:
    """
    Displays the provided context with headings.

    Args:
        context (List[str]): A list of context items to be displayed.
    """
    for i, item in enumerate(context, 1):
        print(f"Context {i}:")
        print(item)
        print("\n")


def read_pdf_content(path: str) -> str:
    """
    Extracts the text content of a PDF file.

    Args:
        path (str): The file path to the PDF document.

    Returns:
        str: The concatenated text content of the PDF document.
    """
    doc = fitz.open(path)
    return "".join(page.get_text() for page in doc)


def bm25_retrieve(bm25: BM25Okapi, cleaned_texts: List[str], query: str, top_k: int = 5) -> List[str]:
    """
    Retrieves the top k relevant text chunks using BM25 retrieval.

    Args:
        bm25 (BM25Okapi): The precomputed BM25 index.
        cleaned_texts (List[str]): The list of cleaned text chunks.
        query (str): The query string.
        top_k (int): The number of top documents to retrieve.

    Returns:
        List[str]: The top k relevant text chunks.
    """
    query_tokens = query.split()
    bm25_scores = bm25.get_scores(query_tokens)
    top_k_indices = np.argsort(bm25_scores)[::-1][:top_k]
    return [cleaned_texts[i] for i in top_k_indices]


async def exponential_backoff(attempt: int) -> None:
    """
    Implements exponential backoff with jitter for rate limiting.

    Args:
        attempt (int): The retry attempt number.
    """
    wait_time = (2 ** attempt) + random.uniform(0, 1)
    print(f"Rate limit hit. Retrying in {wait_time:.2f} seconds...")
    await asyncio.sleep(wait_time)


async def retry_with_backoff(coroutine, max_retries: int = 5):
    """
    Retries a coroutine with exponential backoff upon encountering a RateLimitError.

    Args:
        coroutine: The coroutine to retry.
        max_retries (int): The maximum number of retries.

    Returns:
        The result of the coroutine if successful.

    Raises:
        The last exception if all retries fail.
    """
    for attempt in range(max_retries):
        try:
            return await coroutine
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise e
            await exponential_backoff(attempt)
    raise Exception("Max retries reached")


class EmbeddingProvider(Enum):
    OPENAI = "openai"
    COHERE = "cohere"
    AMAZON_BEDROCK = "bedrock"


class ModelProvider(Enum):
    OPENAI = "openai"
    GROQ = "groq"
    ANTHROPIC = "anthropic"
    AMAZON_BEDROCK = "bedrock"


def get_embedding_provider(provider: EmbeddingProvider, model_id: str = None):
    """
    Returns the embedding provider based on the specified provider and model ID.

    Args:
        provider (EmbeddingProvider): The embedding provider to use.
        model_id (str, optional): The model ID to use for the provider.

    Returns:
        The appropriate embedding provider instance.

    Raises:
        ValueError: If the specified provider is unsupported.
    """
    if provider == EmbeddingProvider.OPENAI:
        return OpenAIEmbeddings()
    elif provider == EmbeddingProvider.COHERE:
        from langchain_cohere import CohereEmbeddings
        return CohereEmbeddings()
    elif provider == EmbeddingProvider.AMAZON_BEDROCK:
        from langchain_community.embeddings import BedrockEmbeddings
        return BedrockEmbeddings(model_id=model_id or "amazon.titan-embed-text-v2:0")
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
