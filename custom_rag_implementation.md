# Custom RAG Pipeline - Complete Implementation Guide

## Objective
Build a production-ready custom RAG pipeline with advanced retrieval techniques using OpenAI GPT-4o-mini and embeddings.

---

## Architecture Overview

```python
"""
Custom RAG Pipeline Architecture

Components:
1. Document Processor - Ingestion and chunking
2. Embedding Manager - Vector creation and storage
3. Retrieval Engine - Multi-stage retrieval with ranking
4. Query Router - Intent classification and routing
5. Context Optimizer - Compression and relevance filtering
6. Generation Engine - LLM-based answer generation
7. Evaluation Module - Quality metrics and monitoring
"""
```

---

## Installation & Setup

```python
# requirements_rag.txt
langchain>=0.1.0
langchain-openai>=0.0.5
langchain-community>=0.0.20
chromadb>=0.4.0
faiss-cpu>=1.7.4
sentence-transformers>=2.2.0
pypdf>=3.17.0
python-dotenv>=1.0.0
tiktoken>=0.5.0
rank-bm25>=0.2.2
cohere>=4.0.0  # For reranking
ragas>=0.1.0  # For evaluation
```

```bash
# Setup
pip install -r requirements_rag.txt

# Environment variables (.env)
OPENAI_API_KEY=your_openai_key
COHERE_API_KEY=your_cohere_key  # Optional for reranking
```

---

## Implementation

### 1. Document Processor

```python
# document_processor.py
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain.schema import Document
import hashlib
from pathlib import Path

class DocumentProcessor:
    """
    Advanced document processing with metadata extraction
    and intelligent chunking strategies.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        add_start_index: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Multiple splitting strategies
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=add_start_index,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def load_documents(self, file_path: str) -> List[Document]:
        """Load documents based on file type."""
        file_path = Path(file_path)

        loaders = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.md': UnstructuredMarkdownLoader,
        }

        loader_class = loaders.get(file_path.suffix)
        if not loader_class:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        loader = loader_class(str(file_path))
        documents = loader.load()

        # Add source metadata
        for doc in documents:
            doc.metadata['source_file'] = str(file_path)
            doc.metadata['file_type'] = file_path.suffix

        return documents

    def process_documents(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """Process and chunk documents with enhanced metadata."""

        # Split documents
        chunks = self.text_splitter.split_documents(documents)

        # Enhance metadata
        for i, chunk in enumerate(chunks):
            # Add chunk ID
            chunk_id = hashlib.md5(
                f"{chunk.metadata.get('source_file', '')}{i}".encode()
            ).hexdigest()
            chunk.metadata['chunk_id'] = chunk_id
            chunk.metadata['chunk_index'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)

            # Add semantic metadata (can be enhanced)
            chunk.metadata['has_code'] = '```' in chunk.page_content
            chunk.metadata['has_list'] = any(
                line.strip().startswith(('- ', '* ', '1.'))
                for line in chunk.page_content.split('\n')
            )

        return chunks

    def semantic_chunking(
        self,
        documents: List[Document],
        similarity_threshold: float = 0.7
    ) -> List[Document]:
        """
        Advanced: Semantic chunking based on content similarity.
        Groups similar sentences together.
        """
        # Implementation for semantic chunking
        # This would use embeddings to group similar content
        pass


# Example Usage
if __name__ == "__main__":
    processor = DocumentProcessor(
        chunk_size=1000,
        chunk_overlap=200
    )

    # Load and process documents
    docs = processor.load_documents("sample.pdf")
    chunks = processor.process_documents(docs)

    print(f"Processed {len(chunks)} chunks from document")
    print(f"Sample chunk metadata: {chunks[0].metadata}")
```

---

### 2. Embedding Manager

```python
# embedding_manager.py
from typing import List, Optional, Dict
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.schema import Document
import chromadb
from chromadb.config import Settings
import pickle
from pathlib import Path

class EmbeddingManager:
    """
    Manages embeddings creation, storage, and retrieval
    with multiple vector store support.
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        persist_directory: str = "./chroma_db",
        collection_name: str = "documents"
    ):
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            chunk_size=1000  # Batch size for API calls
        )

        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vectorstore = None

    def create_vectorstore(
        self,
        documents: List[Document],
        store_type: str = "chroma"
    ):
        """Create vector store from documents."""

        if store_type == "chroma":
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            self.vectorstore.persist()

        elif store_type == "faiss":
            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            # Save FAISS index
            self.vectorstore.save_local(self.persist_directory)

        return self.vectorstore

    def load_vectorstore(self, store_type: str = "chroma"):
        """Load existing vector store."""

        if store_type == "chroma":
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
        elif store_type == "faiss":
            self.vectorstore = FAISS.load_local(
                self.persist_directory,
                self.embeddings
            )

        return self.vectorstore

    def add_documents(self, documents: List[Document]):
        """Add new documents to existing vector store."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")

        self.vectorstore.add_documents(documents)

        # Persist if Chroma
        if isinstance(self.vectorstore, Chroma):
            self.vectorstore.persist()

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[Dict] = None
    ) -> List[Document]:
        """Perform similarity search."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")

        return self.vectorstore.similarity_search(
            query,
            k=k,
            filter=filter_dict
        )

    def mmr_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> List[Document]:
        """
        Maximum Marginal Relevance search for diverse results.
        lambda_mult: 0 = max diversity, 1 = max relevance
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")

        return self.vectorstore.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult
        )


# Example Usage
if __name__ == "__main__":
    from document_processor import DocumentProcessor

    # Process documents
    processor = DocumentProcessor()
    docs = processor.load_documents("sample.pdf")
    chunks = processor.process_documents(docs)

    # Create embeddings and vector store
    embedding_manager = EmbeddingManager()
    vectorstore = embedding_manager.create_vectorstore(chunks)

    # Search
    results = embedding_manager.similarity_search(
        "What is RAG?",
        k=3
    )

    print(f"Found {len(results)} results")
```

---

### 3. Advanced Retrieval Engine

```python
# retrieval_engine.py
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever
)
from langchain.retrievers.document_compressors import (
    LLMChainExtractor,
    EmbeddingsFilter
)
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi
import numpy as np

class AdvancedRetrievalEngine:
    """
    Multi-stage retrieval with ranking, filtering, and fusion.
    """

    def __init__(
        self,
        vectorstore,
        embeddings,
        llm_model: str = "gpt-4o-mini"
    ):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.llm = ChatOpenAI(model=llm_model, temperature=0)

    def create_base_retriever(self, k: int = 10):
        """Create base vector similarity retriever."""
        return self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )

    def create_mmr_retriever(self, k: int = 10):
        """Create MMR retriever for diversity."""
        return self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": k * 3,
                "lambda_mult": 0.5
            }
        )

    def create_compression_retriever(
        self,
        base_retriever,
        compression_type: str = "llm"
    ):
        """
        Create contextual compression retriever.
        Filters and compresses retrieved documents.
        """

        if compression_type == "llm":
            # LLM-based extraction
            compressor = LLMChainExtractor.from_llm(self.llm)
        else:
            # Embedding-based filtering
            compressor = EmbeddingsFilter(
                embeddings=self.embeddings,
                similarity_threshold=0.76
            )

        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

    def create_ensemble_retriever(
        self,
        retrievers: List,
        weights: Optional[List[float]] = None
    ):
        """
        Combine multiple retrievers with weighted fusion.
        """
        if weights is None:
            weights = [1.0 / len(retrievers)] * len(retrievers)

        return EnsembleRetriever(
            retrievers=retrievers,
            weights=weights
        )

    def hybrid_search(
        self,
        query: str,
        documents: List[Document],
        k: int = 5,
        alpha: float = 0.5
    ) -> List[Document]:
        """
        Hybrid search combining dense (vector) and sparse (BM25) retrieval.
        alpha: weight for vector search (1-alpha for BM25)
        """

        # Vector search
        vector_results = self.vectorstore.similarity_search_with_score(
            query, k=k*2
        )

        # BM25 search
        corpus = [doc.page_content for doc in documents]
        tokenized_corpus = [doc.split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split()
        bm25_scores = bm25.get_scores(tokenized_query)

        # Normalize scores
        vector_scores = np.array([score for _, score in vector_results])
        vector_scores = 1 - (vector_scores / vector_scores.max())
        bm25_scores = bm25_scores / bm25_scores.max()

        # Combine scores
        combined_scores = {}
        for i, (doc, score) in enumerate(vector_results):
            doc_id = doc.metadata.get('chunk_id', i)
            combined_scores[doc_id] = alpha * vector_scores[i]

        for i, score in enumerate(bm25_scores):
            doc_id = documents[i].metadata.get('chunk_id', i)
            if doc_id in combined_scores:
                combined_scores[doc_id] += (1 - alpha) * score
            else:
                combined_scores[doc_id] = (1 - alpha) * score

        # Get top-k
        sorted_docs = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        # Return documents
        doc_map = {
            doc.metadata.get('chunk_id', i): doc
            for i, doc in enumerate(documents)
        }

        return [doc_map[doc_id] for doc_id, _ in sorted_docs if doc_id in doc_map]

    def rerank_results(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 3
    ) -> List[Document]:
        """
        Rerank retrieved documents using cross-encoder or LLM.
        """
        try:
            import cohere
            co = cohere.Client()

            # Prepare documents for reranking
            docs_text = [doc.page_content for doc in documents]

            # Rerank using Cohere
            results = co.rerank(
                model="rerank-english-v2.0",
                query=query,
                documents=docs_text,
                top_n=top_k
            )

            # Return reranked documents
            return [documents[result.index] for result in results]

        except Exception as e:
            print(f"Reranking failed: {e}, returning original order")
            return documents[:top_k]


# Example Usage
if __name__ == "__main__":
    from embedding_manager import EmbeddingManager
    from document_processor import DocumentProcessor

    # Setup
    processor = DocumentProcessor()
    docs = processor.load_documents("sample.pdf")
    chunks = processor.process_documents(docs)

    embedding_manager = EmbeddingManager()
    vectorstore = embedding_manager.create_vectorstore(chunks)

    # Create retrieval engine
    retrieval_engine = AdvancedRetrievalEngine(
        vectorstore=vectorstore,
        embeddings=embedding_manager.embeddings
    )

    # Example: Hybrid search
    results = retrieval_engine.hybrid_search(
        query="Explain RAG architecture",
        documents=chunks,
        k=5,
        alpha=0.7
    )

    print(f"Retrieved {len(results)} documents")
```

---

### 4. Query Router & Transformer

```python
# query_processor.py
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class QueryIntent(BaseModel):
    """Query intent classification."""
    intent_type: str = Field(description="Type of query: factual, analytical, procedural, etc.")
    complexity: str = Field(description="Query complexity: simple, medium, complex")
    requires_context: bool = Field(description="Whether query requires retrieved context")
    suggested_k: int = Field(description="Suggested number of documents to retrieve")

class QueryProcessor:
    """
    Process and transform queries for optimal retrieval.
    """

    def __init__(self, llm_model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0)

    def classify_intent(self, query: str) -> QueryIntent:
        """Classify query intent to route appropriately."""

        parser = PydanticOutputParser(pydantic_object=QueryIntent)

        prompt = ChatPromptTemplate.from_template(
            """Analyze the following query and classify its intent.

Query: {query}

{format_instructions}
"""
        )

        chain = prompt | self.llm | parser

        result = chain.invoke({
            "query": query,
            "format_instructions": parser.get_format_instructions()
        })

        return result

    def expand_query(self, query: str, n_expansions: int = 3) -> List[str]:
        """
        Generate multiple query variations for multi-query retrieval.
        """

        prompt = ChatPromptTemplate.from_template(
            """Generate {n} different versions of the following query to improve retrieval.
Make them semantically similar but with different wordings.

Original Query: {query}

Generated Queries (one per line):
"""
        )

        chain = prompt | self.llm

        result = chain.invoke({
            "query": query,
            "n": n_expansions
        })

        # Parse expansions
        expansions = [
            line.strip() for line in result.content.split('\n')
            if line.strip() and not line.strip().startswith('#')
        ]

        return [query] + expansions[:n_expansions]

    def decompose_query(self, query: str) -> List[str]:
        """
        Decompose complex query into simpler sub-queries.
        """

        prompt = ChatPromptTemplate.from_template(
            """Break down the following complex query into simpler sub-queries.
Each sub-query should be self-contained and answerable independently.

Complex Query: {query}

Sub-queries (one per line):
"""
        )

        chain = prompt | self.llm

        result = chain.invoke({"query": query})

        sub_queries = [
            line.strip() for line in result.content.split('\n')
            if line.strip() and line.strip()[0].isdigit()
        ]

        return sub_queries

    def step_back_prompt(self, query: str) -> str:
        """
        Generate a higher-level, more abstract query for better retrieval.
        """

        prompt = ChatPromptTemplate.from_template(
            """Given the specific query below, generate a more general, higher-level query
that would help retrieve broader context.

Specific Query: {query}

General Query:
"""
        )

        chain = prompt | self.llm

        result = chain.invoke({"query": query})

        return result.content.strip()

    def hypothetical_document(self, query: str) -> str:
        """
        HyDE: Generate hypothetical ideal document for query.
        Use this document's embedding for retrieval.
        """

        prompt = ChatPromptTemplate.from_template(
            """Write a hypothetical document that would perfectly answer this query:

Query: {query}

Hypothetical Document:
"""
        )

        chain = prompt | self.llm

        result = chain.invoke({"query": query})

        return result.content.strip()


# Example Usage
if __name__ == "__main__":
    processor = QueryProcessor()

    query = "How do I implement a custom RAG pipeline with reranking?"

    # Classify intent
    intent = processor.classify_intent(query)
    print(f"Intent: {intent.intent_type}, K: {intent.suggested_k}")

    # Expand query
    expansions = processor.expand_query(query, n_expansions=2)
    print(f"Query expansions: {expansions}")

    # Step-back prompting
    general_query = processor.step_back_prompt(query)
    print(f"General query: {general_query}")
```

---

### 5. Generation Engine

```python
# generation_engine.py
from typing import List, Dict, Optional, Any
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class GenerationEngine:
    """
    Generate answers using retrieved context with citations.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0,
        streaming: bool = False
    ):
        callbacks = [StreamingStdOutCallbackHandler()] if streaming else []

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            streaming=streaming,
            callbacks=callbacks
        )

    def generate_answer(
        self,
        query: str,
        context_docs: List[Document],
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """Generate answer with citations from context."""

        # Prepare context
        context = "\n\n".join([
            f"[Source {i+1}]: {doc.page_content}"
            for i, doc in enumerate(context_docs)
        ])

        # Create prompt
        prompt = ChatPromptTemplate.from_template(
            """Answer the question based solely on the provided context.
If the answer cannot be found in the context, say so.
Always cite your sources using [Source N] notation.

Context:
{context}

Question: {query}

Answer:
"""
        )

        chain = prompt | self.llm

        # Generate answer
        result = chain.invoke({
            "context": context,
            "query": query
        })

        response = {
            "answer": result.content,
            "sources": []
        }

        if include_sources:
            response["sources"] = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in context_docs
            ]

        return response

    def generate_with_chain_of_thought(
        self,
        query: str,
        context_docs: List[Document]
    ) -> Dict[str, Any]:
        """Generate answer with explicit reasoning steps."""

        context = "\n\n".join([
            f"[Source {i+1}]: {doc.page_content}"
            for i, doc in enumerate(context_docs)
        ])

        prompt = ChatPromptTemplate.from_template(
            """Answer the question using step-by-step reasoning.

Context:
{context}

Question: {query}

Let's solve this step by step:
1. First, I'll identify relevant information from the context
2. Then, I'll reason through the problem
3. Finally, I'll provide the answer with citations

Answer:
"""
        )

        chain = prompt | self.llm

        result = chain.invoke({
            "context": context,
            "query": query
        })

        return {
            "answer": result.content,
            "reasoning": "explicit",
            "sources": [{"content": doc.page_content, "metadata": doc.metadata}
                       for doc in context_docs]
        }


# Example Usage
if __name__ == "__main__":
    from retrieval_engine import AdvancedRetrievalEngine

    # Assume we have retrieval_engine and query
    query = "What is RAG?"
    context_docs = []  # Retrieved documents

    generator = GenerationEngine()
    response = generator.generate_answer(query, context_docs)

    print(response["answer"])
```

---

### 6. Complete RAG Pipeline

```python
# rag_pipeline.py
from typing import Dict, Any, Optional, List
from document_processor import DocumentProcessor
from embedding_manager import EmbeddingManager
from retrieval_engine import AdvancedRetrievalEngine
from query_processor import QueryProcessor
from generation_engine import GenerationEngine

class CustomRAGPipeline:
    """
    Complete end-to-end RAG pipeline with all components.
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
        persist_directory: str = "./vector_db"
    ):
        # Initialize components
        self.doc_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager(
            embedding_model=embedding_model,
            persist_directory=persist_directory
        )
        self.query_processor = QueryProcessor(llm_model=llm_model)
        self.generator = GenerationEngine(model_name=llm_model)

        self.retrieval_engine = None
        self.documents = []

    def ingest_documents(self, file_paths: List[str]):
        """Ingest and process documents."""

        all_chunks = []
        for file_path in file_paths:
            docs = self.doc_processor.load_documents(file_path)
            chunks = self.doc_processor.process_documents(docs)
            all_chunks.extend(chunks)

        self.documents = all_chunks

        # Create vector store
        vectorstore = self.embedding_manager.create_vectorstore(
            all_chunks,
            store_type="chroma"
        )

        # Initialize retrieval engine
        self.retrieval_engine = AdvancedRetrievalEngine(
            vectorstore=vectorstore,
            embeddings=self.embedding_manager.embeddings
        )

        return len(all_chunks)

    def query(
        self,
        query: str,
        retrieval_strategy: str = "hybrid",
        k: int = 5,
        use_reranking: bool = True
    ) -> Dict[str, Any]:
        """
        Execute complete RAG pipeline.

        Args:
            query: User query
            retrieval_strategy: "basic", "mmr", "hybrid", "multi-query"
            k: Number of documents to retrieve
            use_reranking: Whether to rerank results
        """

        # Step 1: Process query
        intent = self.query_processor.classify_intent(query)
        k = intent.suggested_k or k

        # Step 2: Retrieve documents
        if retrieval_strategy == "basic":
            context_docs = self.embedding_manager.similarity_search(query, k=k)

        elif retrieval_strategy == "mmr":
            context_docs = self.embedding_manager.mmr_search(query, k=k)

        elif retrieval_strategy == "hybrid":
            context_docs = self.retrieval_engine.hybrid_search(
                query, self.documents, k=k
            )

        elif retrieval_strategy == "multi-query":
            # Multi-query retrieval
            query_variations = self.query_processor.expand_query(query, n_expansions=2)
            all_docs = []
            for q in query_variations:
                docs = self.embedding_manager.similarity_search(q, k=k)
                all_docs.extend(docs)

            # Remove duplicates
            seen = set()
            context_docs = []
            for doc in all_docs:
                doc_id = doc.metadata.get('chunk_id', doc.page_content)
                if doc_id not in seen:
                    seen.add(doc_id)
                    context_docs.append(doc)

            context_docs = context_docs[:k]

        else:
            raise ValueError(f"Unknown retrieval strategy: {retrieval_strategy}")

        # Step 3: Rerank (optional)
        if use_reranking and len(context_docs) > 3:
            context_docs = self.retrieval_engine.rerank_results(
                query, context_docs, top_k=min(k, len(context_docs))
            )

        # Step 4: Generate answer
        response = self.generator.generate_answer(
            query, context_docs, include_sources=True
        )

        # Add pipeline metadata
        response['metadata'] = {
            'query_intent': intent.intent_type,
            'retrieval_strategy': retrieval_strategy,
            'num_retrieved': len(context_docs),
            'reranking_used': use_reranking
        }

        return response


# Example Usage
if __name__ == "__main__":
    # Initialize pipeline
    rag = CustomRAGPipeline()

    # Ingest documents
    num_chunks = rag.ingest_documents([
        "document1.pdf",
        "document2.pdf"
    ])
    print(f"Ingested {num_chunks} chunks")

    # Query
    result = rag.query(
        "What is RAG and how does it work?",
        retrieval_strategy="hybrid",
        k=5,
        use_reranking=True
    )

    print(f"\nAnswer: {result['answer']}")
    print(f"\nMetadata: {result['metadata']}")
    print(f"\nSources: {len(result['sources'])}")
```

---

## Evaluation Framework

```python
# evaluation.py
from typing import List, Dict
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall
)
from datasets import Dataset

def evaluate_rag_system(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str]
) -> Dict:
    """
    Evaluate RAG system using RAGAS metrics.
    """

    # Prepare dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truths": ground_truths
    }

    dataset = Dataset.from_dict(data)

    # Evaluate
    result = evaluate(
        dataset,
        metrics=[
            answer_relevancy,
            faithfulness,
            context_precision,
            context_recall
        ]
    )

    return result


# Example usage
if __name__ == "__main__":
    # Test cases
    questions = ["What is RAG?"]
    answers = ["RAG stands for..."]
    contexts = [["RAG is a technique..."]]
    ground_truths = ["RAG is..."]

    results = evaluate_rag_system(
        questions, answers, contexts, ground_truths
    )

    print(results)
```

---

## Production Considerations

### 1. Caching Strategy
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_embedding(text: str):
    """Cache embeddings for frequently used queries."""
    return embeddings.embed_query(text)
```

### 2. Async Processing
```python
import asyncio

async def async_retrieve(queries: List[str]):
    """Process multiple queries concurrently."""
    tasks = [retriever.ainvoke(q) for q in queries]
    return await asyncio.gather(*tasks)
```

### 3. Monitoring
```python
import time
from prometheus_client import Counter, Histogram

# Metrics
query_counter = Counter('rag_queries_total', 'Total queries')
latency_histogram = Histogram('rag_latency_seconds', 'Query latency')

def query_with_monitoring(query: str):
    query_counter.inc()
    start_time = time.time()

    result = rag_pipeline.query(query)

    latency_histogram.observe(time.time() - start_time)
    return result
```

---

## Next Steps

1. Implement each component
2. Test with sample documents
3. Benchmark different strategies
4. Add error handling
5. Implement monitoring
6. Deploy to production

---

**Remember**: This is a complete template. Use Claude Code to generate working implementations for each section!
