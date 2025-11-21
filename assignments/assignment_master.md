# Complete Assignment Collection
## LangChain & LangGraph Course

---

# Module 1 Assignments

---

## Assignment 1.1: Configurable LLM Client
**Points:** 100 | **Time:** 30 minutes | **Difficulty:** Easy

### Objective
Create a reusable, configurable LLM client with proper error handling.

### Requirements
1. Create a `create_llm()` function that:
   - Accepts `model` (default: "gpt-4o-mini")
   - Accepts `temperature` (default: 0)
   - Validates API key exists
   - Returns configured ChatOpenAI instance
   - Raises `ValueError` if API key missing

2. Create a `query_llm()` function that:
   - Accepts an LLM and a prompt
   - Returns just the response content (string)
   - Handles errors gracefully

### Starter Code
```python
"""
Assignment 1.1: Configurable LLM Client
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import Optional


def create_llm(
    model: str = "gpt-4o-mini",
    temperature: float = 0
) -> ChatOpenAI:
    """
    Create a configured LLM instance.

    Args:
        model: Model name to use
        temperature: Creativity setting (0-1)

    Returns:
        Configured ChatOpenAI instance

    Raises:
        ValueError: If OPENAI_API_KEY is not set
    """
    # YOUR CODE HERE
    pass


def query_llm(llm: ChatOpenAI, prompt: str) -> str:
    """
    Send a query to the LLM and return the response.

    Args:
        llm: The LLM instance
        prompt: The prompt to send

    Returns:
        The response content as a string
    """
    # YOUR CODE HERE
    pass


# Test your implementation
if __name__ == "__main__":
    # Test 1: Create LLM with defaults
    print("Test 1: Default LLM creation")
    llm = create_llm()
    response = query_llm(llm, "Say 'Hello!'")
    print(f"  Response: {response}")
    assert "Hello" in response or "hello" in response.lower()
    print("  ✓ PASSED\n")

    # Test 2: Custom temperature
    print("Test 2: Custom temperature")
    llm_creative = create_llm(temperature=0.9)
    response = query_llm(llm_creative, "Invent a word.")
    print(f"  Response: {response[:50]}...")
    print("  ✓ PASSED\n")

    # Test 3: Error handling
    print("Test 3: Error handling")
    original = os.environ.get("OPENAI_API_KEY")
    del os.environ["OPENAI_API_KEY"]
    try:
        llm = create_llm()
        print("  ✗ FAILED - should have raised ValueError")
    except ValueError:
        print("  ✓ PASSED - correctly raised ValueError")
    finally:
        os.environ["OPENAI_API_KEY"] = original

    print("\n" + "="*50)
    print("All tests passed! Submit your assignment.")
```

### Grading Rubric
| Criteria | Points | Description |
|----------|--------|-------------|
| `create_llm` loads env vars | 15 | Calls `load_dotenv()` |
| `create_llm` validates API key | 20 | Checks key exists, raises `ValueError` if not |
| `create_llm` creates LLM correctly | 25 | Uses correct model and temperature |
| `query_llm` works correctly | 25 | Returns string content |
| All tests pass | 10 | No test failures |
| Code quality | 5 | Clean, readable, documented |
| **Total** | **100** | |

---

## Assignment 1.2: Prompt Template Library
**Points:** 100 | **Time:** 45 minutes | **Difficulty:** Easy

### Objective
Build a library of reusable prompt templates for common tasks.

### Requirements
Create a `PromptLibrary` class with these templates:

1. `translator()` - Translates text between languages
2. `summarizer()` - Summarizes text to specified length
3. `sentiment_analyzer()` - Analyzes sentiment of text
4. `question_answerer()` - Answers questions in a specific style

### Starter Code
```python
"""
Assignment 1.2: Prompt Template Library
"""
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


class PromptLibrary:
    """Collection of reusable prompt templates."""

    @staticmethod
    def translator() -> ChatPromptTemplate:
        """
        Translate text between languages.
        Variables: source_language, target_language, text
        """
        # YOUR CODE HERE
        pass

    @staticmethod
    def summarizer() -> ChatPromptTemplate:
        """
        Summarize text to specified length.
        Variables: text, num_sentences
        """
        # YOUR CODE HERE
        pass

    @staticmethod
    def sentiment_analyzer() -> ChatPromptTemplate:
        """
        Analyze sentiment of text.
        Variables: text
        Output format: POSITIVE, NEGATIVE, or NEUTRAL with explanation
        """
        # YOUR CODE HERE
        pass

    @staticmethod
    def question_answerer() -> ChatPromptTemplate:
        """
        Answer questions in a specific style.
        Variables: style (e.g., "like a teacher", "briefly"), question
        """
        # YOUR CODE HERE
        pass


# Test your implementation
if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    lib = PromptLibrary()

    print("Test 1: Translator")
    print("=" * 50)
    chain = lib.translator() | llm
    result = chain.invoke({
        "source_language": "English",
        "target_language": "French",
        "text": "Hello, how are you?"
    })
    print(f"Result: {result.content}")
    assert len(result.content) > 0
    print("✓ PASSED\n")

    print("Test 2: Summarizer")
    print("=" * 50)
    chain = lib.summarizer() | llm
    result = chain.invoke({
        "text": "Artificial intelligence (AI) is intelligence demonstrated by machines, "
                "as opposed to natural intelligence displayed by animals including humans. "
                "AI research has been defined as the field of study of intelligent agents.",
        "num_sentences": "1"
    })
    print(f"Result: {result.content}")
    print("✓ PASSED\n")

    print("Test 3: Sentiment Analyzer")
    print("=" * 50)
    chain = lib.sentiment_analyzer() | llm
    result = chain.invoke({"text": "I absolutely love this product! Best purchase ever!"})
    print(f"Result: {result.content}")
    assert "POSITIVE" in result.content.upper()
    print("✓ PASSED\n")

    print("Test 4: Question Answerer")
    print("=" * 50)
    chain = lib.question_answerer() | llm
    result = chain.invoke({
        "style": "like explaining to a 5-year-old",
        "question": "Why is the sky blue?"
    })
    print(f"Result: {result.content}")
    print("✓ PASSED\n")

    print("All tests passed!")
```

### Grading Rubric
| Criteria | Points | Description |
|----------|--------|-------------|
| `translator()` works correctly | 25 | Handles language pairs properly |
| `summarizer()` respects length | 25 | Summarizes to specified sentences |
| `sentiment_analyzer()` format | 25 | Returns POSITIVE/NEGATIVE/NEUTRAL |
| `question_answerer()` follows style | 15 | Adapts to requested style |
| All tests pass | 10 | No failures |
| **Total** | **100** | |

---

# Module 2 Assignments

---

## Assignment 2.1: Document Q&A with Citations
**Points:** 100 | **Time:** 60 minutes | **Difficulty:** Medium

### Objective
Build a Q&A system that answers questions from documents and cites sources.

### Requirements
1. Store documents with source metadata
2. Retrieve relevant documents for questions
3. Generate answers that cite sources
4. Handle "unknown" questions appropriately

### Complete Implementation Required
See `tutorials/module_02_rag_foundations/lesson_01_understanding_rag.md` for starter code.

### Grading Rubric
| Criteria | Points | Description |
|----------|--------|-------------|
| Documents stored correctly | 20 | With proper metadata |
| Retrieval works | 25 | Returns relevant docs |
| Citations accurate | 25 | Sources match content |
| Unknown handling | 20 | Says "I don't know" |
| Code quality | 10 | Clean and documented |
| **Total** | **100** | |

---

## Assignment 2.2: Multi-Document RAG System
**Points:** 100 | **Time:** 90 minutes | **Difficulty:** Medium

### Objective
Build a RAG system that can load and query multiple document types.

### Requirements
1. Support loading from:
   - Text files (.txt)
   - PDF files (.pdf)
   - Web URLs
2. Chunk documents intelligently
3. Allow filtering by source type
4. Return answers with metadata

### Starter Code
```python
"""
Assignment 2.2: Multi-Document RAG System
"""
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain.prompts import ChatPromptTemplate
from typing import List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class MultiDocRAG:
    """RAG system supporting multiple document types."""

    def __init__(self, persist_dir: str = "./multi_doc_db"):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.persist_dir = persist_dir
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def load_text_file(self, file_path: str) -> int:
        """
        Load a text file into the vector store.

        Args:
            file_path: Path to .txt file

        Returns:
            Number of chunks added
        """
        # YOUR CODE HERE
        pass

    def load_pdf_file(self, file_path: str) -> int:
        """
        Load a PDF file into the vector store.

        Args:
            file_path: Path to .pdf file

        Returns:
            Number of chunks added
        """
        # YOUR CODE HERE
        pass

    def load_web_page(self, url: str) -> int:
        """
        Load a web page into the vector store.

        Args:
            url: URL to load

        Returns:
            Number of chunks added
        """
        # YOUR CODE HERE
        pass

    def query(
        self,
        question: str,
        source_filter: Optional[str] = None,
        k: int = 4
    ) -> Dict:
        """
        Query the RAG system.

        Args:
            question: User's question
            source_filter: Optional filter ("txt", "pdf", "web")
            k: Number of documents to retrieve

        Returns:
            Dict with 'answer', 'sources', and 'source_types'
        """
        # YOUR CODE HERE
        pass


# Test your implementation
if __name__ == "__main__":
    # Create test files
    with open("test_doc.txt", "w") as f:
        f.write("LangChain is a framework for developing LLM applications. "
                "It provides tools for chains, agents, and memory.")

    rag = MultiDocRAG()

    # Test text file loading
    print("Test 1: Load text file")
    chunks = rag.load_text_file("test_doc.txt")
    print(f"  Loaded {chunks} chunks")
    assert chunks > 0
    print("  ✓ PASSED\n")

    # Test querying
    print("Test 2: Query system")
    result = rag.query("What is LangChain?")
    print(f"  Answer: {result['answer'][:100]}...")
    print(f"  Sources: {result['sources']}")
    assert "answer" in result
    assert "sources" in result
    print("  ✓ PASSED\n")

    # Cleanup
    import os
    os.remove("test_doc.txt")

    print("All tests passed!")
```

### Grading Rubric
| Criteria | Points | Description |
|----------|--------|-------------|
| Text file loading | 20 | Loads and chunks .txt files |
| PDF file loading | 20 | Loads and chunks .pdf files |
| Web page loading | 20 | Loads and chunks URLs |
| Query with filter | 20 | Filter by source type works |
| Metadata tracking | 10 | Source types tracked correctly |
| Code quality | 10 | Clean and documented |
| **Total** | **100** | |

---

# Module 3 Assignments

---

## Assignment 3.1: Advanced RAG Pipeline
**Points:** 150 | **Time:** 120 minutes | **Difficulty:** Hard

### Objective
Build a production-quality RAG pipeline with advanced retrieval techniques.

### Requirements
1. Implement hybrid search (vector + keyword)
2. Add query expansion
3. Implement reranking
4. Add answer confidence scoring
5. Include evaluation metrics

### Starter Code
```python
"""
Assignment 3.1: Advanced RAG Pipeline
"""
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RAGResult:
    """Container for RAG results."""
    answer: str
    sources: List[str]
    confidence: float
    retrieval_method: str


class AdvancedRAGPipeline:
    """Production-quality RAG with advanced features."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.vectorstore = None
        self.documents = []
        self.bm25 = None

    def add_documents(self, texts: List[str], sources: List[str]):
        """Add documents to the knowledge base."""
        # YOUR CODE HERE
        pass

    def vector_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Perform vector similarity search."""
        # YOUR CODE HERE
        pass

    def keyword_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Perform BM25 keyword search."""
        # YOUR CODE HERE
        pass

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.5
    ) -> List[str]:
        """
        Combine vector and keyword search.
        alpha: weight for vector search (1-alpha for keyword)
        """
        # YOUR CODE HERE
        pass

    def expand_query(self, query: str, n: int = 3) -> List[str]:
        """Generate query variations using LLM."""
        # YOUR CODE HERE
        pass

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 3
    ) -> List[str]:
        """Rerank documents using LLM scoring."""
        # YOUR CODE HERE
        pass

    def calculate_confidence(self, query: str, answer: str, context: str) -> float:
        """Calculate confidence score for the answer."""
        # YOUR CODE HERE
        pass

    def query(
        self,
        question: str,
        use_hybrid: bool = True,
        use_expansion: bool = True,
        use_reranking: bool = True
    ) -> RAGResult:
        """
        Full RAG pipeline with all advanced features.
        """
        # YOUR CODE HERE
        pass


# Test implementation
if __name__ == "__main__":
    rag = AdvancedRAGPipeline()

    # Add test documents
    documents = [
        "Python is a high-level programming language known for readability.",
        "JavaScript is essential for web development and runs in browsers.",
        "Machine learning is a subset of AI that learns from data.",
        "Neural networks are inspired by biological brain structures.",
        "RAG combines retrieval with generation for accurate answers.",
    ]
    sources = [f"doc_{i}.txt" for i in range(len(documents))]

    rag.add_documents(documents, sources)

    # Test hybrid search
    print("Test 1: Hybrid Search")
    results = rag.hybrid_search("What is Python?", k=3)
    print(f"  Found {len(results)} documents")
    assert len(results) > 0
    print("  ✓ PASSED\n")

    # Test query expansion
    print("Test 2: Query Expansion")
    expansions = rag.expand_query("Python programming")
    print(f"  Generated {len(expansions)} variations")
    assert len(expansions) >= 2
    print("  ✓ PASSED\n")

    # Test full pipeline
    print("Test 3: Full Pipeline")
    result = rag.query("What is machine learning?")
    print(f"  Answer: {result.answer[:100]}...")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Method: {result.retrieval_method}")
    assert result.confidence > 0
    print("  ✓ PASSED\n")

    print("All tests passed!")
```

### Grading Rubric
| Criteria | Points | Description |
|----------|--------|-------------|
| Vector search works | 20 | Returns relevant docs with scores |
| Keyword search works | 20 | BM25 implementation correct |
| Hybrid search combines both | 25 | Proper score fusion |
| Query expansion | 25 | LLM generates variations |
| Reranking | 25 | LLM-based reranking works |
| Confidence scoring | 20 | Reasonable confidence scores |
| Code quality | 15 | Clean, documented, tested |
| **Total** | **150** | |

---

# Module 4 Assignments

---

## Assignment 4.1: Multi-Agent Workflow
**Points:** 150 | **Time:** 120 minutes | **Difficulty:** Hard

### Objective
Build a multi-agent system using LangGraph for collaborative content creation.

### Requirements
1. Create 3 specialized agents:
   - Researcher: Gathers information
   - Writer: Creates content
   - Editor: Reviews and improves
2. Implement state management
3. Add conditional routing
4. Include iteration for quality improvement

### Starter Code
```python
"""
Assignment 4.1: Multi-Agent Content Creation Workflow
"""
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import TypedDict, List, Annotated
from dotenv import load_dotenv

load_dotenv()


class ContentState(TypedDict):
    """State shared between agents."""
    topic: str
    research: str
    draft: str
    feedback: str
    final_content: str
    quality_score: int
    iteration: int


def create_researcher_agent():
    """Create the research agent."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research specialist. Gather key facts and information."),
        ("human", "Research this topic thoroughly: {topic}")
    ])
    return prompt | llm


def create_writer_agent():
    """Create the writing agent."""
    # YOUR CODE HERE
    pass


def create_editor_agent():
    """Create the editing agent."""
    # YOUR CODE HERE
    pass


def researcher_node(state: ContentState) -> ContentState:
    """Research node - gathers information."""
    # YOUR CODE HERE
    pass


def writer_node(state: ContentState) -> ContentState:
    """Writer node - creates content."""
    # YOUR CODE HERE
    pass


def editor_node(state: ContentState) -> ContentState:
    """Editor node - reviews and scores content."""
    # YOUR CODE HERE
    pass


def should_revise(state: ContentState) -> str:
    """Decide if content needs revision."""
    # YOUR CODE HERE
    # Return "revise" if quality_score < 8 and iteration < 3
    # Return "finalize" otherwise
    pass


def build_workflow() -> StateGraph:
    """Build the multi-agent workflow graph."""
    workflow = StateGraph(ContentState)

    # YOUR CODE HERE:
    # 1. Add nodes for researcher, writer, editor
    # 2. Add edges between nodes
    # 3. Add conditional edge from editor (should_revise)
    # 4. Set entry point

    return workflow


def run_content_pipeline(topic: str) -> str:
    """Run the complete content creation pipeline."""
    workflow = build_workflow()
    app = workflow.compile()

    initial_state = {
        "topic": topic,
        "research": "",
        "draft": "",
        "feedback": "",
        "final_content": "",
        "quality_score": 0,
        "iteration": 0
    }

    # YOUR CODE HERE: Run the workflow and return final content
    pass


# Test implementation
if __name__ == "__main__":
    print("Test: Content Creation Pipeline")
    print("=" * 60)

    topic = "The benefits of learning Python programming"
    result = run_content_pipeline(topic)

    print(f"\nFinal Content:\n{result}")
    assert len(result) > 100
    print("\n✓ Test passed!")
```

### Grading Rubric
| Criteria | Points | Description |
|----------|--------|-------------|
| Researcher agent works | 25 | Gathers relevant information |
| Writer agent works | 25 | Creates coherent content |
| Editor agent works | 25 | Provides useful feedback |
| State management | 25 | State flows correctly |
| Conditional routing | 25 | Revision logic works |
| Quality improvement | 15 | Content improves with iterations |
| Code quality | 10 | Clean, documented |
| **Total** | **150** | |

---

# Final Capstone Project

---

## Capstone: Production RAG Application
**Points:** 300 | **Time:** 8+ hours | **Difficulty:** Expert

### Objective
Build a complete, production-ready RAG application with API, evaluation, and documentation.

### Requirements

#### Core Features (150 points)
1. Document ingestion (PDF, TXT, Web)
2. Advanced retrieval (hybrid search, reranking)
3. Answer generation with citations
4. Query routing based on question type

#### API Layer (50 points)
5. FastAPI REST endpoints
6. Proper error handling
7. Request/response validation

#### Quality & Testing (50 points)
8. Unit tests for core components
9. Integration tests for API
10. Evaluation metrics (precision, recall)

#### Documentation (50 points)
11. README with setup instructions
12. API documentation
13. Architecture diagram

### Deliverables
```
capstone_project/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app
│   ├── config.py        # Configuration
│   ├── models.py        # Pydantic models
│   ├── rag_service.py   # RAG logic
│   └── routes.py        # API routes
├── tests/
│   ├── test_rag.py
│   └── test_api.py
├── docs/
│   ├── API.md
│   └── ARCHITECTURE.md
├── requirements.txt
├── Dockerfile
└── README.md
```

### API Endpoints Required
```
POST /documents/upload    - Upload documents
POST /documents/ingest    - Ingest uploaded documents
POST /query               - Ask a question
GET  /health              - Health check
GET  /metrics             - Retrieval metrics
```

### Grading Rubric
| Criteria | Points |
|----------|--------|
| Document ingestion works | 40 |
| Retrieval is accurate | 40 |
| Generation includes citations | 30 |
| Query routing works | 40 |
| API endpoints functional | 30 |
| Error handling complete | 20 |
| Unit tests pass | 25 |
| Integration tests pass | 25 |
| README complete | 20 |
| API documented | 15 |
| Architecture diagram | 15 |
| **Total** | **300** |

---

## Submission Checklist

For each assignment, verify:

- [ ] All requirements implemented
- [ ] All tests pass
- [ ] Code is clean and documented
- [ ] Self-graded against rubric
- [ ] Understood concepts (not just copied)

---

## Getting Help

If stuck, try:
1. Re-read the relevant lesson
2. Check the error message carefully
3. Ask Claude Code with specific context
4. Review the solutions (after attempting)

**Good luck with your assignments!**
