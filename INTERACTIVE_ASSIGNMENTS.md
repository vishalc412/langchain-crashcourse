# Interactive Assignments & Hands-On Exercises
## Become an Expert Through Practice

This guide contains progressive hands-on assignments to help you master LangChain, LangGraph, and AI design patterns. Each assignment builds on previous knowledge and includes clear objectives, starter code, and expected outcomes.

---

## How to Use This Guide

1. **Work through assignments in order** - They build on each other
2. **Don't skip exercises** - Each one teaches important concepts
3. **Test your code** - Make sure everything works before moving on
4. **Challenge yourself** - Try the bonus challenges for deeper learning
5. **Ask for help** - Use Claude Code if you get stuck

---

## Module 1: LangChain Fundamentals

### Assignment 1.1: Your First LLM Call
**Difficulty**: Beginner | **Time**: 15-20 minutes

**Objective**: Set up your environment and make your first API call to GPT-4o-mini.

**Tasks**:
```
1. Create a new Python file: 01_first_llm_call.py
2. Set up environment variables for OPENAI_API_KEY
3. Initialize ChatOpenAI with gpt-4o-mini
4. Send a simple prompt and print the response
5. Add error handling for API failures
```

**Starter Code**:
```python
"""
Assignment 1.1: Your First LLM Call
Complete the TODOs to make your first LLM call.
"""

import os
from dotenv import load_dotenv

# TODO 1: Import ChatOpenAI from langchain_openai


# TODO 2: Load environment variables
load_dotenv()

# TODO 3: Initialize the LLM with model="gpt-4o-mini"


# TODO 4: Create a simple message and invoke the LLM


# TODO 5: Print the response


# TODO 6: Add try/except to handle potential errors


if __name__ == "__main__":
    # Run your code here
    pass
```

**Expected Output**:
```
Successfully connected to GPT-4o-mini!
Response: [LLM's response to your prompt]
```

**Self-Check Questions**:
- [ ] Does your code load environment variables correctly?
- [ ] Can you change the temperature parameter?
- [ ] What happens if you provide an invalid API key?

**Bonus Challenge**:
Create a function that takes any prompt and returns the response, with configurable temperature and max_tokens.

---

### Assignment 1.2: Prompt Templates
**Difficulty**: Beginner | **Time**: 20-30 minutes

**Objective**: Learn to create and use prompt templates for consistent, reusable prompts.

**Tasks**:
```
1. Create a prompt template with variables
2. Use the template to generate multiple outputs
3. Create a chain combining template and LLM
4. Test with different input variables
```

**Starter Code**:
```python
"""
Assignment 1.2: Prompt Templates
Create reusable prompt templates.
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# TODO 1: Create a prompt template for a product description generator
# Variables needed: {product_name}, {target_audience}, {key_features}


# TODO 2: Initialize the LLM


# TODO 3: Create a chain: template | llm


# TODO 4: Test with at least 3 different products


# TODO 5: Create a second template for a different use case
# (e.g., email writer, code reviewer, etc.)


if __name__ == "__main__":
    # Test your templates here
    pass
```

**Test Cases**:
```python
# Test Case 1
product1 = {
    "product_name": "Smart Water Bottle",
    "target_audience": "fitness enthusiasts",
    "key_features": "temperature tracking, hydration reminders, BPA-free"
}

# Test Case 2
product2 = {
    "product_name": "AI Writing Assistant",
    "target_audience": "content creators",
    "key_features": "grammar checking, tone adjustment, SEO optimization"
}

# Test Case 3 - Create your own!
```

**Self-Check Questions**:
- [ ] Can you add more variables to your template?
- [ ] How do you handle missing variables?
- [ ] What's the difference between ChatPromptTemplate and PromptTemplate?

---

### Assignment 1.3: Memory Systems
**Difficulty**: Intermediate | **Time**: 30-40 minutes

**Objective**: Implement conversation memory to create a chatbot that remembers context.

**Tasks**:
```
1. Create a chatbot with ConversationBufferMemory
2. Create the same chatbot with ConversationSummaryMemory
3. Compare memory usage between the two
4. Implement a multi-turn conversation
```

**Starter Code**:
```python
"""
Assignment 1.3: Memory Systems
Build a chatbot with conversation memory.
"""

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv

load_dotenv()

class ChatbotWithMemory:
    def __init__(self, memory_type: str = "buffer"):
        # TODO 1: Initialize LLM

        # TODO 2: Initialize memory based on memory_type
        # Options: "buffer" or "summary"

        # TODO 3: Create conversation chain
        pass

    def chat(self, message: str) -> str:
        # TODO 4: Send message and get response
        pass

    def get_memory(self) -> str:
        # TODO 5: Return current memory contents
        pass

    def clear_memory(self):
        # TODO 6: Clear the memory
        pass


def compare_memory_types():
    """Compare buffer vs summary memory."""
    # TODO 7: Create both types of chatbots

    # TODO 8: Send the same series of messages to both

    # TODO 9: Compare and print memory contents
    pass


if __name__ == "__main__":
    # Test conversation
    bot = ChatbotWithMemory(memory_type="buffer")

    conversation = [
        "Hi, my name is Alice.",
        "I'm learning about LangChain.",
        "What's my name?",  # Should remember "Alice"
        "What am I learning about?"  # Should remember "LangChain"
    ]

    for message in conversation:
        print(f"User: {message}")
        response = bot.chat(message)
        print(f"Bot: {response}\n")

    # Run comparison
    compare_memory_types()
```

**Expected Behavior**:
- Bot should remember user's name throughout conversation
- Bot should recall topics discussed
- Summary memory should condense long conversations

**Self-Check Questions**:
- [ ] What happens when conversation gets very long with buffer memory?
- [ ] When would you prefer summary memory over buffer memory?
- [ ] Can you implement entity memory?

---

## Module 2: RAG Foundations

### Assignment 2.1: Document Loading and Chunking
**Difficulty**: Intermediate | **Time**: 30-45 minutes

**Objective**: Load documents and implement intelligent chunking strategies.

**Tasks**:
```
1. Load a PDF document
2. Implement recursive text splitting
3. Add metadata to chunks
4. Visualize chunk distribution
5. Compare different chunk sizes
```

**Starter Code**:
```python
"""
Assignment 2.1: Document Loading and Chunking
Learn to process documents for RAG.
"""

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import matplotlib.pyplot as plt

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        # TODO 1: Initialize text splitter with parameters
        pass

    def load_document(self, file_path: str):
        # TODO 2: Load document based on file type
        pass

    def chunk_documents(self, documents) -> List:
        # TODO 3: Split documents into chunks
        pass

    def add_metadata(self, chunks) -> List:
        # TODO 4: Add metadata to each chunk
        # Include: chunk_index, word_count, has_code, source_page
        pass

    def visualize_chunks(self, chunks):
        # TODO 5: Create visualization of chunk sizes
        # Use matplotlib to create a histogram
        pass

    def compare_chunk_sizes(self, documents, sizes: List[int] = [500, 1000, 2000]):
        # TODO 6: Compare different chunk sizes
        # Return statistics for each size
        pass


if __name__ == "__main__":
    # Create sample text file for testing
    sample_text = """
    # Introduction to RAG

    Retrieval Augmented Generation (RAG) is a technique that combines
    the power of large language models with external knowledge sources.

    ## How RAG Works

    1. Document Processing: Documents are split into chunks
    2. Embedding: Chunks are converted to vectors
    3. Storage: Vectors are stored in a vector database
    4. Retrieval: Relevant chunks are retrieved for each query
    5. Generation: LLM generates response using retrieved context

    ## Benefits of RAG

    - Reduces hallucinations
    - Provides up-to-date information
    - Enables source attribution
    - More cost-effective than fine-tuning
    """

    # Save to file
    with open("sample_doc.txt", "w") as f:
        f.write(sample_text)

    # Test your processor
    processor = DocumentProcessor()
    docs = processor.load_document("sample_doc.txt")
    chunks = processor.chunk_documents(docs)

    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"Length: {len(chunk.page_content)} characters")
        print(f"Preview: {chunk.page_content[:100]}...")
```

---

### Assignment 2.2: Build Your First RAG System
**Difficulty**: Intermediate | **Time**: 45-60 minutes

**Objective**: Create a complete RAG pipeline from scratch.

**Tasks**:
```
1. Create document processor
2. Set up vector store with embeddings
3. Implement retrieval
4. Create answer generation
5. Add source citations
```

**Starter Code**:
```python
"""
Assignment 2.2: Build Your First RAG System
Create a complete RAG pipeline.
"""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

class SimpleRAG:
    def __init__(self, persist_directory: str = "./chroma_db"):
        # TODO 1: Initialize embeddings, LLM, and vectorstore path
        self.embeddings = None
        self.llm = None
        self.vectorstore = None
        self.persist_directory = persist_directory

    def ingest_documents(self, texts: List[str], metadata: List[Dict] = None):
        """Ingest documents into the vector store."""
        # TODO 2: Split texts into chunks

        # TODO 3: Create vector store from documents

        # TODO 4: Persist the vector store
        pass

    def retrieve(self, query: str, k: int = 3) -> List:
        """Retrieve relevant documents."""
        # TODO 5: Perform similarity search
        pass

    def generate_answer(self, query: str, context_docs: List) -> Dict:
        """Generate answer from retrieved context."""
        # TODO 6: Create context string from documents

        # TODO 7: Create prompt with context and query

        # TODO 8: Generate answer with citations

        # TODO 9: Return answer with sources
        pass

    def query(self, question: str) -> Dict:
        """Complete RAG pipeline."""
        # TODO 10: Combine retrieve and generate
        pass


def test_rag_system():
    """Test the RAG system with sample data."""

    # Sample knowledge base
    documents = [
        "LangChain is a framework for developing applications powered by language models. It provides tools for prompt management, chains, and agents.",
        "RAG (Retrieval Augmented Generation) combines LLMs with external knowledge. It retrieves relevant documents and uses them to generate informed responses.",
        "Vector databases store embeddings for similarity search. Popular options include Chroma, Pinecone, and FAISS.",
        "Embeddings are numerical representations of text. OpenAI's text-embedding-3-small is commonly used for its balance of quality and cost.",
        "Chain-of-thought prompting helps LLMs reason through complex problems step by step, improving accuracy on reasoning tasks.",
        "LangGraph extends LangChain with graph-based workflows, enabling complex multi-agent systems and cyclic reasoning patterns."
    ]

    # TODO 11: Create RAG system and test with questions
    rag = SimpleRAG()
    rag.ingest_documents(documents)

    test_questions = [
        "What is LangChain?",
        "How does RAG work?",
        "What are some vector database options?",
        "What is chain-of-thought prompting?"
    ]

    for question in test_questions:
        print(f"\n{'='*50}")
        print(f"Question: {question}")
        result = rag.query(question)
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")


if __name__ == "__main__":
    test_rag_system()
```

**Expected Output**:
```
==================================================
Question: What is LangChain?
Answer: LangChain is a framework for developing applications powered by language models...
Sources: [Source 1: "LangChain is a framework..."]
```

**Bonus Challenges**:
1. Add MMR (Maximum Marginal Relevance) retrieval
2. Implement query expansion before retrieval
3. Add confidence scores to answers

---

### Assignment 2.3: Advanced Retrieval Techniques
**Difficulty**: Advanced | **Time**: 60-90 minutes

**Objective**: Implement and compare multiple retrieval strategies.

**Tasks**:
```
1. Implement hybrid search (vector + BM25)
2. Add query expansion
3. Implement reranking
4. Create evaluation metrics
5. Compare all approaches
```

**Starter Code**:
```python
"""
Assignment 2.3: Advanced Retrieval Techniques
Compare different retrieval strategies.
"""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class RetrievalResult:
    """Store retrieval results with metrics."""
    documents: List
    scores: List[float]
    method: str
    latency_ms: float

class AdvancedRetriever:
    def __init__(self, documents: List[str]):
        # TODO 1: Initialize all components
        self.documents = documents
        self.embeddings = None
        self.vectorstore = None
        self.bm25 = None
        self.llm = None
        self._setup()

    def _setup(self):
        """Set up vector store and BM25 index."""
        # TODO 2: Create vector store

        # TODO 3: Create BM25 index
        pass

    def vector_search(self, query: str, k: int = 5) -> RetrievalResult:
        """Standard vector similarity search."""
        # TODO 4: Implement vector search
        pass

    def bm25_search(self, query: str, k: int = 5) -> RetrievalResult:
        """BM25 keyword search."""
        # TODO 5: Implement BM25 search
        pass

    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5) -> RetrievalResult:
        """Hybrid search combining vector and BM25."""
        # TODO 6: Implement hybrid search
        # alpha: weight for vector search (1-alpha for BM25)
        pass

    def expand_query(self, query: str, n_expansions: int = 3) -> List[str]:
        """Generate query variations."""
        # TODO 7: Use LLM to generate query variations
        pass

    def multi_query_search(self, query: str, k: int = 5) -> RetrievalResult:
        """Search with multiple query variations."""
        # TODO 8: Expand query and search with all variations
        pass

    def rerank_results(self, query: str, documents: List, top_k: int = 3) -> List:
        """Rerank documents using LLM."""
        # TODO 9: Use LLM to score and rerank documents
        pass


class RetrievalEvaluator:
    """Evaluate retrieval quality."""

    def __init__(self, ground_truth: Dict[str, List[int]]):
        """
        ground_truth: Dict mapping queries to relevant document indices
        """
        self.ground_truth = ground_truth

    def precision_at_k(self, retrieved: List[int], relevant: List[int], k: int) -> float:
        """Calculate precision@k."""
        # TODO 10: Implement precision@k
        pass

    def recall_at_k(self, retrieved: List[int], relevant: List[int], k: int) -> float:
        """Calculate recall@k."""
        # TODO 11: Implement recall@k
        pass

    def mrr(self, retrieved: List[int], relevant: List[int]) -> float:
        """Calculate Mean Reciprocal Rank."""
        # TODO 12: Implement MRR
        pass

    def evaluate_retriever(self, retriever, method: str) -> Dict:
        """Evaluate a retrieval method."""
        # TODO 13: Run evaluation on all test queries
        pass


def run_comparison():
    """Compare all retrieval methods."""

    # Test documents
    documents = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Deep learning uses neural networks with multiple layers to learn complex patterns.",
        "Natural language processing (NLP) focuses on the interaction between computers and human language.",
        "Computer vision enables machines to interpret and understand visual information from the world.",
        "Reinforcement learning trains agents through rewards and penalties in an environment.",
        "Transfer learning allows models trained on one task to be applied to different but related tasks.",
        "Supervised learning uses labeled data to train models to predict outcomes.",
        "Unsupervised learning finds patterns in data without explicit labels.",
    ]

    # Ground truth: which documents are relevant for each query
    ground_truth = {
        "What is machine learning?": [0, 6, 7],
        "How does deep learning work?": [1, 0],
        "What is NLP used for?": [2],
        "Explain transfer learning": [5, 0],
    }

    # TODO 14: Create retriever and evaluator
    # TODO 15: Compare all methods and print results
    pass


if __name__ == "__main__":
    run_comparison()
```

---

## Module 3: LangGraph & Agents

### Assignment 3.1: Build a ReAct Agent
**Difficulty**: Intermediate | **Time**: 45-60 minutes

**Objective**: Create an agent that reasons and acts using tools.

**Tasks**:
```
1. Define custom tools
2. Create ReAct prompt
3. Build agent with tool execution
4. Test with multi-step problems
```

**Starter Code**:
```python
"""
Assignment 3.1: Build a ReAct Agent
Create an agent with reasoning and tools.
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from typing import List, Dict
import re
import json

class ReActAgent:
    """ReAct agent with custom tools."""

    def __init__(self, tools: List[Tool], max_iterations: int = 5):
        # TODO 1: Initialize LLM and tools
        self.llm = None
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations

    def _create_prompt(self, task: str, scratchpad: str) -> str:
        """Create ReAct prompt."""
        # TODO 2: Create prompt template
        template = """
You are a helpful AI assistant that solves problems step by step.

Available tools:
{tools}

Use this format:
Thought: [Your reasoning]
Action: [Tool name]
Action Input: [Input for the tool]
Observation: [Result from tool]
... (repeat until solved)
Thought: I have the final answer
Final Answer: [Your answer]

Task: {task}

{scratchpad}
"""
        pass

    def _parse_action(self, response: str) -> tuple:
        """Parse action and action input from response."""
        # TODO 3: Extract action and action_input using regex
        pass

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Execute a tool and return result."""
        # TODO 4: Execute the appropriate tool
        pass

    def run(self, task: str) -> str:
        """Run the ReAct loop."""
        # TODO 5: Implement the main ReAct loop
        scratchpad = ""

        for i in range(self.max_iterations):
            # Get LLM response
            # Parse action
            # Execute tool
            # Add observation to scratchpad
            # Check for final answer
            pass


def create_tools() -> List[Tool]:
    """Create a set of useful tools."""

    # TODO 6: Implement calculator tool
    def calculator(expression: str) -> str:
        pass

    # TODO 7: Implement weather tool (mock)
    def get_weather(city: str) -> str:
        pass

    # TODO 8: Implement search tool (mock)
    def search(query: str) -> str:
        pass

    return [
        Tool(name="Calculator", func=calculator, description="..."),
        Tool(name="Weather", func=get_weather, description="..."),
        Tool(name="Search", func=search, description="..."),
    ]


def test_agent():
    """Test the ReAct agent."""
    tools = create_tools()
    agent = ReActAgent(tools)

    # Test problems
    problems = [
        "What is 15% of 230?",
        "If I have $500 and spend 23% on food, how much do I have left?",
        "What's the weather in Tokyo and convert the temperature to Fahrenheit?",
    ]

    for problem in problems:
        print(f"\n{'='*50}")
        print(f"Problem: {problem}")
        result = agent.run(problem)
        print(f"Answer: {result}")


if __name__ == "__main__":
    test_agent()
```

---

### Assignment 3.2: Multi-Agent System with LangGraph
**Difficulty**: Advanced | **Time**: 90-120 minutes

**Objective**: Build a collaborative multi-agent system for content creation.

**Tasks**:
```
1. Define specialized agents (Researcher, Writer, Editor)
2. Create agent state management
3. Build workflow graph
4. Implement agent communication
5. Add conditional routing
```

**Starter Code**:
```python
"""
Assignment 3.2: Multi-Agent System
Build a collaborative content creation system.
"""

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import TypedDict, List, Annotated
import operator

# Define state
class ContentState(TypedDict):
    topic: str
    research: str
    outline: str
    draft: str
    feedback: List[str]
    final_content: str
    iteration: int
    status: str


class ResearcherAgent:
    """Agent that researches the topic."""

    def __init__(self):
        # TODO 1: Initialize LLM with appropriate prompt
        pass

    def __call__(self, state: ContentState) -> ContentState:
        # TODO 2: Research the topic and update state
        pass


class WriterAgent:
    """Agent that writes content based on research."""

    def __init__(self):
        # TODO 3: Initialize LLM with writing prompt
        pass

    def __call__(self, state: ContentState) -> ContentState:
        # TODO 4: Write draft based on research
        pass


class EditorAgent:
    """Agent that reviews and provides feedback."""

    def __init__(self):
        # TODO 5: Initialize LLM with editing prompt
        pass

    def __call__(self, state: ContentState) -> ContentState:
        # TODO 6: Review draft and provide feedback
        # Should include quality score
        pass


def should_continue(state: ContentState) -> str:
    """Determine if more iterations are needed."""
    # TODO 7: Check quality score and iteration count
    # Return "revise" or "finalize"
    pass


def create_workflow() -> StateGraph:
    """Create the multi-agent workflow."""

    # TODO 8: Create workflow graph
    workflow = StateGraph(ContentState)

    # TODO 9: Add nodes for each agent

    # TODO 10: Add edges with conditional routing

    # TODO 11: Set entry point and compile

    return workflow


def visualize_workflow(workflow):
    """Visualize the workflow (optional)."""
    # TODO 12: Create visualization using graphviz
    pass


def run_content_pipeline(topic: str):
    """Run the complete content creation pipeline."""
    workflow = create_workflow()
    app = workflow.compile()

    # Initial state
    initial_state = {
        "topic": topic,
        "research": "",
        "outline": "",
        "draft": "",
        "feedback": [],
        "final_content": "",
        "iteration": 0,
        "status": "started"
    }

    # TODO 13: Run workflow and track progress

    return final_state


if __name__ == "__main__":
    topic = "The Future of Artificial Intelligence in Healthcare"
    result = run_content_pipeline(topic)

    print("\n" + "="*50)
    print("FINAL CONTENT")
    print("="*50)
    print(result["final_content"])
```

---

## Module 4: Design Patterns

### Assignment 4.1: Implement All RAG Variations
**Difficulty**: Advanced | **Time**: 120+ minutes

**Objective**: Implement and compare all RAG patterns.

**Tasks**:
```
1. Basic RAG
2. RAG Fusion
3. Self-RAG
4. Corrective RAG
5. Adaptive RAG
6. Create comparison framework
```

**Architecture Reference**:
```
┌─────────────────────────────────────────────────────────────┐
│                    RAG PATTERN COMPARISON                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  BASIC RAG                                                   │
│  Query → Retrieve → Generate                                 │
│  Simplest approach, good baseline                           │
│                                                              │
│  RAG FUSION                                                  │
│  Query → Expand → Multi-Retrieve → Fuse → Generate          │
│  Better recall through query diversity                      │
│                                                              │
│  SELF-RAG                                                    │
│  Query → Retrieve → Generate → Critique → (Repeat)          │
│  Self-correction for higher quality                         │
│                                                              │
│  CORRECTIVE RAG                                              │
│  Query → Retrieve → Grade → Filter/Fallback → Generate      │
│  Quality gates on retrieved content                         │
│                                                              │
│  ADAPTIVE RAG                                                │
│  Query → Classify → Route to Best Strategy → Generate       │
│  Dynamic strategy selection                                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Starter Code**:
```python
"""
Assignment 4.1: RAG Pattern Implementations
Implement and compare all RAG variations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dataclasses import dataclass
import time

@dataclass
class RAGResult:
    answer: str
    sources: List[Dict]
    latency_ms: float
    pattern: str
    metadata: Dict[str, Any]


class BaseRAG(ABC):
    """Base class for all RAG implementations."""

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    @abstractmethod
    def query(self, question: str) -> RAGResult:
        pass


class BasicRAG(BaseRAG):
    """Standard RAG implementation."""

    def query(self, question: str) -> RAGResult:
        # TODO 1: Implement basic RAG
        pass


class RAGFusion(BaseRAG):
    """RAG with query expansion and fusion."""

    def query(self, question: str) -> RAGResult:
        # TODO 2: Implement RAG Fusion
        pass

    def _expand_query(self, query: str, n: int = 3) -> List[str]:
        # TODO 3: Generate query variations
        pass

    def _reciprocal_rank_fusion(self, results: List[List]) -> List:
        # TODO 4: Implement RRF scoring
        pass


class SelfRAG(BaseRAG):
    """RAG with self-reflection and correction."""

    def query(self, question: str) -> RAGResult:
        # TODO 5: Implement Self-RAG
        pass

    def _critique(self, question: str, answer: str, context: str) -> Dict:
        # TODO 6: Self-critique the answer
        pass

    def _should_retry(self, critique: Dict) -> bool:
        # TODO 7: Determine if retry needed
        pass


class CorrectiveRAG(BaseRAG):
    """RAG with retrieval quality assessment."""

    def query(self, question: str) -> RAGResult:
        # TODO 8: Implement Corrective RAG
        pass

    def _grade_document(self, question: str, document: str) -> str:
        # TODO 9: Grade document relevance
        pass

    def _fallback_search(self, question: str) -> List:
        # TODO 10: Implement fallback strategy
        pass


class AdaptiveRAG(BaseRAG):
    """RAG with dynamic strategy selection."""

    def __init__(self, vectorstore):
        super().__init__(vectorstore)
        self.strategies = {
            "simple": BasicRAG(vectorstore),
            "fusion": RAGFusion(vectorstore),
            "self": SelfRAG(vectorstore),
            "corrective": CorrectiveRAG(vectorstore),
        }

    def query(self, question: str) -> RAGResult:
        # TODO 11: Implement Adaptive RAG
        pass

    def _classify_query(self, question: str) -> str:
        # TODO 12: Classify query to select strategy
        pass


class RAGComparison:
    """Compare different RAG implementations."""

    def __init__(self, vectorstore, test_cases: List[Dict]):
        self.vectorstore = vectorstore
        self.test_cases = test_cases
        self.patterns = {
            "basic": BasicRAG(vectorstore),
            "fusion": RAGFusion(vectorstore),
            "self": SelfRAG(vectorstore),
            "corrective": CorrectiveRAG(vectorstore),
            "adaptive": AdaptiveRAG(vectorstore),
        }

    def run_comparison(self) -> Dict:
        # TODO 13: Run all patterns on all test cases
        pass

    def evaluate_results(self, results: Dict) -> Dict:
        # TODO 14: Calculate metrics for each pattern
        # Metrics: latency, answer quality, source relevance
        pass

    def print_report(self, results: Dict, metrics: Dict):
        # TODO 15: Print formatted comparison report
        pass


if __name__ == "__main__":
    # Create test data
    documents = [
        # Add your test documents here
    ]

    test_cases = [
        {"question": "...", "expected_topics": [...]},
        # Add more test cases
    ]

    # Run comparison
    # comparison = RAGComparison(vectorstore, test_cases)
    # results = comparison.run_comparison()
    # comparison.print_report(results, comparison.evaluate_results(results))
```

---

## Final Project: Production RAG System

### Capstone Project
**Difficulty**: Expert | **Time**: 8-12 hours

**Objective**: Build a complete, production-ready RAG system with all advanced features.

**Requirements**:
```
1. Document ingestion pipeline (PDF, TXT, Web)
2. Hybrid retrieval (vector + BM25)
3. Query transformation and routing
4. Multiple RAG strategies
5. Answer generation with citations
6. Evaluation metrics
7. REST API with FastAPI
8. Monitoring and logging
9. Unit and integration tests
10. Documentation
```

**Deliverables**:
```
project/
├── app/
│   ├── main.py           # FastAPI application
│   ├── config.py         # Configuration
│   ├── models.py         # Pydantic models
│   ├── ingestion.py      # Document processing
│   ├── retrieval.py      # Retrieval engine
│   ├── generation.py     # Answer generation
│   └── evaluation.py     # Metrics
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_api.py
├── docs/
│   ├── API.md
│   └── ARCHITECTURE.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

**Evaluation Criteria**:
- [ ] All components functional
- [ ] Tests passing
- [ ] API documented
- [ ] Code quality (linting, typing)
- [ ] Performance benchmarks included
- [ ] Error handling implemented
- [ ] Logging configured

---

## Self-Assessment Rubric

Rate yourself on each skill (1-5):

### LangChain Fundamentals
- [ ] LLM initialization and configuration
- [ ] Prompt templates
- [ ] Chains and composition
- [ ] Memory systems
- [ ] Output parsers

### RAG Skills
- [ ] Document processing
- [ ] Embeddings and vector stores
- [ ] Basic retrieval
- [ ] Advanced retrieval (hybrid, reranking)
- [ ] Answer generation with citations

### LangGraph
- [ ] Graph construction
- [ ] State management
- [ ] Conditional routing
- [ ] Multi-agent systems

### Design Patterns
- [ ] Chain-of-Thought
- [ ] ReAct
- [ ] RAG variations
- [ ] Evaluation patterns

### Production Skills
- [ ] API development
- [ ] Testing
- [ ] Monitoring
- [ ] Documentation

---

## Getting Help

If you're stuck on any assignment:

```
Ask Claude Code:
"I'm working on Assignment [X.X] and stuck on [specific task].
Here's my current code: [code]
The error/issue is: [description]
Please help me understand and solve this."
```

---

**Good luck with your learning journey!**

Remember: The goal is not just to complete these assignments, but to deeply understand the concepts. Take your time, experiment, and ask questions!
