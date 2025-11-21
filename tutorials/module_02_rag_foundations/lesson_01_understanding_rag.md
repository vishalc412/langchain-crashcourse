# Module 2, Lesson 1: Understanding RAG Architecture

## Lesson Overview
| Duration | Difficulty | Prerequisites |
|----------|------------|---------------|
| 60 minutes | Intermediate | Module 1 completed |

## Learning Objectives
By the end of this lesson, you will be able to:
- [ ] Explain what RAG is and why it's important
- [ ] Describe each component of a RAG system
- [ ] Understand how documents become searchable
- [ ] Identify when to use RAG vs fine-tuning

---

## Part 1: LEARN IT - The RAG Revolution

### What Problem Does RAG Solve?

**The LLM Knowledge Problem:**

```
┌─────────────────────────────────────────────────────────────┐
│              WITHOUT RAG: LLM LIMITATIONS                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  You: "What were our Q3 2024 sales numbers?"                │
│                                                              │
│  LLM: "I don't have access to your company's sales          │
│        data. My knowledge was cut off in [date]             │
│        and I don't have access to private documents."       │
│                                                              │
│  Problems:                                                   │
│  ✗ LLM doesn't know YOUR data                               │
│  ✗ Knowledge cutoff means outdated info                     │
│  ✗ Can't access private/internal documents                  │
│  ✗ May hallucinate answers it doesn't know                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                WITH RAG: GROUNDED ANSWERS                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  You: "What were our Q3 2024 sales numbers?"                │
│                                                              │
│  RAG System:                                                 │
│  1. Searches your company documents                          │
│  2. Finds Q3 2024 Sales Report                              │
│  3. Extracts relevant sections                              │
│  4. Sends to LLM with context                               │
│                                                              │
│  LLM: "According to the Q3 2024 Sales Report,               │
│        total revenue was $2.4M, representing a              │
│        15% increase from Q2. [Source: Q3_Report.pdf]"       │
│                                                              │
│  Benefits:                                                   │
│  ✓ Answers based on YOUR actual data                        │
│  ✓ Always up-to-date (as your docs are updated)            │
│  ✓ Can cite sources                                         │
│  ✓ Reduced hallucinations                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### RAG = Retrieval + Augmented + Generation

```
R - RETRIEVAL:    Find relevant information from your documents
A - AUGMENTED:    Add that information to the LLM's context
G - GENERATION:   LLM generates an answer using that context
```

### The Complete RAG Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    COMPLETE RAG PIPELINE                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  PHASE 1: INDEXING (One-time setup)                         │
│  ════════════════════════════════════                        │
│                                                              │
│   Documents          Split              Embed                │
│   [PDF, TXT, etc.] → [Chunks] →        [Vectors]            │
│        │                │                  │                 │
│        │                │                  ▼                 │
│        │                │           Vector Database          │
│        │                │           [Chroma/FAISS]           │
│        │                │                  │                 │
│        ▼                ▼                  │                 │
│   ┌─────────┐    ┌─────────────┐          │                 │
│   │ Load    │    │ "RAG is a   │          │                 │
│   │ PDF     │ →  │  technique  │ →  [0.2, 0.5, 0.1, ...]   │
│   │         │    │  that..."   │          │                 │
│   └─────────┘    └─────────────┘          │                 │
│                        │                   │                 │
│                        │                   ▼                 │
│                        │            Stored for later         │
│                                                              │
│  PHASE 2: QUERYING (Every user question)                    │
│  ════════════════════════════════════════                    │
│                                                              │
│   User Question                                              │
│        │                                                     │
│        ▼                                                     │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ "What is RAG?" → [0.3, 0.4, 0.2, ...]              │   │
│   │      (Embed the question)                           │   │
│   └─────────────────────────────────────────────────────┘   │
│        │                                                     │
│        ▼                                                     │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ Search Vector DB for similar embeddings             │   │
│   │ Find top 3-5 most relevant chunks                   │   │
│   └─────────────────────────────────────────────────────┘   │
│        │                                                     │
│        ▼                                                     │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ PROMPT:                                             │   │
│   │ "Using this context: [retrieved chunks]            │   │
│   │  Answer this question: What is RAG?"               │   │
│   └─────────────────────────────────────────────────────┘   │
│        │                                                     │
│        ▼                                                     │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ LLM generates answer based on context               │   │
│   │ "RAG is a technique that combines..."              │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Understanding Embeddings

**What is an Embedding?**

An embedding is a list of numbers that represents the meaning of text. Similar texts have similar numbers.

```
┌─────────────────────────────────────────────────────────────┐
│                    HOW EMBEDDINGS WORK                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Text                          Embedding (simplified)        │
│  ────                          ────────────────────         │
│  "I love dogs"          →      [0.8, 0.2, 0.9, 0.1]        │
│  "I adore puppies"      →      [0.7, 0.3, 0.85, 0.15]      │
│  "The sky is blue"      →      [0.1, 0.9, 0.2, 0.8]        │
│                                                              │
│  Notice: "dogs" and "puppies" sentences have SIMILAR        │
│          numbers because they have SIMILAR meanings!        │
│                                                              │
│  The "sky" sentence has DIFFERENT numbers because           │
│  it has a DIFFERENT meaning.                                │
│                                                              │
│  ┌───────────────────────────────────────────────────┐     │
│  │  Similarity Calculation:                          │     │
│  │  "I love dogs" ↔ "I adore puppies" = 0.95 (high) │     │
│  │  "I love dogs" ↔ "The sky is blue" = 0.12 (low)  │     │
│  └───────────────────────────────────────────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### RAG vs Fine-Tuning: When to Use Each

| Factor | RAG | Fine-Tuning |
|--------|-----|-------------|
| **Setup Time** | Hours | Days/Weeks |
| **Cost** | Lower | Higher |
| **Data Updates** | Easy (just add docs) | Requires retraining |
| **Data Privacy** | Data stays local | Data used in training |
| **Best For** | Q&A, search, current info | Style, format, specialized tasks |
| **Hallucination** | Lower (grounded in docs) | Can still occur |

**Rule of Thumb:**
- Use **RAG** when you need to answer questions about specific documents
- Use **Fine-tuning** when you need to change HOW the model responds

---

## Part 2: SEE IT - Instructor Demo

### Demo 1: Manual RAG (Understanding the Steps)

```python
# File: demos/01_manual_rag.py
"""
INSTRUCTOR DEMO: Manual RAG process
Understanding each step before using LangChain helpers
"""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import numpy as np

load_dotenv()

print("MANUAL RAG DEMONSTRATION")
print("=" * 60)

# Our "knowledge base" - in reality, this would come from documents
documents = [
    "Python was created by Guido van Rossum and released in 1991.",
    "JavaScript is the most popular language for web development.",
    "RAG stands for Retrieval Augmented Generation.",
    "Machine learning is a subset of artificial intelligence.",
    "Python is known for its simple and readable syntax.",
]

# STEP 1: Create embeddings for all documents
print("\n📚 STEP 1: Creating embeddings for documents...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

doc_embeddings = []
for i, doc in enumerate(documents):
    embedding = embeddings.embed_query(doc)
    doc_embeddings.append(embedding)
    print(f"  Doc {i+1}: {doc[:40]}... → [{embedding[0]:.3f}, {embedding[1]:.3f}, ...]")

# STEP 2: Embed the user's question
print("\n❓ STEP 2: Embedding user question...")
question = "Who created Python?"
question_embedding = embeddings.embed_query(question)
print(f"  Question: '{question}'")
print(f"  Embedding: [{question_embedding[0]:.3f}, {question_embedding[1]:.3f}, ...]")

# STEP 3: Find most similar documents
print("\n🔍 STEP 3: Finding similar documents...")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarities = []
for i, doc_emb in enumerate(doc_embeddings):
    sim = cosine_similarity(question_embedding, doc_emb)
    similarities.append((i, sim, documents[i]))
    print(f"  Doc {i+1}: similarity = {sim:.3f}")

# Sort by similarity and get top 2
similarities.sort(key=lambda x: x[1], reverse=True)
top_docs = similarities[:2]

print("\n✅ Top relevant documents:")
for idx, sim, doc in top_docs:
    print(f"  [{sim:.3f}] {doc}")

# STEP 4: Generate answer with context
print("\n🤖 STEP 4: Generating answer with context...")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

context = "\n".join([doc for _, _, doc in top_docs])

prompt = ChatPromptTemplate.from_template("""
Answer the question based ONLY on the following context:

Context:
{context}

Question: {question}

Answer:
""")

chain = prompt | llm
response = chain.invoke({"context": context, "question": question})

print(f"\n📝 FINAL ANSWER:")
print(f"  {response.content}")
```

### Demo 2: LangChain RAG (Simplified)

```python
# File: demos/02_langchain_rag.py
"""
INSTRUCTOR DEMO: RAG with LangChain helpers
Same result, much less code!
"""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

print("LANGCHAIN RAG DEMONSTRATION")
print("=" * 60)

# Our knowledge base
documents = [
    Document(page_content="Python was created by Guido van Rossum and released in 1991."),
    Document(page_content="JavaScript is the most popular language for web development."),
    Document(page_content="RAG stands for Retrieval Augmented Generation."),
    Document(page_content="Machine learning is a subset of artificial intelligence."),
    Document(page_content="Python is known for its simple and readable syntax."),
]

# STEP 1 & 2: Create vector store (handles embedding automatically!)
print("\n📚 Creating vector store...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="demo"
)
print("  ✓ Vector store created!")

# STEP 3: Search (one line!)
print("\n🔍 Searching for relevant documents...")
question = "Who created Python?"
results = vectorstore.similarity_search(question, k=2)

print(f"  Question: '{question}'")
print(f"  Found {len(results)} relevant documents:")
for i, doc in enumerate(results):
    print(f"    {i+1}. {doc.page_content}")

# STEP 4: Generate answer
print("\n🤖 Generating answer...")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template("""
Answer based on this context:
{context}

Question: {question}

Answer:
""")

context = "\n".join([doc.page_content for doc in results])
chain = prompt | llm
response = chain.invoke({"context": context, "question": question})

print(f"\n📝 ANSWER: {response.content}")
```

---

## Part 3: DO IT - Guided Practice

### Exercise 1: Create Your First Vector Store (15 minutes)

```python
# File: exercises/ex1_vector_store.py
"""
EXERCISE 1: Create a vector store from scratch
"""

from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

# TODO 1: Create these documents about your favorite topic
# (or use these programming facts)
documents = [
    Document(
        page_content="___",  # Add a fact
        metadata={"source": "fact1", "topic": "programming"}
    ),
    Document(
        page_content="___",  # Add another fact
        metadata={"source": "fact2", "topic": "programming"}
    ),
    # Add at least 3 more documents
]

# TODO 2: Create embeddings
embeddings = OpenAIEmbeddings(model="___")

# TODO 3: Create vector store
vectorstore = Chroma.from_documents(
    documents=___,
    embedding=___
)

# TODO 4: Test with a search
query = "___"  # Ask a question about your documents
results = vectorstore.similarity_search(___, k=___)

# TODO 5: Print results
print("Search Results:")
for i, doc in enumerate(results):
    print(f"{i+1}. {doc.page_content}")
    print(f"   Metadata: {doc.metadata}")
```

### Exercise 2: Build a Simple Q&A System (20 minutes)

```python
# File: exercises/ex2_simple_qa.py
"""
EXERCISE 2: Build a complete Q&A system
"""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

class SimpleQA:
    def __init__(self, documents: list):
        """Initialize the Q&A system with documents."""
        # TODO 1: Store embeddings model
        self.embeddings = ___

        # TODO 2: Create vector store from documents
        self.vectorstore = ___

        # TODO 3: Initialize LLM
        self.llm = ___

        # TODO 4: Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer questions based only on the provided context. "
                      "If you don't know, say 'I don't have that information.'"),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])

    def ask(self, question: str, k: int = 3) -> str:
        """Ask a question and get an answer."""
        # TODO 5: Search for relevant documents

        # TODO 6: Format context

        # TODO 7: Generate and return answer
        pass


# Test your Q&A system
documents = [
    Document(page_content="LangChain is a framework for building LLM applications."),
    Document(page_content="Vector databases store embeddings for similarity search."),
    Document(page_content="RAG combines retrieval with generation for better answers."),
    Document(page_content="Embeddings are numerical representations of text meaning."),
    Document(page_content="Chroma is a popular open-source vector database."),
]

qa = SimpleQA(documents)

# Test questions
questions = [
    "What is LangChain?",
    "What are embeddings?",
    "How does RAG work?",
]

for q in questions:
    print(f"\nQ: {q}")
    print(f"A: {qa.ask(q)}")
```

---

## Part 4: PROVE IT - Lesson Assignment

### Assignment 2.1: Build a Document Q&A System

**Objective:** Create a Q&A system that can answer questions about a set of documents with source citations.

**Requirements:**
1. Accept multiple documents as input
2. Allow users to ask questions
3. Return answers WITH source citations
4. Handle "I don't know" cases gracefully

**Starter Code:**
```python
# File: assignments/assignment_2_1.py
"""
ASSIGNMENT 2.1: Document Q&A with Citations

Build a system that answers questions and cites sources.
"""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()


class DocumentQA:
    """Q&A system with source citations."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.vectorstore = None
        self.documents = []

    def add_documents(self, texts: List[str], sources: List[str]) -> int:
        """
        Add documents to the knowledge base.

        Args:
            texts: List of document contents
            sources: List of source names (e.g., filenames)

        Returns:
            Number of documents added
        """
        # YOUR CODE HERE
        pass

    def ask(self, question: str) -> Dict:
        """
        Ask a question and get answer with citations.

        Args:
            question: The user's question

        Returns:
            Dict with 'answer' and 'sources' keys
        """
        # YOUR CODE HERE
        # Should return: {"answer": "...", "sources": ["source1", "source2"]}
        pass


def test_document_qa():
    """Test the DocumentQA system."""

    # Create test documents
    texts = [
        "The Python programming language was created by Guido van Rossum. "
        "Development started in 1989 and the first version was released in 1991.",

        "Python's design philosophy emphasizes code readability with notable "
        "use of significant whitespace. Its language constructs aim to help "
        "programmers write clear, logical code.",

        "Python supports multiple programming paradigms, including structured, "
        "object-oriented, and functional programming.",

        "The name 'Python' comes from Monty Python's Flying Circus, not the snake. "
        "Guido van Rossum was a fan of the comedy group.",
    ]

    sources = [
        "python_history.txt",
        "python_design.txt",
        "python_paradigms.txt",
        "python_name_origin.txt",
    ]

    # Initialize and add documents
    qa = DocumentQA()
    num_added = qa.add_documents(texts, sources)
    print(f"Added {num_added} documents\n")

    # Test questions
    test_questions = [
        "Who created Python?",
        "Why is Python called Python?",
        "What programming paradigms does Python support?",
        "What is the capital of France?",  # Should say "I don't know"
    ]

    for question in test_questions:
        print("=" * 60)
        print(f"Q: {question}")
        result = qa.ask(question)
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print()


if __name__ == "__main__":
    test_document_qa()
```

**Expected Output:**
```
Added 4 documents

============================================================
Q: Who created Python?
A: Python was created by Guido van Rossum. Development started in 1989
   and the first version was released in 1991.
Sources: ['python_history.txt', 'python_name_origin.txt']

============================================================
Q: Why is Python called Python?
A: The name 'Python' comes from Monty Python's Flying Circus, not the
   snake. Guido van Rossum was a fan of the comedy group.
Sources: ['python_name_origin.txt']

============================================================
Q: What programming paradigms does Python support?
A: Python supports multiple programming paradigms, including structured,
   object-oriented, and functional programming.
Sources: ['python_paradigms.txt']

============================================================
Q: What is the capital of France?
A: I don't have information about the capital of France in my knowledge base.
Sources: []
```

**Grading Rubric:**
| Criteria | Points |
|----------|--------|
| Documents are properly stored with metadata | 25 |
| Search returns relevant documents | 25 |
| Answers include accurate citations | 25 |
| Handles unknown questions gracefully | 15 |
| Code is clean and well-documented | 10 |
| **Total** | **100** |

---

## Lesson Checkpoint

Before moving to the next lesson, verify:

- [ ] I can explain what RAG is and why it's useful
- [ ] I understand how embeddings represent text meaning
- [ ] I can create a vector store and search it
- [ ] I can build a basic Q&A system with citations
- [ ] I completed Assignment 2.1

---

## Key Takeaways

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG KEY CONCEPTS                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. EMBEDDINGS = Numbers that capture meaning               │
│     Similar text → Similar numbers                          │
│                                                              │
│  2. VECTOR STORE = Database for embeddings                  │
│     Enables fast similarity search                          │
│                                                              │
│  3. RETRIEVAL = Find relevant documents                     │
│     Based on similarity to question                         │
│                                                              │
│  4. AUGMENTATION = Add context to prompt                    │
│     Give LLM the information it needs                       │
│                                                              │
│  5. GENERATION = LLM creates answer                         │
│     Grounded in retrieved context                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Next Lesson Preview

In **Lesson 2.2: Document Loading & Processing**, you'll learn:
- How to load PDFs, text files, and web pages
- Chunking strategies for optimal retrieval
- Adding metadata to documents

**Continue to:** `tutorials/module_02_rag_foundations/lesson_02_document_loading.md`
