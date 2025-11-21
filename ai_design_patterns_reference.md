# AI Design Patterns Reference Guide
## Quick Reference for LangChain/LangGraph Patterns

---

## Pattern Categories

1. **Reasoning Patterns** - How AI thinks
2. **Retrieval Patterns** - How AI accesses knowledge
3. **Agent Patterns** - How AI takes actions
4. **Memory Patterns** - How AI remembers
5. **Orchestration Patterns** - How AI coordinates
6. **Evaluation Patterns** - How AI improves

---

## 1. REASONING PATTERNS

### Pattern 1.1: Chain-of-Thought (CoT)

**Purpose**: Break down complex problems into steps

**When to Use**:
- Complex reasoning tasks
- Math problems
- Multi-step analysis
- Debugging logic

**Architecture Diagram**:
```
┌─────────────────────────────────────────────────────────────┐
│                  CHAIN-OF-THOUGHT FLOW                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Input Problem                                              │
│        │                                                     │
│        ▼                                                     │
│   ┌─────────────────┐                                       │
│   │ Step 1: Identify │ ──► What do we know?                 │
│   │ Known Variables  │                                       │
│   └────────┬────────┘                                       │
│            │                                                 │
│            ▼                                                 │
│   ┌─────────────────┐                                       │
│   │ Step 2: Identify │ ──► What do we need to find?         │
│   │ Target/Goal     │                                       │
│   └────────┬────────┘                                       │
│            │                                                 │
│            ▼                                                 │
│   ┌─────────────────┐                                       │
│   │ Step 3: Apply   │ ──► Use relevant formulas/logic       │
│   │ Reasoning       │                                       │
│   └────────┬────────┘                                       │
│            │                                                 │
│            ▼                                                 │
│   ┌─────────────────┐                                       │
│   │ Step 4: Verify  │ ──► Check answer makes sense          │
│   │ Result          │                                       │
│   └────────┬────────┘                                       │
│            │                                                 │
│            ▼                                                 │
│       Final Answer                                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Implementation Template**:
```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

def chain_of_thought(problem: str, llm_model: str = "gpt-4o-mini"):
    """Solve problem with step-by-step reasoning."""

    llm = ChatOpenAI(model=llm_model, temperature=0)

    prompt = ChatPromptTemplate.from_template("""
Solve this problem step by step:

Problem: {problem}

Let's break this down:
Step 1: [Identify what we know]
Step 2: [Determine what we need to find]
Step 3: [Apply relevant concepts]
Step 4: [Calculate/Reason]
Step 5: [Verify answer]

Solution:
""")

    chain = prompt | llm
    result = chain.invoke({"problem": problem})

    return result.content

# Example usage
problem = "If a store sells 45 items at $12 each and has $200 in expenses, what's the profit?"
solution = chain_of_thought(problem)
print(solution)
```

**Variations**:
1. **Zero-shot CoT**: Just add "Let's think step by step"
2. **Few-shot CoT**: Provide example reasoning
3. **Self-consistency CoT**: Generate multiple reasoning paths, take majority vote

```python
def self_consistency_cot(problem: str, n_samples: int = 5):
    """Generate multiple reasoning paths."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    answers = []
    for _ in range(n_samples):
        result = chain_of_thought(problem)
        # Extract final answer
        answers.append(result)

    # Take majority vote
    from collections import Counter
    final_answer = Counter(answers).most_common(1)[0][0]

    return final_answer
```

---

### Pattern 1.2: Tree of Thoughts (ToT)

**Purpose**: Explore multiple reasoning branches

**When to Use**:
- Strategic planning
- Creative problem solving
- Decision making with uncertainty

**Architecture Diagram**:
```
┌─────────────────────────────────────────────────────────────┐
│                    TREE OF THOUGHTS                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│                     Problem                                  │
│                        │                                     │
│         ┌──────────────┼──────────────┐                     │
│         │              │              │                      │
│         ▼              ▼              ▼                      │
│    ┌─────────┐   ┌─────────┐   ┌─────────┐                  │
│    │Thought 1│   │Thought 2│   │Thought 3│                  │
│    │Score:0.8│   │Score:0.6│   │Score:0.9│ ◄── Best         │
│    └────┬────┘   └────┬────┘   └────┬────┘                  │
│         │              │              │                      │
│         ▼              ▼              ▼                      │
│    ┌─────────┐   ┌─────────┐   ┌─────────┐                  │
│    │Expand   │   │ Prune   │   │Expand   │                  │
│    │Further  │   │(Low Scr)│   │Further  │                  │
│    └────┬────┘   └─────────┘   └────┬────┘                  │
│         │                           │                        │
│         ▼                           ▼                        │
│   [Sub-thoughts]             [Sub-thoughts]                  │
│         │                           │                        │
│         └───────────┬───────────────┘                        │
│                     │                                        │
│                     ▼                                        │
│              Best Solution                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Implementation Template**:
```python
from typing import List, Dict
import json

class TreeOfThoughts:
    """Explore multiple reasoning paths."""

    def __init__(self, llm_model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0.7)

    def generate_thoughts(self, problem: str, n_thoughts: int = 3) -> List[str]:
        """Generate different initial approaches."""
        prompt = ChatPromptTemplate.from_template("""
Given this problem: {problem}

Generate {n} different approaches to solve it.
For each approach, provide a one-sentence description.

Output as JSON list: ["approach1", "approach2", ...]
""")

        chain = prompt | self.llm
        result = chain.invoke({"problem": problem, "n": n_thoughts})

        return json.loads(result.content)

    def evaluate_thought(self, thought: str, problem: str) -> float:
        """Evaluate promise of a thought."""
        prompt = ChatPromptTemplate.from_template("""
Problem: {problem}
Approach: {thought}

Rate this approach from 0.0 to 1.0 based on:
- Feasibility
- Completeness
- Efficiency

Output only the number.
""")

        chain = prompt | self.llm
        result = chain.invoke({"problem": problem, "thought": thought})

        return float(result.content.strip())

    def solve(self, problem: str, depth: int = 2) -> str:
        """Solve using tree exploration."""
        # Generate initial thoughts
        thoughts = self.generate_thoughts(problem)

        # Evaluate and select best
        scored_thoughts = [
            (thought, self.evaluate_thought(thought, problem))
            for thought in thoughts
        ]

        best_thought = max(scored_thoughts, key=lambda x: x[1])[0]

        # Continue exploration (simplified)
        return f"Best approach: {best_thought}"

# Example
tot = TreeOfThoughts()
solution = tot.solve("Plan a cross-country road trip on a $2000 budget")
```

---

### Pattern 1.3: ReAct (Reasoning + Acting)

**Purpose**: Interleave reasoning and tool usage

**When to Use**:
- Tasks requiring external information
- Multi-step processes
- Interactive problem solving

**Architecture Diagram**:
```
┌─────────────────────────────────────────────────────────────┐
│                      ReAct LOOP                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   User Query                                                 │
│       │                                                      │
│       ▼                                                      │
│   ┌───────────────────────────────────────┐                 │
│   │         THOUGHT                        │                 │
│   │  "I need to find information about..." │                 │
│   └───────────────────┬───────────────────┘                 │
│                       │                                      │
│                       ▼                                      │
│   ┌───────────────────────────────────────┐                 │
│   │         ACTION                         │                 │
│   │  Tool: Search                          │                 │
│   │  Input: "query terms"                  │                 │
│   └───────────────────┬───────────────────┘                 │
│                       │                                      │
│                       ▼                                      │
│   ┌───────────────────────────────────────┐                 │
│   │        OBSERVATION                     │                 │
│   │  "Results from tool execution..."      │                 │
│   └───────────────────┬───────────────────┘                 │
│                       │                                      │
│           ┌───────────┴───────────┐                         │
│           │                       │                          │
│           ▼                       ▼                          │
│     Need more info?          Have enough?                    │
│           │                       │                          │
│           │                       ▼                          │
│           │              ┌─────────────────┐                │
│           └──────────────│ FINAL ANSWER    │                │
│             (Loop back)  └─────────────────┘                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Implementation Template**:
```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate

def create_react_agent_example():
    """Create ReAct agent with tools."""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Define tools
    def calculator(expression: str) -> str:
        """Evaluate mathematical expression."""
        try:
            return str(eval(expression))
        except:
            return "Error in calculation"

    def search(query: str) -> str:
        """Simulate web search."""
        return f"Search results for: {query}"

    tools = [
        Tool(
            name="Calculator",
            func=calculator,
            description="Useful for math calculations. Input should be a valid Python expression."
        ),
        Tool(
            name="Search",
            func=search,
            description="Useful for finding current information. Input should be a search query."
        )
    ]

    # ReAct prompt template
    template = """Answer the following question using this format:

Question: {input}

Thought: [Your reasoning about what to do]
Action: [Tool name]
Action Input: [Input for the tool]
Observation: [Result from tool]
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: [Your final answer]

Begin!

Question: {input}
{agent_scratchpad}
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["input", "agent_scratchpad"]
    )

    # Create agent
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5
    )

    return agent_executor

# Example usage
agent = create_react_agent_example()
result = agent.invoke({
    "input": "What is 25% of 480, and why is this calculation useful?"
})
```

---

## 2. RETRIEVAL PATTERNS (RAG)

### Pattern 2.1: Basic RAG

**Purpose**: Ground LLM in external knowledge

**Architecture Diagram**:
```
┌─────────────────────────────────────────────────────────────┐
│                      BASIC RAG FLOW                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Documents    ────►  Chunking  ────►  Embedding  ────►      │
│  (PDF, TXT)          (Split)          (Vectors)             │
│                                            │                 │
│                                            ▼                 │
│                                      Vector Store            │
│                                      (Chroma/FAISS)          │
│                                            │                 │
│                         ┌──────────────────┘                 │
│                         │                                    │
│  User Query ────► Embed Query ────► Similarity Search        │
│                                            │                 │
│                                            ▼                 │
│                                   Retrieved Chunks           │
│                                            │                 │
│                                            ▼                 │
│                              ┌──────────────────────┐        │
│                              │   Prompt Template    │        │
│                              │   Context: {chunks}  │        │
│                              │   Question: {query}  │        │
│                              └──────────┬───────────┘        │
│                                         │                    │
│                                         ▼                    │
│                                      LLM (GPT-4o-mini)       │
│                                         │                    │
│                                         ▼                    │
│                                     Answer                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Implementation Template**:
```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

def basic_rag_pipeline(documents: List[str], query: str):
    """Simple RAG implementation."""

    # 1. Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.create_documents(documents)

    # 2. Create embeddings and vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(splits, embeddings)

    # 3. Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 4. Create QA chain
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # 5. Query
    result = qa_chain.invoke({"query": query})

    return result

# Example
docs = ["RAG stands for Retrieval Augmented Generation...", "..."]
result = basic_rag_pipeline(docs, "What is RAG?")
```

---

### Pattern 2.2: RAG Fusion (Multi-Query)

**Purpose**: Improve retrieval through query diversity

**Architecture Diagram**:
```
┌─────────────────────────────────────────────────────────────┐
│                     RAG FUSION FLOW                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Original Query                                              │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────────────────────────┐                    │
│  │      QUERY EXPANSION (LLM)          │                    │
│  │  Generate N query variations        │                    │
│  └─────────────────────────────────────┘                    │
│       │                                                      │
│       ├──────────────┬──────────────┬──────────────┐        │
│       │              │              │              │         │
│       ▼              ▼              ▼              ▼         │
│   Query 1        Query 2        Query 3        Query N       │
│       │              │              │              │         │
│       ▼              ▼              ▼              ▼         │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐     │
│  │Retrieve │   │Retrieve │   │Retrieve │   │Retrieve │     │
│  │ Docs    │   │ Docs    │   │ Docs    │   │ Docs    │     │
│  └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘     │
│       │              │              │              │         │
│       └──────────────┴──────────────┴──────────────┘        │
│                         │                                    │
│                         ▼                                    │
│              ┌─────────────────────┐                        │
│              │ RECIPROCAL RANK     │                        │
│              │ FUSION (RRF)        │                        │
│              │ Score & Deduplicate │                        │
│              └──────────┬──────────┘                        │
│                         │                                    │
│                         ▼                                    │
│                  Top-K Documents                             │
│                         │                                    │
│                         ▼                                    │
│                   Generate Answer                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Implementation Template**:
```python
from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads

def rag_fusion(vectorstore, query: str, llm_model: str = "gpt-4o-mini"):
    """RAG with multiple query variations."""

    llm = ChatOpenAI(model=llm_model, temperature=0.7)

    # Generate query variations
    prompt = ChatPromptTemplate.from_template(
        """Generate 3 different versions of this query to improve retrieval:

Original: {query}

Variations:
1.
2.
3.
"""
    )

    chain = prompt | llm
    variations_text = chain.invoke({"query": query}).content

    # Extract variations
    queries = [query] + [
        line.split('. ', 1)[1].strip()
        for line in variations_text.split('\n')
        if line.strip() and line[0].isdigit()
    ]

    # Retrieve for each query
    all_docs = []
    for q in queries:
        docs = vectorstore.similarity_search(q, k=4)
        all_docs.extend(docs)

    # Remove duplicates using Reciprocal Rank Fusion
    unique_docs = []
    seen_contents = set()
    for doc in all_docs:
        if doc.page_content not in seen_contents:
            unique_docs.append(doc)
            seen_contents.add(doc.page_content)

    return unique_docs[:4]  # Return top 4
```

---

### Pattern 2.3: Self-RAG (Self-Reflective RAG)

**Purpose**: RAG that critiques and corrects itself

**Architecture Diagram**:
```
┌─────────────────────────────────────────────────────────────┐
│                     SELF-RAG FLOW                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   User Query                                                 │
│       │                                                      │
│       ▼                                                      │
│   ┌─────────────────┐                                       │
│   │  Initial        │                                       │
│   │  Retrieval      │                                       │
│   └────────┬────────┘                                       │
│            │                                                 │
│            ▼                                                 │
│   ┌─────────────────┐                                       │
│   │  Generate       │                                       │
│   │  Initial Answer │                                       │
│   └────────┬────────┘                                       │
│            │                                                 │
│            ▼                                                 │
│   ┌─────────────────────────────────────┐                   │
│   │       SELF-CRITIQUE                  │                   │
│   │  Is answer relevant? ──► Score      │                   │
│   │  Is it grounded?    ──► Score       │                   │
│   │  Any hallucinations?──► Score       │                   │
│   └────────┬────────────────────────────┘                   │
│            │                                                 │
│     ┌──────┴──────┐                                         │
│     │             │                                          │
│     ▼             ▼                                          │
│   Good          Needs Improvement                            │
│     │             │                                          │
│     │             ▼                                          │
│     │      ┌─────────────────┐                              │
│     │      │ Retrieve More   │                              │
│     │      │ Regenerate      │                              │
│     │      └────────┬────────┘                              │
│     │               │                                        │
│     └───────────────┼───────────────────────────────────────│
│                     │                                        │
│                     ▼                                        │
│              Final Answer                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Implementation Template**:
```python
def self_rag(vectorstore, query: str, llm_model: str = "gpt-4o-mini"):
    """RAG with self-reflection and correction."""

    llm = ChatOpenAI(model=llm_model, temperature=0)

    # Initial retrieval and generation
    docs = vectorstore.similarity_search(query, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Generate initial answer
    initial_prompt = ChatPromptTemplate.from_template("""
Context: {context}
Question: {query}

Answer:
""")

    initial_answer = (initial_prompt | llm).invoke({
        "context": context,
        "query": query
    }).content

    # Self-critique
    critique_prompt = ChatPromptTemplate.from_template("""
Question: {query}
Answer: {answer}
Context: {context}

Evaluate this answer:
1. Is it relevant to the question?
2. Is it supported by the context?
3. Are there any unsupported claims?

If issues exist, provide a corrected answer.

Evaluation:
""")

    critique = (critique_prompt | llm).invoke({
        "query": query,
        "answer": initial_answer,
        "context": context
    }).content

    # If critique suggests improvements, retrieve more and regenerate
    if "corrected answer" in critique.lower():
        # Additional retrieval with refined query
        refined_query = f"{query} [specific aspects needing clarification]"
        additional_docs = vectorstore.similarity_search(refined_query, k=2)

        # Regenerate with expanded context
        expanded_context = context + "\n\n" + "\n\n".join([
            doc.page_content for doc in additional_docs
        ])

        final_answer = (initial_prompt | llm).invoke({
            "context": expanded_context,
            "query": query
        }).content

        return {
            "answer": final_answer,
            "reflection": critique,
            "improved": True
        }

    return {
        "answer": initial_answer,
        "reflection": critique,
        "improved": False
    }
```

---

### Pattern 2.4: Corrective RAG (CRAG)

**Purpose**: Verify retrieval quality and correct if needed

**Architecture Diagram**:
```
┌─────────────────────────────────────────────────────────────┐
│                  CORRECTIVE RAG (CRAG)                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   User Query ────► Retrieve Documents                        │
│                           │                                  │
│                           ▼                                  │
│              ┌──────────────────────────┐                   │
│              │   RELEVANCE GRADER       │                   │
│              │   For each document:     │                   │
│              │   - RELEVANT             │                   │
│              │   - PARTIALLY RELEVANT   │                   │
│              │   - NOT RELEVANT         │                   │
│              └────────────┬─────────────┘                   │
│                           │                                  │
│           ┌───────────────┼───────────────┐                 │
│           │               │               │                  │
│           ▼               ▼               ▼                  │
│      All Good      Mixed Results     Poor Results            │
│           │               │               │                  │
│           │               ▼               ▼                  │
│           │        ┌─────────────┐  ┌─────────────┐         │
│           │        │ Filter Out  │  │ Web Search  │         │
│           │        │ Irrelevant  │  │ Fallback    │         │
│           │        └──────┬──────┘  └──────┬──────┘         │
│           │               │                │                 │
│           └───────────────┴────────────────┘                 │
│                           │                                  │
│                           ▼                                  │
│                   Generate Answer                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Implementation Template**:
```python
def corrective_rag(vectorstore, query: str, llm_model: str = "gpt-4o-mini"):
    """RAG with retrieval quality assessment."""

    llm = ChatOpenAI(model=llm_model, temperature=0)

    # Retrieve documents
    docs = vectorstore.similarity_search(query, k=4)

    # Assess relevance of each document
    relevance_prompt = ChatPromptTemplate.from_template("""
Query: {query}
Document: {doc}

Is this document relevant to answering the query?
Answer only: RELEVANT, PARTIALLY_RELEVANT, or NOT_RELEVANT
""")

    assessed_docs = []
    for doc in docs:
        assessment = (relevance_prompt | llm).invoke({
            "query": query,
            "doc": doc.page_content
        }).content.strip()

        assessed_docs.append((doc, assessment))

    # Filter and potentially expand
    relevant_docs = [
        doc for doc, assessment in assessed_docs
        if assessment in ["RELEVANT", "PARTIALLY_RELEVANT"]
    ]

    if len(relevant_docs) < 2:
        # Retrieval quality is low, try web search or different strategy
        print("Low retrieval quality, expanding search...")
        # Implement fallback retrieval strategy

    # Generate answer with quality-filtered context
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    answer_prompt = ChatPromptTemplate.from_template("""
High-quality context: {context}
Question: {query}

Based on the relevant context provided, answer the question.
If the context is insufficient, state what information is missing.

Answer:
""")

    answer = (answer_prompt | llm).invoke({
        "context": context,
        "query": query
    }).content

    return {
        "answer": answer,
        "relevant_docs": len(relevant_docs),
        "total_docs": len(docs)
    }
```

---

### Pattern 2.5: Adaptive RAG

**Purpose**: Route queries to appropriate retrieval strategy

**Architecture Diagram**:
```
┌─────────────────────────────────────────────────────────────┐
│                     ADAPTIVE RAG                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   User Query                                                 │
│       │                                                      │
│       ▼                                                      │
│   ┌─────────────────────────────────────┐                   │
│   │       QUERY CLASSIFIER              │                   │
│   │   Analyze query type:               │                   │
│   │   - Simple Fact                     │                   │
│   │   - Complex Reasoning               │                   │
│   │   - Multi-hop                       │                   │
│   │   - Procedural                      │                   │
│   └───────────────────┬─────────────────┘                   │
│                       │                                      │
│     ┌─────────────────┼─────────────────┬─────────────────┐ │
│     │                 │                 │                 │ │
│     ▼                 ▼                 ▼                 ▼ │
│ ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐│
│ │ Direct  │     │  RAG    │     │ Self-   │     │Corrective│
│ │Retrieval│     │ Fusion  │     │  RAG    │     │   RAG   ││
│ │ k=2     │     │Multi-Q  │     │Reflectn │     │ Verify  ││
│ └────┬────┘     └────┬────┘     └────┬────┘     └────┬────┘│
│      │               │               │               │      │
│      └───────────────┴───────────────┴───────────────┘      │
│                          │                                   │
│                          ▼                                   │
│                   Generate Answer                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Implementation Template**:
```python
from enum import Enum

class QueryType(Enum):
    SIMPLE_FACT = "simple_fact"
    COMPLEX_REASONING = "complex_reasoning"
    MULTI_HOP = "multi_hop"
    PROCEDURAL = "procedural"

def adaptive_rag(vectorstore, query: str, llm_model: str = "gpt-4o-mini"):
    """Route query to optimal retrieval strategy."""

    llm = ChatOpenAI(model=llm_model, temperature=0)

    # Classify query
    classifier_prompt = ChatPromptTemplate.from_template("""
Classify this query:

Query: {query}

Classification (choose one):
- SIMPLE_FACT: Direct factual question
- COMPLEX_REASONING: Requires analysis
- MULTI_HOP: Needs multiple pieces of information
- PROCEDURAL: How-to question

Classification:
""")

    classification = (classifier_prompt | llm).invoke({
        "query": query
    }).content.strip()

    # Route to appropriate strategy
    if "SIMPLE_FACT" in classification:
        # Direct retrieval
        docs = vectorstore.similarity_search(query, k=2)
        strategy = "direct"

    elif "MULTI_HOP" in classification:
        # Decompose query and retrieve for each part
        docs = rag_fusion(vectorstore, query)
        strategy = "fusion"

    elif "COMPLEX_REASONING" in classification:
        # Use self-reflective RAG
        result = self_rag(vectorstore, query)
        return {
            **result,
            "strategy": "self-rag",
            "classification": classification
        }

    else:  # PROCEDURAL
        # Use corrective RAG
        result = corrective_rag(vectorstore, query)
        return {
            **result,
            "strategy": "corrective",
            "classification": classification
        }

    # Generate answer
    context = "\n\n".join([doc.page_content for doc in docs])
    answer = (ChatPromptTemplate.from_template(
        "Context: {context}\nQuestion: {query}\nAnswer:"
    ) | llm).invoke({"context": context, "query": query}).content

    return {
        "answer": answer,
        "strategy": strategy,
        "classification": classification
    }
```

---

## 3. AGENT PATTERNS

### Pattern 3.1: Tool-Using Agent

**Architecture Diagram**:
```
┌─────────────────────────────────────────────────────────────┐
│                 TOOL-USING AGENT                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  User Request ────► Agent Planner                            │
│                          │                                   │
│                          ▼                                   │
│              ┌───────────────────────┐                      │
│              │   SELECT TOOL         │                      │
│              │   Based on task needs │                      │
│              └───────────┬───────────┘                      │
│                          │                                   │
│      ┌───────────────────┼───────────────────┐              │
│      │                   │                   │               │
│      ▼                   ▼                   ▼               │
│ ┌──────────┐       ┌──────────┐       ┌──────────┐         │
│ │Calculator│       │ Search   │       │  Code    │         │
│ │   Tool   │       │   Tool   │       │ Executor │         │
│ └────┬─────┘       └────┬─────┘       └────┬─────┘         │
│      │                  │                  │                │
│      └──────────────────┴──────────────────┘                │
│                         │                                    │
│                         ▼                                    │
│                   Tool Result                                │
│                         │                                    │
│                         ▼                                    │
│                 Process & Respond                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Implementation Template**:
```python
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

def create_tool_agent(llm_model: str = "gpt-4o-mini"):
    """Agent with multiple tools."""

    llm = ChatOpenAI(model=llm_model, temperature=0)

    # Define tools
    tools = [
        Tool(
            name="Calculator",
            func=lambda x: eval(x),
            description="For math calculations"
        ),
        Tool(
            name="StringLength",
            func=lambda x: len(x),
            description="Returns length of string"
        )
    ]

    # Initialize agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    return agent
```

---

### Pattern 3.2: Multi-Agent Collaboration

**Architecture Diagram**:
```
┌─────────────────────────────────────────────────────────────┐
│              MULTI-AGENT COLLABORATION                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   User Task                                                  │
│       │                                                      │
│       ▼                                                      │
│   ┌─────────────────────────────────────┐                   │
│   │         SUPERVISOR AGENT            │                   │
│   │   Coordinates and delegates tasks   │                   │
│   └───────────────────┬─────────────────┘                   │
│                       │                                      │
│       ┌───────────────┼───────────────┐                     │
│       │               │               │                      │
│       ▼               ▼               ▼                      │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                 │
│  │Researcher│    │ Writer  │    │ Critic  │                 │
│  │  Agent  │    │  Agent  │    │  Agent  │                 │
│  └────┬────┘    └────┬────┘    └────┬────┘                 │
│       │              │              │                        │
│       ▼              ▼              ▼                        │
│  Research        Write Draft    Review &                     │
│  Information     Content        Feedback                     │
│       │              │              │                        │
│       └──────────────┴──────────────┘                        │
│                      │                                       │
│                      ▼                                       │
│              Final Output                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Implementation Template**:
```python
from langgraph.graph import Graph, END

def create_multi_agent_system():
    """Collaborative multi-agent system."""

    # Define agents
    class ResearchAgent:
        def __call__(self, state):
            # Research information
            return {"research": "Research findings..."}

    class WriterAgent:
        def __call__(self, state):
            # Write based on research
            return {"draft": "Written content..."}

    class EditorAgent:
        def __call__(self, state):
            # Edit and finalize
            return {"final": "Final version..."}

    # Create workflow
    workflow = Graph()

    workflow.add_node("researcher", ResearchAgent())
    workflow.add_node("writer", WriterAgent())
    workflow.add_node("editor", EditorAgent())

    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "editor")
    workflow.add_edge("editor", END)

    workflow.set_entry_point("researcher")

    return workflow.compile()
```

---

## 4. MEMORY PATTERNS

### Pattern 4.1: Conversation Buffer Memory

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history"
)
```

### Pattern 4.2: Conversation Summary Memory

```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    return_messages=True
)
```

### Pattern 4.3: Entity Memory

```python
from langchain.memory import ConversationEntityMemory

memory = ConversationEntityMemory(
    llm=ChatOpenAI(model="gpt-4o-mini")
)
```

**Memory Pattern Comparison**:
```
┌─────────────────────────────────────────────────────────────┐
│                   MEMORY PATTERNS                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Buffer Memory          Summary Memory        Entity Memory  │
│  ┌─────────────┐       ┌─────────────┐      ┌─────────────┐ │
│  │ Stores all  │       │Summarizes   │      │ Tracks      │ │
│  │ messages    │       │conversation │      │ entities    │ │
│  │ verbatim    │       │             │      │ mentioned   │ │
│  └─────────────┘       └─────────────┘      └─────────────┘ │
│                                                              │
│  Pros:                  Pros:                Pros:           │
│  - Complete history     - Token efficient    - Contextual    │
│  - Simple               - Long conversations - Tracks topics │
│                                                              │
│  Cons:                  Cons:                Cons:           │
│  - Token expensive      - May lose details   - Complex setup │
│  - Limited length       - LLM cost for      - Maintenance   │
│                           summarization                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. ORCHESTRATION PATTERNS

### Pattern 5.1: Sequential Chain

```python
from langchain.chains import LLMChain, SequentialChain

# Chain 1: Generate outline
outline_chain = LLMChain(llm=llm, prompt=outline_prompt, output_key="outline")

# Chain 2: Write content
content_chain = LLMChain(llm=llm, prompt=content_prompt, output_key="content")

# Combine
overall_chain = SequentialChain(
    chains=[outline_chain, content_chain],
    input_variables=["topic"],
    output_variables=["outline", "content"]
)
```

### Pattern 5.2: Router Chain

```python
from langchain.chains.router import MultiPromptChain

# Different prompts for different topics
prompts = {
    "technical": technical_prompt,
    "creative": creative_prompt,
    "analytical": analytical_prompt
}

router_chain = MultiPromptChain.from_prompts(
    llm=llm,
    prompt_infos=prompts
)
```

---

## 6. EVALUATION PATTERNS

### Pattern 6.1: LLM-as-Judge

```python
def llm_evaluate(question: str, answer: str, context: str):
    """Use LLM to evaluate answer quality."""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    eval_prompt = ChatPromptTemplate.from_template("""
Evaluate this answer on a scale of 1-10:

Question: {question}
Answer: {answer}
Context: {context}

Criteria:
- Relevance (1-10)
- Accuracy (1-10)
- Completeness (1-10)
- Clarity (1-10)

Provide scores and justification.
""")

    result = (eval_prompt | llm).invoke({
        "question": question,
        "answer": answer,
        "context": context
    })

    return result.content
```

### Pattern 6.2: Critique and Revision

```python
def critique_and_revise(content: str, criteria: str):
    """Iteratively improve content through criticism."""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    for iteration in range(3):
        # Critique
        critique = (ChatPromptTemplate.from_template("""
Content: {content}
Criteria: {criteria}

Provide specific critique and suggestions for improvement.
""") | llm).invoke({"content": content, "criteria": criteria}).content

        # Revise
        content = (ChatPromptTemplate.from_template("""
Original: {content}
Critique: {critique}

Provide revised version addressing the critique.
""") | llm).invoke({"content": content, "critique": critique}).content

    return content
```

---

## Pattern Selection Guide

| Use Case | Recommended Pattern |
|----------|-------------------|
| Complex math/logic | Chain-of-Thought |
| Strategic planning | Tree of Thoughts |
| Need external tools | ReAct Agent |
| Knowledge grounding | Basic RAG |
| Diverse retrieval | RAG Fusion |
| Quality assurance | Self-RAG or CRAG |
| Dynamic routing | Adaptive RAG |
| Multi-step tasks | Multi-Agent |
| Conversation | Memory Patterns |
| Iterative improvement | Critique-Revise |

---

## Combining Patterns

```python
def advanced_system(query: str):
    """Combine multiple patterns."""

    # 1. Classify query (Adaptive RAG)
    classification = classify_query(query)

    # 2. Retrieve with appropriate strategy
    docs = retrieve_with_strategy(query, classification)

    # 3. Generate with Chain-of-Thought
    answer = chain_of_thought_generate(query, docs)

    # 4. Self-critique and revise
    final_answer = self_critique(answer, query, docs)

    # 5. Evaluate
    evaluation = llm_evaluate(query, final_answer, docs)

    return {
        "answer": final_answer,
        "evaluation": evaluation,
        "strategy": classification
    }
```

---

**Use this as a reference when building your systems!**

Each pattern can be:
- Used standalone
- Combined with others
- Customized for your use case
- Extended with additional logic

Ask Claude Code to: "Implement [PATTERN_NAME] from the design patterns reference with a complete working example for [YOUR_USE_CASE]"
