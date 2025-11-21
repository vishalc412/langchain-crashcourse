# LangChain & LangGraph Complete Course Curriculum
## Instructor-Led Tutorial Series

**Course Duration**: 4-6 Weeks
**Difficulty**: Beginner to Advanced
**Prerequisites**: Basic Python knowledge

---

## Welcome, Future AI Developer!

This course will transform you from a complete beginner to a confident LangChain and LangGraph developer. Each lesson follows a proven learning structure:

```
┌─────────────────────────────────────────────────────────────┐
│                    LESSON STRUCTURE                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. LEARN IT     →  Theory & Concepts (10-15 min)           │
│                     - What is it?                            │
│                     - Why do we need it?                     │
│                     - How does it work?                      │
│                                                              │
│  2. SEE IT       →  Instructor Demo (15-20 min)             │
│                     - Watch working code                     │
│                     - Understand each line                   │
│                     - See the output                         │
│                                                              │
│  3. DO IT        →  Guided Practice (20-30 min)             │
│                     - Follow along exercises                 │
│                     - Fill in the blanks                     │
│                     - Run and test                           │
│                                                              │
│  4. PROVE IT     →  Assignment (30-60 min)                  │
│                     - Independent challenge                  │
│                     - Apply what you learned                 │
│                     - Self-assessment                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Course Overview

```
Week 1: Foundation
├── Module 1: LangChain Fundamentals
│   ├── Lesson 1.1: Environment Setup & First LLM Call
│   ├── Lesson 1.2: Prompt Templates & Formatting
│   ├── Lesson 1.3: Chains - Connecting Components
│   └── Lesson 1.4: Memory - Giving AI Context

Week 2: RAG Basics
├── Module 2: RAG Foundations
│   ├── Lesson 2.1: Understanding RAG Architecture
│   ├── Lesson 2.2: Document Loading & Processing
│   ├── Lesson 2.3: Embeddings & Vector Stores
│   └── Lesson 2.4: Building Your First RAG System

Week 3: Advanced Techniques
├── Module 3: Advanced RAG
│   ├── Lesson 3.1: Advanced Retrieval Strategies
│   ├── Lesson 3.2: Query Transformation
│   ├── Lesson 3.3: Reranking & Filtering
│   └── Lesson 3.4: RAG Evaluation

Week 4: Workflows & Agents
├── Module 4: LangGraph
│   ├── Lesson 4.1: Introduction to Graph-Based AI
│   ├── Lesson 4.2: Building Stateful Workflows
│   ├── Lesson 4.3: Conditional Routing
│   └── Lesson 4.4: Multi-Agent Systems

Week 5: Design Patterns
├── Module 5: AI Design Patterns
│   ├── Lesson 5.1: ReAct Pattern
│   ├── Lesson 5.2: Chain-of-Thought
│   ├── Lesson 5.3: RAG Variations
│   └── Lesson 5.4: Combining Patterns

Week 6: Capstone
└── Final Project: Production RAG Application
```

---

## Before You Begin

### Required Software
```bash
# Python 3.9 or higher
python --version  # Should show 3.9+

# pip (Python package manager)
pip --version
```

### Required Accounts
1. **OpenAI Account** - Get API key from https://platform.openai.com/api-keys
2. **GitHub Account** (optional) - For saving your projects

### Initial Setup (Do This First!)

```bash
# Step 1: Create project directory
mkdir langchain-learning
cd langchain-learning

# Step 2: Create virtual environment
python -m venv venv

# Step 3: Activate virtual environment
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Step 4: Install packages
pip install langchain langchain-openai langgraph chromadb python-dotenv

# Step 5: Create .env file with your API key
echo "OPENAI_API_KEY=your-api-key-here" > .env

# Step 6: Verify installation
python -c "import langchain; print('LangChain installed!')"
```

---

## How to Navigate This Course

### File Organization
```
langchain-crashcourse/
├── COURSE_CURRICULUM.md      ← You are here (Start here!)
├── tutorials/
│   ├── module_01_fundamentals/
│   │   ├── lesson_01_setup.md
│   │   ├── lesson_02_prompts.md
│   │   └── ...
│   ├── module_02_rag_foundations/
│   └── ...
├── assignments/
│   ├── assignment_01.md
│   ├── assignment_02.md
│   └── ...
├── solutions/                 ← Check only after attempting!
└── resources/
    └── cheatsheets/
```

### Recommended Learning Path

**If you're a complete beginner:**
1. Start with Module 1, Lesson 1
2. Complete every exercise
3. Do all assignments
4. Don't skip ahead!

**If you know Python but not AI:**
1. Skim Module 1 quickly
2. Focus on Module 2 onwards
3. Do all RAG-related assignments

**If you have some LangChain experience:**
1. Skip to Module 3 or 4
2. Use Module 1-2 as reference
3. Focus on advanced patterns

---

## Module Summaries

### Module 1: LangChain Fundamentals
**What You'll Learn:**
- How to connect to OpenAI's GPT-4o-mini
- Creating reusable prompt templates
- Building chains that combine multiple steps
- Adding memory to create chatbots

**Key Skills:**
- `ChatOpenAI` initialization
- `ChatPromptTemplate` usage
- `LLMChain` creation
- `ConversationBufferMemory` implementation

**By the End:** You'll build a chatbot that remembers conversation history.

---

### Module 2: RAG Foundations
**What You'll Learn:**
- Why RAG is essential for AI applications
- Processing documents into searchable chunks
- Creating and storing embeddings
- Retrieving relevant information

**Key Skills:**
- Document loading with `PyPDFLoader`
- Text splitting with `RecursiveCharacterTextSplitter`
- Vector storage with `Chroma`
- Similarity search

**By the End:** You'll build a Q&A system that answers questions from your documents.

---

### Module 3: Advanced RAG
**What You'll Learn:**
- Hybrid search (combining keyword and semantic search)
- Query expansion and transformation
- Reranking for better results
- Evaluating RAG systems

**Key Skills:**
- BM25 + vector search
- Query decomposition
- Cross-encoder reranking
- RAGAS evaluation metrics

**By the End:** You'll build a production-quality RAG pipeline.

---

### Module 4: LangGraph
**What You'll Learn:**
- Graph-based AI workflows
- State management across nodes
- Conditional branching
- Multi-agent collaboration

**Key Skills:**
- `StateGraph` creation
- Node and edge definition
- Conditional routing
- Agent orchestration

**By the End:** You'll build a multi-agent system that collaborates to solve tasks.

---

### Module 5: AI Design Patterns
**What You'll Learn:**
- ReAct (Reasoning + Acting)
- Chain-of-Thought prompting
- RAG variations (Fusion, Self-RAG, CRAG)
- Pattern combination strategies

**By the End:** You'll know when and how to apply each pattern.

---

## Assessment Structure

### Per-Lesson Checkpoints
Each lesson includes:
- 3-5 quick quiz questions
- 1 hands-on exercise
- Self-assessment checklist

### Module Assignments
Each module has a graded assignment:
- Clear requirements
- Starter code provided
- Rubric for self-grading
- Solution available (after attempting)

### Final Project
A comprehensive project combining all skills:
- Build a production RAG application
- Include advanced retrieval
- Add evaluation metrics
- Document your work

---

## Getting Help

### If You're Stuck

**Option 1: Check the Resources**
```
resources/
├── common_errors.md
├── faq.md
└── cheatsheets/
```

**Option 2: Ask Claude Code**
```
"I'm working on [lesson name] and stuck on [specific problem].
My code: [paste code]
Error: [paste error]
Please explain what's wrong and how to fix it."
```

**Option 3: Review Previous Lessons**
Often, issues come from missing something in earlier lessons.

---

## Let's Begin!

**Your first task:** Go to `tutorials/module_01_fundamentals/lesson_01_setup.md`

Remember:
- Take your time
- Run every code example
- Complete every exercise
- Ask for help when stuck

**You've got this! Let's build amazing AI applications together.**

---

## Quick Reference: Key Concepts

| Term | Definition |
|------|------------|
| **LLM** | Large Language Model - AI that understands and generates text |
| **RAG** | Retrieval Augmented Generation - Combining LLMs with external knowledge |
| **Embedding** | Numerical representation of text for similarity comparison |
| **Vector Store** | Database optimized for storing and searching embeddings |
| **Chain** | Sequence of LLM operations connected together |
| **Agent** | AI that can use tools and make decisions |
| **LangGraph** | Framework for building stateful, multi-step AI workflows |

---

**Version:** 2.0
**Last Updated:** 2025
**Author:** LangChain Course Team
