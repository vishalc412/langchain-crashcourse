# LangChain & LangGraph Complete Course
## Instructor-Led Tutorial Series

**Duration:** 4-6 Weeks | **Level:** Beginner to Advanced | **Model:** GPT-4o-mini

---

## Welcome!

This is a **structured, instructor-led course** designed to take you from complete beginner to confident LangChain developer. Unlike typical documentation, this course follows a proven learning methodology:

```
LEARN IT → SEE IT → DO IT → PROVE IT
(Theory)   (Demo)   (Practice)  (Assignment)
```

---

## Quick Start

### Step 1: Setup (5 minutes)
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Step 2: Start Learning
Open `COURSE_CURRICULUM.md` and begin with Module 1, Lesson 1.

---

## Course Structure

```
📁 langchain-crashcourse/
│
├── 📄 COURSE_CURRICULUM.md          ← START HERE!
│
├── 📁 tutorials/                     ← Step-by-step lessons
│   ├── 📁 module_01_fundamentals/
│   │   ├── lesson_01_setup.md        (45 min)
│   │   ├── lesson_02_prompts.md      (50 min)
│   │   ├── lesson_03_chains.md       (60 min)
│   │   └── lesson_04_memory.md       (60 min)
│   │
│   ├── 📁 module_02_rag_foundations/
│   │   ├── lesson_01_understanding_rag.md  (60 min)
│   │   ├── lesson_02_document_loading.md   (45 min)
│   │   ├── lesson_03_vector_stores.md      (45 min)
│   │   └── lesson_04_basic_rag.md          (60 min)
│   │
│   ├── 📁 module_03_advanced_rag/
│   ├── 📁 module_04_langgraph/
│   └── 📁 module_05_design_patterns/
│
├── 📁 assignments/                   ← Graded exercises
│   ├── README.md
│   └── assignment_master.md          (All assignments)
│
├── 📁 solutions/                     ← Check after attempting!
│
└── 📁 resources/                     ← Reference materials
```

---

## What You'll Learn

### Module 1: LangChain Fundamentals (Week 1)
- Environment setup and configuration
- Making LLM calls with GPT-4o-mini
- Creating reusable prompt templates
- Building chains for multi-step processes
- Adding memory for conversations

**Outcome:** Build a chatbot with conversation memory

### Module 2: RAG Foundations (Week 2)
- Understanding RAG architecture
- Loading and processing documents
- Creating embeddings and vector stores
- Building retrieval pipelines
- Generating answers with citations

**Outcome:** Build a document Q&A system

### Module 3: Advanced RAG (Week 3)
- Hybrid search (vector + keyword)
- Query transformation and expansion
- Reranking strategies
- RAG evaluation metrics

**Outcome:** Build a production-quality RAG pipeline

### Module 4: LangGraph (Week 4)
- Graph-based workflows
- State management
- Conditional routing
- Multi-agent systems

**Outcome:** Build a multi-agent collaboration system

### Module 5: Design Patterns (Week 5)
- ReAct (Reasoning + Acting)
- Chain-of-Thought prompting
- RAG variations (Fusion, Self-RAG, CRAG)
- Pattern combination strategies

**Outcome:** Implement all major AI design patterns

---

## Lesson Format

Each lesson follows this structure:

### 1. LEARN IT (10-15 min)
- Theory and concepts
- Visual diagrams
- Real-world examples

### 2. SEE IT (15-20 min)
- Complete working demos
- Line-by-line explanations
- Expected outputs

### 3. DO IT (20-30 min)
- Guided exercises
- Fill-in-the-blank code
- Self-check questions

### 4. PROVE IT (30-60 min)
- Independent assignment
- Clear requirements
- Grading rubric

---

## Assessment

| Type | Frequency | Points |
|------|-----------|--------|
| Lesson Exercises | Each lesson | Practice |
| Module Assignments | End of module | 100-150 |
| Capstone Project | Final | 300 |

### Grading Scale
- **A (90-100%):** Excellent
- **B (80-89%):** Good
- **C (70-79%):** Satisfactory
- **D (60-69%):** Needs Improvement

---

## Prerequisites

### Required
- Basic Python knowledge (variables, functions, classes)
- Python 3.9+ installed
- OpenAI API key

### Helpful but Not Required
- Understanding of APIs
- Basic command line usage

---

## Files Overview

| File | Purpose |
|------|---------|
| `COURSE_CURRICULUM.md` | Main course guide - START HERE |
| `QUICKSTART_GUIDE.md` | Prompt templates for Claude Code |
| `langchain_langraph_master_plan.md` | Complete learning roadmap |
| `custom_rag_implementation.md` | Production RAG code |
| `ai_design_patterns_reference.md` | Pattern implementations |
| `INTERACTIVE_ASSIGNMENTS.md` | Additional exercises |
| `requirements.txt` | Python dependencies |
| `.env.example` | Environment template |

---

## Learning Paths

### Path A: Complete Beginner (6 weeks)
Follow every lesson and assignment in order.

### Path B: Python Developer (4 weeks)
- Skim Module 1
- Focus on Modules 2-4
- Complete all assignments

### Path C: Experienced with LLMs (2 weeks)
- Skip to Module 3
- Focus on advanced patterns
- Build capstone project

---

## Getting Help

### Option 1: Check Resources
Look in the `resources/` folder for FAQs and troubleshooting.

### Option 2: Ask Claude Code
```
"I'm on Module [X], Lesson [Y] and stuck on [problem].
My code: [paste code]
Error: [paste error]
Please help me understand what's wrong."
```

### Option 3: Review Previous Lessons
Often issues come from missing something earlier.

---

## Success Tips

1. **Don't skip lessons** - They build on each other
2. **Type the code** - Don't just copy/paste
3. **Run every example** - See it work
4. **Do all exercises** - Practice makes perfect
5. **Attempt before solutions** - Struggle is learning
6. **Ask for help** - When stuck for >15 minutes

---

## Technical Requirements

### Minimum
- Python 3.9+
- 4GB RAM
- Internet connection
- OpenAI API key ($5-10 budget)

### Recommended
- Python 3.11+
- 8GB RAM
- VS Code or PyCharm
- Git installed

---

## Ready to Begin?

1. Complete the setup steps above
2. Open `COURSE_CURRICULUM.md`
3. Start Module 1, Lesson 1
4. Take your time and enjoy learning!

**Let's build amazing AI applications together!**

---

**Version:** 2.0 (Instructor-Led Edition)
**Last Updated:** 2025
