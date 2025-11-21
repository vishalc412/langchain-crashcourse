# Module 1, Lesson 1: Environment Setup & Your First LLM Call

## Lesson Overview
| Duration | Difficulty | Prerequisites |
|----------|------------|---------------|
| 45 minutes | Beginner | Basic Python |

## Learning Objectives
By the end of this lesson, you will be able to:
- [ ] Set up a Python environment for LangChain development
- [ ] Securely manage API keys using environment variables
- [ ] Make your first API call to GPT-4o-mini
- [ ] Understand the basic LangChain architecture

---

## Part 1: LEARN IT - Understanding the Basics

### What is LangChain?

LangChain is a framework that makes it easy to build applications powered by language models. Think of it as a toolkit that handles all the complex parts of working with AI.

```
┌─────────────────────────────────────────────────────────────┐
│                 WITHOUT LANGCHAIN                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  You need to:                                                │
│  ✗ Write API calls manually                                 │
│  ✗ Handle errors yourself                                   │
│  ✗ Build memory systems from scratch                        │
│  ✗ Manage document processing                               │
│  ✗ Create vector search logic                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  WITH LANGCHAIN                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  LangChain provides:                                         │
│  ✓ Ready-to-use LLM connectors                              │
│  ✓ Built-in error handling                                  │
│  ✓ Memory systems out of the box                            │
│  ✓ Document loaders and processors                          │
│  ✓ Vector store integrations                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### What is GPT-4o-mini?

GPT-4o-mini is OpenAI's efficient language model. It's:
- **Fast**: Quick response times
- **Affordable**: Lower cost than GPT-4
- **Capable**: Handles most tasks well
- **Perfect for learning**: Great balance of cost and capability

### How Does an LLM Call Work?

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM CALL FLOW                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Your Code                                                  │
│      │                                                       │
│      ▼                                                       │
│   ┌─────────────────┐                                       │
│   │  LangChain      │  ← Prepares your request              │
│   │  (ChatOpenAI)   │                                       │
│   └────────┬────────┘                                       │
│            │                                                 │
│            ▼                                                 │
│   ┌─────────────────┐                                       │
│   │  OpenAI API     │  ← Sends to OpenAI servers            │
│   │  (Internet)     │                                       │
│   └────────┬────────┘                                       │
│            │                                                 │
│            ▼                                                 │
│   ┌─────────────────┐                                       │
│   │  GPT-4o-mini    │  ← AI processes your request          │
│   │  (AI Model)     │                                       │
│   └────────┬────────┘                                       │
│            │                                                 │
│            ▼                                                 │
│      Response returned to your code                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Why Use Environment Variables for API Keys?

**NEVER put API keys directly in your code!**

```python
# ❌ WRONG - Never do this!
api_key = "sk-abc123..."  # Anyone who sees your code gets your key!

# ✓ CORRECT - Use environment variables
import os
api_key = os.getenv("OPENAI_API_KEY")  # Key is stored safely outside code
```

**Benefits:**
1. **Security**: Keys aren't exposed in code
2. **Flexibility**: Easy to change without editing code
3. **Sharing**: You can share code without sharing keys
4. **Best Practice**: Industry standard approach

---

## Part 2: SEE IT - Instructor Demo

### Demo 1: Setting Up the Environment

First, let's see the complete setup process:

```python
# File: 01_setup_demo.py
"""
INSTRUCTOR DEMO: Setting up LangChain environment
Watch how each step works before trying yourself.
"""

# Step 1: Import required libraries
# os - for accessing environment variables
# dotenv - for loading .env files
# langchain_openai - for connecting to OpenAI

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Step 2: Load environment variables from .env file
# This reads your .env file and makes variables available
load_dotenv()

# Step 3: Verify API key is loaded (don't print the actual key!)
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print("✓ API key loaded successfully!")
    print(f"  Key starts with: {api_key[:7]}...")  # Only show first 7 chars
else:
    print("✗ API key not found!")
    print("  Make sure you have a .env file with OPENAI_API_KEY=your-key")

# Step 4: Initialize the LLM
# model: which AI model to use
# temperature: 0 = focused/deterministic, 1 = creative/random
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

print("✓ LLM initialized successfully!")
print(f"  Model: {llm.model_name}")
```

**Expected Output:**
```
✓ API key loaded successfully!
  Key starts with: sk-proj...
✓ LLM initialized successfully!
  Model: gpt-4o-mini
```

### Demo 2: Making Your First LLM Call

```python
# File: 02_first_call_demo.py
"""
INSTRUCTOR DEMO: Making your first LLM call
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# Setup
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Method 1: Simple invoke with string
print("=" * 50)
print("METHOD 1: Simple string input")
print("=" * 50)

response = llm.invoke("What is 2 + 2? Reply in one word.")
print(f"Response: {response.content}")
print(f"Response type: {type(response)}")

# Method 2: Using message objects (more control)
print("\n" + "=" * 50)
print("METHOD 2: Using HumanMessage")
print("=" * 50)

message = HumanMessage(content="Explain Python in exactly 10 words.")
response = llm.invoke([message])
print(f"Response: {response.content}")

# Method 3: Multiple messages (conversation)
print("\n" + "=" * 50)
print("METHOD 3: Conversation style")
print("=" * 50)

from langchain.schema import SystemMessage

messages = [
    SystemMessage(content="You are a helpful coding tutor. Be concise."),
    HumanMessage(content="What is a variable?")
]
response = llm.invoke(messages)
print(f"Response: {response.content}")
```

**Expected Output:**
```
==================================================
METHOD 1: Simple string input
==================================================
Response: Four.
Response type: <class 'langchain_core.messages.ai.AIMessage'>

==================================================
METHOD 2: Using HumanMessage
==================================================
Response: Python is a readable, versatile programming language for many applications.

==================================================
METHOD 3: Conversation style
==================================================
Response: A variable is a named container that stores data values in your program.
```

### Demo 3: Understanding Responses

```python
# File: 03_response_demo.py
"""
INSTRUCTOR DEMO: Understanding LLM responses
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Make a call
response = llm.invoke("What is LangChain?")

# Explore the response object
print("RESPONSE OBJECT EXPLORATION")
print("=" * 50)

# The main content
print(f"1. Content (the answer):")
print(f"   {response.content[:100]}...")

# Response metadata
print(f"\n2. Response type:")
print(f"   {type(response).__name__}")

# Token usage (important for cost tracking!)
print(f"\n3. Response metadata:")
print(f"   {response.response_metadata}")

# Additional info
print(f"\n4. All available attributes:")
for attr in ['content', 'type', 'id', 'response_metadata']:
    if hasattr(response, attr):
        print(f"   - {attr}")
```

---

## Part 3: DO IT - Guided Practice

Now it's your turn! Follow these exercises step by step.

### Exercise 1: Environment Setup (10 minutes)

**Task:** Set up your development environment.

```python
# File: exercises/ex1_setup.py
"""
EXERCISE 1: Set up your environment
Follow each TODO step carefully.
"""

# TODO 1: Create a .env file in your project root
# It should contain: OPENAI_API_KEY=your-actual-key-here

# TODO 2: Import the required libraries
# Hint: You need os, load_dotenv from dotenv, and ChatOpenAI from langchain_openai
# YOUR CODE HERE:


# TODO 3: Load environment variables
# YOUR CODE HERE:


# TODO 4: Check if API key is loaded
# YOUR CODE HERE:
api_key = ___  # Fill in the function to get the environment variable

if api_key:
    print("✓ Setup successful!")
else:
    print("✗ Setup failed - check your .env file")
```

<details>
<summary>Click to see solution</summary>

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print("✓ Setup successful!")
else:
    print("✗ Setup failed - check your .env file")
```
</details>

### Exercise 2: Your First Call (10 minutes)

**Task:** Make an LLM call and print the response.

```python
# File: exercises/ex2_first_call.py
"""
EXERCISE 2: Make your first LLM call
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# TODO 1: Initialize the LLM with model="gpt-4o-mini" and temperature=0
# YOUR CODE HERE:
llm = ___

# TODO 2: Ask the LLM "What is Python?" and store the response
# YOUR CODE HERE:
response = ___

# TODO 3: Print only the content of the response (not the whole object)
# Hint: Use response.content
# YOUR CODE HERE:
print(___)
```

<details>
<summary>Click to see solution</summary>

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
response = llm.invoke("What is Python?")
print(response.content)
```
</details>

### Exercise 3: Temperature Experiment (10 minutes)

**Task:** Understand how temperature affects responses.

```python
# File: exercises/ex3_temperature.py
"""
EXERCISE 3: Experiment with temperature settings
Run this multiple times and observe the differences!
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

question = "Give me a creative name for a coffee shop."

# TODO 1: Create an LLM with temperature=0 (deterministic)
llm_focused = ChatOpenAI(model="gpt-4o-mini", temperature=___)

# TODO 2: Create an LLM with temperature=1 (creative)
llm_creative = ChatOpenAI(model="gpt-4o-mini", temperature=___)

# TODO 3: Ask both LLMs the same question
print("TEMPERATURE = 0 (Focused/Consistent)")
print("-" * 40)
for i in range(3):
    response = llm_focused.invoke(question)
    print(f"  Attempt {i+1}: {response.content}")

print("\nTEMPERATURE = 1 (Creative/Varied)")
print("-" * 40)
for i in range(3):
    response = llm_creative.invoke(question)
    print(f"  Attempt {i+1}: {response.content}")

# TODO 4: Answer this question in a comment:
# What pattern do you notice between temperature=0 and temperature=1?
# YOUR OBSERVATION:
```

<details>
<summary>Click to see solution and explanation</summary>

```python
llm_focused = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_creative = ChatOpenAI(model="gpt-4o-mini", temperature=1)
```

**Observation:** With temperature=0, you get the same (or very similar) answer every time. With temperature=1, you get different creative answers each time.

**Use temperature=0 when:** You want consistent, predictable answers (facts, code, analysis)
**Use temperature=1 when:** You want variety and creativity (brainstorming, writing)
</details>

---

## Part 4: PROVE IT - Lesson Assignment

### Assignment 1.1: Build a Configurable LLM Client

**Objective:** Create a reusable function that initializes an LLM with custom settings.

**Requirements:**
1. Create a function called `create_llm` that accepts:
   - `model` (default: "gpt-4o-mini")
   - `temperature` (default: 0)
2. The function should verify the API key exists
3. Return the initialized LLM or raise an error if setup fails
4. Add a test that demonstrates the function works

**Starter Code:**
```python
# File: assignments/assignment_1_1.py
"""
ASSIGNMENT 1.1: Build a Configurable LLM Client

Complete the function below and test it.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def create_llm(model: str = "gpt-4o-mini", temperature: float = 0) -> ChatOpenAI:
    """
    Create and return a configured LLM instance.

    Args:
        model: The model name to use
        temperature: Creativity setting (0-1)

    Returns:
        Configured ChatOpenAI instance

    Raises:
        ValueError: If API key is not configured
    """
    # YOUR CODE HERE
    # Step 1: Load environment variables

    # Step 2: Check if API key exists

    # Step 3: Create and return the LLM

    pass  # Remove this line when you implement the function


def test_create_llm():
    """Test the create_llm function."""
    print("Testing create_llm function...")

    # Test 1: Default settings
    print("\nTest 1: Default settings")
    llm = create_llm()
    response = llm.invoke("Say 'Hello, World!'")
    print(f"  Response: {response.content}")
    assert "Hello" in response.content, "Basic call failed"
    print("  ✓ Passed!")

    # Test 2: Custom temperature
    print("\nTest 2: Custom temperature")
    llm_creative = create_llm(temperature=0.8)
    response = llm_creative.invoke("Invent a word and define it.")
    print(f"  Response: {response.content[:100]}...")
    print("  ✓ Passed!")

    # Test 3: Error handling (temporarily break API key)
    print("\nTest 3: Error handling")
    original_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = ""
    try:
        llm = create_llm()
        print("  ✗ Failed - should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Correctly raised error: {e}")
    finally:
        if original_key:
            os.environ["OPENAI_API_KEY"] = original_key

    print("\n" + "=" * 50)
    print("All tests passed! Great job!")


if __name__ == "__main__":
    test_create_llm()
```

**Grading Rubric:**
| Criteria | Points |
|----------|--------|
| Function loads environment variables | 20 |
| Function checks for API key | 20 |
| Function creates LLM with correct parameters | 30 |
| All tests pass | 20 |
| Code is clean and well-commented | 10 |
| **Total** | **100** |

---

## Lesson Checkpoint

Before moving to the next lesson, verify:

- [ ] I can explain what LangChain does
- [ ] I understand why we use environment variables for API keys
- [ ] I can initialize a ChatOpenAI instance
- [ ] I know the difference between temperature=0 and temperature=1
- [ ] I completed Assignment 1.1

---

## Common Errors & Solutions

### Error: "API key not found"
```
Solution: Make sure your .env file:
1. Is in the project root directory
2. Contains: OPENAI_API_KEY=sk-your-key (no quotes, no spaces)
3. load_dotenv() is called before accessing the key
```

### Error: "ModuleNotFoundError: No module named 'langchain_openai'"
```
Solution: Install the package:
pip install langchain-openai
```

### Error: "RateLimitError"
```
Solution:
1. Wait a minute and try again
2. Check your OpenAI account has credits
3. Check your usage limits at platform.openai.com
```

---

## Next Lesson Preview

In **Lesson 1.2: Prompt Templates**, you'll learn:
- How to create reusable prompt templates
- Using variables in prompts
- Building professional-quality prompts

**Continue to:** `tutorials/module_01_fundamentals/lesson_02_prompts.md`
