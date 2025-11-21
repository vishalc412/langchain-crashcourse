# Module 1, Lesson 2: Prompt Templates & Formatting

## Lesson Overview
| Duration | Difficulty | Prerequisites |
|----------|------------|---------------|
| 50 minutes | Beginner | Lesson 1.1 completed |

## Learning Objectives
By the end of this lesson, you will be able to:
- [ ] Create prompt templates with variables
- [ ] Use different message types (System, Human, AI)
- [ ] Format prompts for different use cases
- [ ] Build reusable prompt libraries

---

## Part 1: LEARN IT - Understanding Prompt Templates

### Why Use Prompt Templates?

Without templates, you'd write prompts like this:

```python
# ❌ Hard to maintain, prone to errors
prompt = "You are a " + role + ". Answer this question about " + topic + ": " + question
```

With templates:

```python
# ✓ Clean, reusable, maintainable
template = "You are a {role}. Answer this question about {topic}: {question}"
```

### The Anatomy of a Good Prompt

```
┌─────────────────────────────────────────────────────────────┐
│                    PROMPT STRUCTURE                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ SYSTEM MESSAGE (Who is the AI?)                      │    │
│  │ "You are a helpful Python tutor who explains        │    │
│  │  concepts simply with examples."                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ CONTEXT (What should AI know?)                       │    │
│  │ "The student is a beginner learning their first     │    │
│  │  programming language."                              │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ INSTRUCTION (What should AI do?)                     │    │
│  │ "Explain the following concept in 3 simple          │    │
│  │  sentences, then provide a code example."           │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ INPUT (What is the specific request?)                │    │
│  │ "Concept: {concept}"                                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ OUTPUT FORMAT (How should AI respond?)               │    │
│  │ "Format your response as:                           │    │
│  │  Explanation: ...                                    │    │
│  │  Example: ..."                                       │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Message Types in LangChain

| Type | Purpose | Example |
|------|---------|---------|
| `SystemMessage` | Sets AI behavior/personality | "You are a helpful assistant" |
| `HumanMessage` | User's input | "What is Python?" |
| `AIMessage` | AI's response (for history) | "Python is a programming language..." |

---

## Part 2: SEE IT - Instructor Demo

### Demo 1: Basic Prompt Template

```python
# File: demos/01_basic_template.py
"""
INSTRUCTOR DEMO: Creating basic prompt templates
"""

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create a simple template with ONE variable
print("DEMO 1: Simple Template with One Variable")
print("=" * 50)

template = ChatPromptTemplate.from_template(
    "Translate the following English text to {language}: {text}"
)

# See what the template looks like
print(f"Template variables: {template.input_variables}")

# Format the template with actual values
formatted = template.format(language="Spanish", text="Hello, how are you?")
print(f"Formatted prompt: {formatted}")

# Use with LLM
chain = template | llm
response = chain.invoke({"language": "Spanish", "text": "Hello, how are you?"})
print(f"Response: {response.content}")
```

**Output:**
```
DEMO 1: Simple Template with One Variable
==================================================
Template variables: ['language', 'text']
Formatted prompt: Translate the following English text to Spanish: Hello, how are you?
Response: Hola, ¿cómo estás?
```

### Demo 2: Multi-Message Template

```python
# File: demos/02_multi_message.py
"""
INSTRUCTOR DEMO: Templates with System and Human messages
"""

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print("DEMO 2: Multi-Message Template")
print("=" * 50)

# Create template with system message (AI personality) and human message
template = ChatPromptTemplate.from_messages([
    ("system", "You are a {profession} with {years} years of experience. "
               "Answer questions in a {tone} manner."),
    ("human", "{question}")
])

# See the structure
print(f"Template variables: {template.input_variables}")

# Use the template
chain = template | llm

response = chain.invoke({
    "profession": "senior Python developer",
    "years": "10",
    "tone": "friendly and encouraging",
    "question": "How do I learn Python effectively?"
})

print(f"\nResponse:\n{response.content}")
```

### Demo 3: Reusable Prompt Library

```python
# File: demos/03_prompt_library.py
"""
INSTRUCTOR DEMO: Building a reusable prompt library
"""

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print("DEMO 3: Prompt Library")
print("=" * 50)

# Create a library of reusable prompts
class PromptLibrary:
    """Collection of reusable prompts for common tasks."""

    @staticmethod
    def summarizer():
        """Prompt for summarizing text."""
        return ChatPromptTemplate.from_messages([
            ("system", "You are an expert summarizer. Create concise, "
                      "accurate summaries that capture key points."),
            ("human", "Summarize the following text in {num_sentences} sentences:\n\n{text}")
        ])

    @staticmethod
    def code_explainer():
        """Prompt for explaining code."""
        return ChatPromptTemplate.from_messages([
            ("system", "You are a patient coding teacher. Explain code clearly "
                      "for {skill_level} programmers."),
            ("human", "Explain this {language} code:\n```{language}\n{code}\n```")
        ])

    @staticmethod
    def email_writer():
        """Prompt for writing professional emails."""
        return ChatPromptTemplate.from_messages([
            ("system", "You are a professional communication expert. "
                      "Write clear, {tone} emails."),
            ("human", "Write an email about: {topic}\nKey points: {key_points}")
        ])


# Use the library
library = PromptLibrary()

# Example 1: Summarizer
print("\n1. Using Summarizer Prompt:")
print("-" * 40)
chain = library.summarizer() | llm
response = chain.invoke({
    "num_sentences": "2",
    "text": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves."
})
print(f"Summary: {response.content}")

# Example 2: Code Explainer
print("\n2. Using Code Explainer Prompt:")
print("-" * 40)
chain = library.code_explainer() | llm
response = chain.invoke({
    "skill_level": "beginner",
    "language": "python",
    "code": "for i in range(5):\n    print(i)"
})
print(f"Explanation: {response.content}")
```

---

## Part 3: DO IT - Guided Practice

### Exercise 1: Create Your First Template (10 minutes)

**Task:** Create a template for generating product descriptions.

```python
# File: exercises/ex1_product_template.py
"""
EXERCISE 1: Create a product description template
"""

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# TODO 1: Create a template with these variables:
# - product_name: Name of the product
# - target_audience: Who the product is for
# - key_features: Main features of the product
# - tone: Writing style (professional, casual, exciting, etc.)

template = ChatPromptTemplate.from_messages([
    ("system", "___"),  # Fill in: describe AI's role
    ("human", "___")    # Fill in: the actual request with variables
])

# TODO 2: Test your template with this product:
test_input = {
    "product_name": "SmartFit Watch Pro",
    "target_audience": "fitness enthusiasts",
    "key_features": "heart rate monitoring, GPS tracking, 7-day battery",
    "tone": "exciting"
}

# TODO 3: Create chain and invoke
chain = ___ | ___
response = chain.invoke(___)

print("Product Description:")
print(response.content)
```

<details>
<summary>Click to see solution</summary>

```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are a skilled marketing copywriter. Write compelling "
               "product descriptions that are {tone} and appeal to the target audience."),
    ("human", """Write a product description for:
Product: {product_name}
Target Audience: {target_audience}
Key Features: {key_features}

Keep it under 100 words.""")
])

chain = template | llm
response = chain.invoke(test_input)
```
</details>

### Exercise 2: Multi-Role Template (15 minutes)

**Task:** Create a template that can switch between different expert roles.

```python
# File: exercises/ex2_multi_role.py
"""
EXERCISE 2: Create a multi-role expert template
The same template should work for different expert types.
"""

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# TODO 1: Create a template that works for ANY expert type
# Variables needed: expert_type, expertise_area, user_question

template = ChatPromptTemplate.from_messages([
    ("system", "___"),  # The AI should act as {expert_type}
    ("human", "___")    # The question about {expertise_area}
])

# TODO 2: Test with different experts
test_cases = [
    {
        "expert_type": "nutritionist",
        "expertise_area": "healthy eating",
        "user_question": "What's a good breakfast for energy?"
    },
    {
        "expert_type": "financial advisor",
        "expertise_area": "personal finance",
        "user_question": "How should a beginner start investing?"
    },
    {
        "expert_type": "fitness coach",
        "expertise_area": "exercise",
        "user_question": "What's a simple workout routine for beginners?"
    }
]

chain = template | llm

for i, test in enumerate(test_cases, 1):
    print(f"\n{'='*50}")
    print(f"TEST {i}: {test['expert_type'].upper()}")
    print(f"{'='*50}")
    response = chain.invoke(test)
    print(f"Q: {test['user_question']}")
    print(f"A: {response.content[:200]}...")
```

<details>
<summary>Click to see solution</summary>

```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are an experienced {expert_type} specializing in {expertise_area}. "
               "Provide helpful, accurate advice. Keep responses concise but informative."),
    ("human", "{user_question}")
])
```
</details>

### Exercise 3: Structured Output Template (15 minutes)

**Task:** Create a template that produces consistently structured output.

```python
# File: exercises/ex3_structured.py
"""
EXERCISE 3: Create a template for structured output
The output should always follow a specific format.
"""

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# TODO 1: Create a template that analyzes a topic and returns
# output in this EXACT format:
#
# TOPIC: [topic name]
# SUMMARY: [2-3 sentence summary]
# KEY POINTS:
# 1. [point 1]
# 2. [point 2]
# 3. [point 3]
# DIFFICULTY: [Beginner/Intermediate/Advanced]
# LEARN MORE: [one resource suggestion]

template = ChatPromptTemplate.from_messages([
    ("system", """You are an educational content analyzer.
Always respond in this EXACT format:

TOPIC: [topic name]
SUMMARY: [2-3 sentence summary]
KEY POINTS:
1. [point 1]
2. [point 2]
3. [point 3]
DIFFICULTY: [Beginner/Intermediate/Advanced]
LEARN MORE: [one resource suggestion]"""),
    ("human", "Analyze this topic for someone learning programming: {topic}")
])

# TODO 2: Test with these topics
topics = ["Variables", "Functions", "Object-Oriented Programming"]

chain = template | llm

for topic in topics:
    print(f"\n{'='*50}")
    response = chain.invoke({"topic": topic})
    print(response.content)
```

---

## Part 4: PROVE IT - Lesson Assignment

### Assignment 1.2: Build a Prompt Template Library

**Objective:** Create a comprehensive library of prompt templates for a coding assistant application.

**Requirements:**
1. Create a `CodingAssistantPrompts` class with at least 4 templates:
   - `code_reviewer`: Reviews code and suggests improvements
   - `bug_finder`: Identifies potential bugs in code
   - `code_generator`: Generates code from descriptions
   - `code_documenter`: Adds documentation to code

2. Each template must:
   - Use appropriate system messages
   - Have clear, descriptive variable names
   - Include format instructions in the prompt

3. Create a test function that demonstrates each template

**Starter Code:**
```python
# File: assignments/assignment_1_2.py
"""
ASSIGNMENT 1.2: Build a Coding Assistant Prompt Library

Create a comprehensive library of prompts for a coding assistant.
"""

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


class CodingAssistantPrompts:
    """Library of prompts for a coding assistant application."""

    @staticmethod
    def code_reviewer() -> ChatPromptTemplate:
        """
        Template for reviewing code quality.
        Variables: language, code
        """
        # YOUR CODE HERE
        pass

    @staticmethod
    def bug_finder() -> ChatPromptTemplate:
        """
        Template for identifying bugs.
        Variables: language, code, description (what the code should do)
        """
        # YOUR CODE HERE
        pass

    @staticmethod
    def code_generator() -> ChatPromptTemplate:
        """
        Template for generating code from descriptions.
        Variables: language, task_description, requirements
        """
        # YOUR CODE HERE
        pass

    @staticmethod
    def code_documenter() -> ChatPromptTemplate:
        """
        Template for adding documentation.
        Variables: language, code, doc_style (Google, NumPy, etc.)
        """
        # YOUR CODE HERE
        pass


def test_prompts():
    """Test all prompts in the library."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    library = CodingAssistantPrompts()

    # Test 1: Code Reviewer
    print("TEST 1: Code Reviewer")
    print("=" * 50)
    chain = library.code_reviewer() | llm
    response = chain.invoke({
        "language": "python",
        "code": """
def calc(x,y):
    return x+y
"""
    })
    print(response.content[:300])

    # Test 2: Bug Finder
    print("\n\nTEST 2: Bug Finder")
    print("=" * 50)
    chain = library.bug_finder() | llm
    response = chain.invoke({
        "language": "python",
        "code": """
def divide_numbers(a, b):
    return a / b
""",
        "description": "Safely divide two numbers"
    })
    print(response.content[:300])

    # Test 3: Code Generator
    print("\n\nTEST 3: Code Generator")
    print("=" * 50)
    chain = library.code_generator() | llm
    response = chain.invoke({
        "language": "python",
        "task_description": "Check if a string is a palindrome",
        "requirements": "Handle case insensitivity and ignore spaces"
    })
    print(response.content[:400])

    # Test 4: Code Documenter
    print("\n\nTEST 4: Code Documenter")
    print("=" * 50)
    chain = library.code_documenter() | llm
    response = chain.invoke({
        "language": "python",
        "code": """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
""",
        "doc_style": "Google"
    })
    print(response.content[:400])

    print("\n\n" + "=" * 50)
    print("All tests completed!")


if __name__ == "__main__":
    test_prompts()
```

**Grading Rubric:**
| Criteria | Points |
|----------|--------|
| code_reviewer template is well-designed | 20 |
| bug_finder template identifies issues effectively | 20 |
| code_generator produces working code | 20 |
| code_documenter creates proper documentation | 20 |
| Code is clean, well-organized | 10 |
| All tests run successfully | 10 |
| **Total** | **100** |

---

## Lesson Checkpoint

Before moving to the next lesson, verify:

- [ ] I can create prompt templates with variables
- [ ] I understand the difference between system and human messages
- [ ] I can build reusable prompt libraries
- [ ] I know how to format prompts for consistent output
- [ ] I completed Assignment 1.2

---

## Quick Reference: Prompt Template Patterns

```python
# Pattern 1: Simple template
ChatPromptTemplate.from_template("Question: {question}")

# Pattern 2: Multi-message
ChatPromptTemplate.from_messages([
    ("system", "You are {role}"),
    ("human", "{question}")
])

# Pattern 3: With examples (few-shot)
ChatPromptTemplate.from_messages([
    ("system", "Translate to {language}"),
    ("human", "Hello"),
    ("ai", "Hola"),  # Example
    ("human", "{text}")  # Actual input
])
```

---

## Next Lesson Preview

In **Lesson 1.3: Chains**, you'll learn:
- How to connect multiple LLM calls together
- Sequential processing of data
- Building complex workflows

**Continue to:** `tutorials/module_01_fundamentals/lesson_03_chains.md`
