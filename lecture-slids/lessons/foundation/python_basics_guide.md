# Python Basics for Machine Learning
## A Self-Study Guide for MPS311/439

**Dr. Wei Xing - University of Sheffield**

---

## Welcome!

If you're reading this, you probably felt a bit lost in Lesson 1's lab. That's completely normal, and you're in the right place. This guide will help you understand the Python basics you need for machine learning - nothing more, nothing less.

**Important**: You don't need to memorize any of this. The goal is to understand concepts so that when you see them in labs, they make sense. Think of this as a reference guide, not a textbook to study.

**How to use this guide**:
1. Work through it in Google Colab (you'll learn by doing)
2. Use the AI prompts provided to get instant help
3. Take your time - there's no rush
4. Come back to sections when you need a refresher

Let's start from the very beginning.

---

## Part 1: Getting Started in Google Colab

### Opening Google Colab

1. Go to **colab.research.google.com** in your browser
2. Sign in with your Google account
3. Click **"New notebook"**
4. You now have a blank code cell ready to use

### What is a code cell?

A code cell is like a calculator - you type something in, press the play button (or Shift+Enter), and it shows you the result. Let's try it.

**Your first code**: Type this in the code cell and press Shift+Enter:

```python
print("Hello, I'm learning Python!")
```

**What happened?** The computer printed your message below the cell. Congratulations - you just wrote your first Python code!

### Using AI to Learn: Your New Superpower

Here's a secret: professional programmers use AI to help them code every single day. You should too! In Google Colab, you can use ChatGPT (in another browser tab) to help you understand anything.

**Try this right now**: Open ChatGPT and paste this prompt:

```
I'm learning Python for a machine learning course. I just wrote my first 
line: print("Hello, I'm learning Python!")

Can you explain what this line does in very simple terms, like I've never 
programmed before? Then show me 2 more examples of print statements I could try.
```

**What to do with AI's response**:
1. Read the explanation - does it make sense?
2. Copy the examples it gives you into new code cells in Colab
3. Run them (Shift+Enter) and see what happens
4. If something confuses you, ask AI follow-up questions

**Remember**: AI is your patient tutor. Ask it to explain things multiple times in different ways until you understand.

---

## Part 2: Variables - Storing Information

### What is a variable?

Think of a variable like a labeled box where you store information. You give it a name and put something inside.

**Try this in Colab**:

```python
age = 21
name = "Alex"
print(age)
print(name)
```

**What happened?**: 
- You created a box called `age` and put the number 21 in it
- You created a box called `name` and put the text "Alex" in it
- You printed what's in each box

### Using AI to Practice

**Paste this in ChatGPT**:

```
I'm learning about Python variables. Can you:
1. Explain why we use variables (give me a real-world analogy)
2. Give me 3 examples of creating variables with different types of information
3. Show me what happens if I try to use a variable before creating it

Keep explanations very simple please!
```

**Try the examples ChatGPT gives you in Colab.** Run each one and see what happens.

### Common Mistake: Variable Names

Variable names can't have spaces. This doesn't work:

```python
my age = 21  # This will give an error!
```

This works:

```python
my_age = 21  # Use underscore instead of space
myAge = 21   # Or capitalize the second word
```

**AI Prompt if you get confused**:

```
I got an error with my variable name. Here's my code:
[paste your code]

What's wrong and how do I fix it?
```

---

## Part 3: Basic Math and Numbers

Python can do math just like a calculator.

**Try these in Colab**:

```python
# Addition
result = 10 + 5
print(result)

# Subtraction
result = 10 - 5
print(result)

# Multiplication
result = 10 * 5
print(result)

# Division
result = 10 / 5
print(result)
```

**Note**: Lines starting with `#` are comments - Python ignores them. They're notes for humans reading the code.

### Using Variables in Math

You can do math with variables too:

```python
price = 100
discount = 20
final_price = price - discount
print(final_price)
```

**Practice with AI**:

```
Give me 5 simple math problems I can solve in Python using variables. 
Make them practical examples (like calculating discounts, areas, etc).
Show me the code for the first one, then just describe the others so 
I can try coding them myself.
```

---

## Part 4: Lists - Collections of Things

Sometimes you need to store multiple related items. That's what lists are for.

**Try this**:

```python
# A list of numbers
temperatures = [20, 22, 19, 21, 23]
print(temperatures)

# A list of text
cities = ["Sheffield", "London", "Manchester"]
print(cities)
```

### Getting Items from a List

Lists are numbered starting from 0 (yes, zero!):

```python
cities = ["Sheffield", "London", "Manchester"]
print(cities[0])  # Prints "Sheffield" (first item)
print(cities[1])  # Prints "London" (second item)
print(cities[2])  # Prints "Manchester" (third item)
```

**Why start from 0?** It's a computer science thing. Just remember: first item is [0], second is [1], third is [2].

### AI Help with Lists

**When you're confused about lists**:

```
I'm working with Python lists and I'm confused about:
[describe what confuses you]

Can you explain using a simple real-world example, then show me 
code that demonstrates the concept?
```

**Common list operations to ask AI about**:
- "How do I add an item to a list in Python?"
- "How do I find out how many items are in a list?"
- "What does list[0] mean and why does counting start at 0?"

---

## Part 5: Understanding Methods - The Dot Notation

This is where Lesson 1 got confusing for many students. Let's break it down.

### What is a method?

A method is an action you can perform on something. You write it with a dot: `thing.action()`

**Example with a list**:

```python
numbers = [3, 1, 4, 1, 5]
numbers.sort()  # This sorts the list
print(numbers)  # Now prints [1, 1, 3, 4, 5]
```

**What happened?**: 
- `numbers` is the list
- `.sort()` is the action (method)
- The dot connects the list to the action

### Common Methods You'll See in Labs

```python
# With text (called "strings")
message = "hello"
message.upper()  # Makes it uppercase: "HELLO"

# With lists
my_list = [1, 2, 3]
my_list.append(4)  # Adds 4 to the end: [1, 2, 3, 4]
```

### Using AI to Understand Methods

**This is crucial**: When you see a method you don't understand in the lab, immediately ask AI:

```
I see this code in my lab:
[paste the confusing line]

Can you:
1. Explain what the .something() part does in simple terms
2. Show me a simpler example of the same method
3. Tell me what the result will be
```

**Practice right now**:

Copy this code into Colab:

```python
cities = ["sheffield", "london", "manchester"]
cities.sort()
print(cities)
```

Then ask ChatGPT to explain what `.sort()` does and why the output looks the way it does.

---

## Part 6: Understanding Error Messages (Don't Panic!)

Errors are normal. Professional programmers see errors all day long. The key is learning to read them.

### Common Error Types

**NameError** - You used something that doesn't exist:

```python
print(x)  # Error! We never created 'x'
```

**Fix**: Create the variable first:

```python
x = 10
print(x)  # Works!
```

**SyntaxError** - You made a typo or forgot something:

```python
print("hello"  # Error! Missing closing )
```

### Using AI to Fix Errors

**This is the most powerful AI use**: When you get an error, don't panic. Use this exact process:

1. **Copy the entire error message** (the red text)
2. **Copy your code** (the part that's not working)
3. **Paste this into ChatGPT**:

```
I'm learning Python and got this error:
[paste the entire error message]

Here's my code:
[paste your code]

Can you:
1. Explain what the error means in simple terms
2. Show me exactly what I need to fix
3. Explain why the error happened so I don't make the same mistake again
```

**Try this right now**: Deliberately create an error in Colab (like forgetting a closing parenthesis), then practice using AI to fix it.

---

## Part 7: Working in Google Colab - Practical Tips

### Running Cells

- **Shift+Enter**: Run current cell and move to next one
- **Ctrl+Enter**: Run current cell and stay on it
- Click the ▶ play button: Run the cell

### Adding New Cells

- **+Code** button at the top
- Or hover between cells and click **+Code** that appears

### If Something Breaks

Sometimes code stops working. Here's what to do:

1. **Runtime → Restart runtime** (at the top menu)
2. Run your cells again from the beginning
3. If still broken, ask AI for help

### Saving Your Work

Colab auto-saves to your Google Drive, but it's good practice to:
- **File → Save** regularly
- Give your notebook a meaningful name

---

## Part 8: Essential Skills for Our Labs

Here are the specific things you'll need to understand for our machine learning labs. Don't try to memorize - just understand the concepts.

### 1. Importing Libraries

This loads tools we need:

```python
import pandas as pd
```

**What it means**: "Load the pandas tool and let me call it 'pd' for short"

**AI prompt when you see imports**:

```
I see this import statement:
[paste the import]

What is this library used for? Keep it simple and relevant to 
machine learning and data analysis.
```

### 2. Loading Data

```python
data = pd.read_csv("file.csv")
```

**What it means**: "Read data from a file and store it in a variable called 'data'"

### 3. Looking at Data

```python
data.head()  # Shows first few rows
data.shape   # Shows how big the data is
```

### 4. Asking About Data

```python
data['column_name'].value_counts()  # Counts different values
```

**Strategy**: When you see code like this in labs, break it down with AI:

```
Break down this code line-by-line:
[paste the code]

Explain each part separately:
- What is data?
- What does ['column_name'] mean?
- What does .value_counts() do?
- What will the result look like?
```

---

## Part 9: Your AI Learning Strategy

Here's how to use AI effectively for this course:

### Good AI Habits

**1. Be specific about your level**:
```
I'm a complete beginner learning Python for machine learning. 
[your question]
Explain like I've never programmed before.
```

**2. Ask for examples**:
```
Show me 3 different examples of [concept], from simplest to 
slightly more complex.
```

**3. Ask "why" not just "how"**:
```
Not just how to do [X], but why would I want to do [X] in 
machine learning?
```

**4. Request step-by-step breakdowns**:
```
Break this code down line by line and explain what each part does:
[paste code]
```

**5. Learn from errors**:
```
I got this error: [paste error]
My code: [paste code]
Explain what went wrong and why, then show me the fix.
```

### Practice Prompts for Common Lab Situations

**When code is confusing**:
```
I'm looking at this code from my lab and it's confusing:
[paste code]

Can you:
1. Explain what this code is trying to accomplish overall
2. Break down each line
3. Show me a simpler version that does the same thing
```

**When you need to modify code**:
```
I have this working code:
[paste code]

I need to change it to [describe what you want instead].
Show me what to change and explain why.
```

**When starting an exercise**:
```
My lab asks me to: [paste the exercise instructions]

I'm not sure where to start. Can you:
1. Outline the steps I need to take (but don't give me the full code)
2. Give me a hint for step 1
3. Let me try, then I'll come back for the next step
```

---

## Part 10: Next Steps

### What to Do Now

1. **Work through this guide in Colab** - actually type and run the examples
2. **Practice with AI** - try the prompts suggested throughout
3. **Review Lesson 1 lab** - try it again with AI help, but don't worry about finishing everything
4. **Come to office hours** if anything is still confusing (Tuesday 12-1pm, Hicks I22)

### Optional Python Resources

If you want more practice (totally optional):

- **Think Python**: https://allendowney.github.io/ThinkPython/ (free online, starts from zero)
- **Beginning Python by Magnus Hetland**: Available through the library
- **Python Tutor**: http://pythontutor.com (visualizes code execution)

### Remember

- You don't need to be a Python expert for this course
- AI is your learning partner - use it constantly
- Errors are normal and fixable
- Ask questions - in labs, office hours, or the forum
- It gets easier with practice

### Most Important

**You can do this.** The Python is just a tool. The machine learning concepts are what matter, and we'll teach those clearly. Focus on understanding, not memorizing, and use AI to help you learn.

See you in the labs!

---

**Dr. Wei Xing**  
w.xing@sheffield.ac.uk  
Office Hours: Tuesday 12-1pm, Hicks Building I22