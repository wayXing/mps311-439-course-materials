---
theme: default
background: https://cover.sli.dev
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  ## Lesson 1: Welcome to Machine Learning
  MPS311/439 Machine Learning - Dr. Wei Xing
drawings:
  persist: false
transition: slide-left
title: 'Lesson 1: Welcome to Machine Learning'
routerMode: hash
mdc: true
---
# Lesson 1: Welcome to Machine Learning

## Course Roadmap, Tools, and Your First ML Workflow

<div class="pt-12">
  <span class="text-xl">
    MPS311/439 Machine Learning<br>
    Dr. Wei Xing<br>
    2026–27
  </span>
</div>

---

# Raise Your Hand If...

<div class="text-xl space-y-4">

- 📱 Netflix recommended something you actually watched
- 🛒 Amazon suggested something you bought  
- 🎵 Spotify found you a new favorite song
- 📧 Gmail filtered spam correctly today
- 🗺️ Google Maps found you the fastest route

</div>

<div class="text-center mt-8 text-2xl text-blue-600">
<strong>That's Machine Learning changing your life</strong>
</div>

---

# Your Instructor: Dr. Wei Xing

<div class="grid grid-cols-2 gap-8">

<div>

## Background
- **Research**: ML applications in Engineering Design and Optimization
- **Passion**: Making ML accessible to everyone
- **Office**: Hicks Building I22
- **Email**: w.xing@sheffield.ac.uk
- **Website**: wxing.me

</div>

<div>

## My Promise to You
- ✅ **Practical focus** over mathematical proofs
- ✅ **AI tools welcome** (ChatGPT, Copilot)
- ✅ **Real support** - I want you to succeed
- ✅ **No silly questions** - everyone learns differently

</div>

</div>

---

# Course Philosophy: Your Success First

<div class="text-center text-xl mb-8">
"Learning should be challenging but not overwhelming"
</div>

<div class="grid grid-cols-3 gap-6">

<div class="bg-blue-50 p-4 rounded">
<h3 class="text-blue-800 font-bold">🎯 Core for All</h3>
Essential concepts everyone needs
</div>

<div class="bg-green-50 p-4 rounded">
<h3 class="text-green-800 font-bold">🚀 Advanced for 439</h3>
Extra challenges for those ready
</div>

<div class="bg-purple-50 p-4 rounded">
<h3 class="text-purple-800 font-bold">🤝 AI-Assisted</h3>
Use modern tools to learn faster
</div>

</div>

- Learning by doing!
- Problem-oriented thinking!
- Learn the methodology NOT remember the code!
- Use AI wisely and efficiently!


<!-- <div class="my-8 flex flex-col items-center">

  <img src="https://upload.wikimedia.org/wikipedia/commons/4/4b/Artificial_intelligence_in_life_sciences.png" alt="AI in Life Sciences" class="w-2/3 rounded shadow mb-4" />

  <div class="text-lg text-gray-700 text-center max-w-2xl">
    <strong>Why Machine Learning?</strong><br>
    <ul class="list-disc list-inside text-left mt-2">
      <li>ML powers modern science, business, and daily life</li>
      <li>It helps us <span class="text-blue-700 font-semibold">find patterns</span> in complex data</li>
      <li>From <span class="text-green-700 font-semibold">healthcare</span> to <span class="text-purple-700 font-semibold">entertainment</span>, ML is everywhere</li>
    </ul>
    <div class="mt-4 italic text-gray-500">
      This course will help you understand, use, and even build ML systems!
    </div>
  </div>

</div> -->



---

# Quick Schedule Overview

<div class="grid grid-cols-2 gap-8">

<div>

## Session Pattern
- **Tuesday 11am**: Lecture (here!)
- **Friday 4pm**: Hands-on lab
- **Optional Workshops**: between Lessons 4–5 and Lessons 9–10

## Assessment Timeline
- **Lesson 6**: Assessment 1 due (35%)
- **20 January 2027**: Assessment 2 due (65%)
- **4 weeks**: Feedback turnaround

</div>

<div>

## Learning Journey
1. **Lessons 1–3**: Regression fundamentals
2. **Lessons 4–6**: Classification methods  
3. **Lessons 7–8**: Unsupervised learning
4. **Lessons 9–10**: Deep learning & CNNs

<div class="bg-yellow-50 p-3 rounded mt-4">
<strong>Today's goal</strong>: Get excited & get set up!
</div>

Detailed schedule: [MPS311/439 Schedule](https://docs.google.com/spreadsheets/d/1PqoSdwmew5_Cr4ZE7H4Xef_hs62oVcGrbIKkUWr-yfc/edit?usp=sharing)

</div>

</div>

---

# The Big Question: What IS Machine Learning?

<div class="text-center text-2xl mb-8">
Let's figure this out together...
</div>

<div class="bg-gray-50 p-6 rounded">

## Traditional Programming
**Input + Rules → Output**

Example: Calculate compound interest
- Input: Principal, rate, time
- Rules: A = P(1 + r)^t  
- Output: Final amount

</div>

---

# Machine Learning: Flipping the Script

<div class="bg-blue-50 p-6 rounded mb-6">

## Machine Learning  
**Input + Output → Rules**

Example: Predict house prices
- Input: Size, location, age, bedrooms...
- Output: Actual sale prices (training data)
- **ML finds the rules**: What makes prices high/low?

</div>

<div class="text-center text-lg text-blue-600">
<strong>We give examples, ML finds patterns</strong>
</div>

---

# The Machine Learning Workflow

<div class="grid grid-cols-4 gap-6 items-center">

<!-- Step 1: Define -->
<div class="flex flex-col items-center">
  <span class="text-4xl mb-2" aria-label="Target">🎯</span>
  <h3 class="font-bold text-blue-800 mb-1">1. Define</h3>
  <ul class="text-sm text-gray-700 text-center">
    <li>What do you want to predict?</li>
    <li>Identify input & output</li>
    <li>Is ML the right tool?</li>
  </ul>
</div>

<!-- Step 2: Collect -->
<div class="flex flex-col items-center">
  <span class="text-4xl mb-2" aria-label="Database">🗂️</span>
  <h3 class="font-bold text-green-800 mb-1">2. Collect</h3>
  <ul class="text-sm text-gray-700 text-center">
    <li>Get relevant data</li>
    <li>Check for quality</li>
    <li>Clean & prepare</li>
  </ul>
</div>

<!-- Step 3: Model & Train -->
<div class="flex flex-col items-center">
  <span class="text-4xl mb-2" aria-label="Robot">🤖</span>
  <h3 class="font-bold text-purple-800 mb-1">3. Model & Train</h3>
  <ul class="text-sm text-gray-700 text-center">
    <li>Choose a model</li>
    <li>Train on data</li>
    <li>Tune parameters</li>
  </ul>
</div>

<!-- Step 4: Validate -->
<div class="flex flex-col items-center">
  <span class="text-4xl mb-2" aria-label="Checkmark">✅</span>
  <h3 class="font-bold text-red-700 mb-1">4. Validate</h3>
  <ul class="text-sm text-gray-700 text-center">
    <li>Test on new data</li>
    <li>Measure performance</li>
    <li>Check accuracy</li>
  </ul>
</div>

</div>

<!-- <div class="mt-8 text-center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/0/00/Machine_learning_process_flowchart.png" alt="Informative diagram of the machine learning process, showing steps from data collection to model deployment" class="mx-auto w-2/3 rounded shadow border border-blue-100" />
  <div class="text-sm text-gray-500 mt-2">A typical machine learning workflow (source: Wikimedia Commons)</div>
</div> -->

<div class="mt-4 text-center text-blue-700 text-lg">
  <strong>ML is a cycle: define, collect, train, validate.</strong>
</div>


<!-- ---

# Live Demo: Pattern Recognition
## Let's See ML in Action

<div class="text-center text-xl mb-4">
🖥️ **Switching to Jupyter Notebook** 🖥️
</div>

<div class="bg-blue-50 p-6 rounded">

## Demo Goals:
- 📊 **Show real data** (house sizes vs prices)
- 🔧 **Adjust noise levels** live - see how ML handles messy data
- 🎯 **Make predictions** for new houses
- 📈 **Visualize the pattern** ML discovered

</div>

<div class="text-center mt-6 text-lg text-green-600">
<strong>Students: Watch how we go from "scattered dots" to "useful predictions"</strong>
</div> -->

---

# Why This Matters: ML is Everywhere

<div class="grid grid-cols-2 gap-6">

<div>

## You Already Use ML Daily
- 🎬 **Netflix**: "You might like..."
- 🛒 **Amazon**: Product recommendations  
- 📧 **Gmail**: Spam detection
- 🗺️ **Maps**: Traffic prediction
- 📱 **Photos**: Face recognition

</div>

<div>

## Industry Applications
- 💊 **Healthcare**: Drug discovery, diagnosis
- 🏦 **Finance**: Fraud detection, trading
- 🚗 **Transport**: Autonomous vehicles
- 🛡️ **Security**: Threat detection
- 🌱 **Climate**: Weather prediction

</div>

</div>

<div class="text-center mt-6 text-lg text-green-600">
<strong>This course: You'll build these systems</strong>
</div>

<!-- ---

# Interactive Demo: Multiple Patterns
## Different Problems, Different Solutions

<div class="text-center text-xl mb-4">
🖥️ **Back to Jupyter Notebook** 🖥️
</div>

<div class="bg-green-50 p-6 rounded">

## What We'll Show:
- 📈 **Regression**: Predicting house prices (continuous numbers)
- 🎯 **Classification**: Spam vs real emails (categories)  
- 🔍 **Clustering**: Finding customer groups (hidden patterns)

</div>

<div class="text-center mt-6 text-lg text-blue-600">
<strong>Same toolkit, different problems!</strong>
</div> -->

---

# Tools Tour: Your ML Toolkit

<div class="grid grid-cols-2 gap-8">

<div>

## Google Colab ⭐
- **Browser-based** Python notebooks
- **Free GPU access** for deep learning
- **No installation** needed
- **Easy sharing** for assignments

## Python Libraries
- **pandas**: Data manipulation
- **matplotlib**: Visualization  
- **scikit-learn**: ML algorithms
- **Keras**: Deep learning

</div>

<div>

## AI Assistants (Encouraged!)
- **ChatGPT**: Code help & debugging
- **GitHub Copilot**: Auto-completion
- **Claude**: Explanation & examples

## Course Support
- **Blackboard**: Materials & submission
- **Discussion Forum**: Peer help
- **Office Hours**: Tuesday 12-1pm
- **Workshops**: Extra support

</div>

</div>

---

# Live Colab Demo
## Let's Get You Set Up

<div class="bg-blue-50 p-6 rounded">

### Follow Along:
1. **Go to**: colab.research.google.com
2. **Sign in** with Google account  
3. **New Notebook** → "MPS311 Lesson 1"

</div>

<div class="bg-green-50 p-4 rounded mt-4">

## What We'll Test later on Friday's lab
- ✅ **Import key libraries** (pandas, matplotlib, sklearn)
- ✅ **Create your first plot**
- ✅ **Verify everything works**
- 🎉 **Celebrate your ML-ready environment!**

</div>

---

# AI Assistant Demo: Getting Help

<div class="bg-green-50 p-6 rounded">

## Try This ChatGPT Prompt:

*"I'm learning machine learning in Python. I have this error: 'NameError: name 'pd' is not defined'. What does this mean and how do I fix it?"*

### Good AI Assistant Habits:
- ✅ **Copy exact error messages**
- ✅ **Provide context** ("I'm trying to...")  
- ✅ **Ask for explanations**, not just fixes
- ✅ **Test the suggestions** and understand them
- ✅ **Chain of thought** ("Let's think step by step...")

### This Course Approach:
- AI tools are **learning aids**, not replacements
- Use them to **understand concepts** better
- Still need to **understand the fundamentals**

</div>

---

# What's Coming Up

<div class="grid grid-cols-2 gap-8">

<div>

## This Friday's Lab
- **Colab setup** & environment
- **Python refresher** (pandas, matplotlib)
- **Data exploration** with Netflix dataset
- **First taste** of data insights
- **AI debugging** practice

</div>

<div>

## Next Week: Linear Regression
- **Your first ML algorithm**
- **Predict continuous values**
- **Understand training vs testing**
- **Assessment 1 released**

<div class="bg-yellow-50 p-3 rounded mt-4">
<strong>Prep</strong>: Bring questions to Friday's lab!
</div>

</div>

</div>

---

# Your ML Journey Starts Now

<div class="text-center">

<div class="text-3xl mb-6">🎯</div>

## By End of This Course:
- **Build** recommendation systems
- **Predict** house prices, stock trends  
- **Classify** images with neural networks
- **Use** industry-standard tools confidently

<div class="bg-blue-50 p-4 rounded mt-8">
<strong>Remember</strong>: Every expert was once a beginner<br>
I'm here to help you succeed! 🚀
</div>

</div>

---

# Questions & Getting Help

<div class="grid grid-cols-2 gap-8">

<div>

## Quick Questions
- **Module Discussion Forum** on Blackboard
- Helps other students too!

## Personal/Private Issues  
- **Email**: w.xing@sheffield.ac.uk
- Response within 48 hours

## Urgent Technical Issues
- **Email** with "MPS311 URGENT" 

</div>

<div>

## Office Hours
- **Tuesday 12:00-1:00 pm**
- **Hicks Building I22**
- No appointment needed!

## What to Bring Friday
- ✅ Laptop/access to Colab
- ✅ Questions from today
- ✅ Excitement for hands-on coding!

</div>

</div>

<div class="text-center mt-8 text-xl text-blue-600">
<strong>See you Friday - let's code! 💻</strong>
</div>

<!-- COURSE_FEEDBACK_QR:START -->
---
layout: center
class: text-center
---

# 30-second feedback

<p class="text-xl mb-4">Scan this code to share anonymous feedback or post a question for this lecture.</p>

<img src="./feedback-qr.svg" alt="Feedback QR code for Lesson 01 lecture" class="w-44 mx-auto rounded-lg shadow" />

<p class="text-sm mt-4 opacity-70">MPS311/439 · 2026-27 · Lesson 01 lecture</p>
<!-- COURSE_FEEDBACK_QR:END -->
