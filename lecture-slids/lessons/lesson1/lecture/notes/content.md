# Lesson 1: Why Learn Machine Learning?

Machine learning is often introduced as a collection of algorithms: linear regression, decision trees, neural networks, and many others. This course takes a different view.

The central skill is learning how to turn a real question into a problem that data can help us answer. That requires more than running code. We must decide what the problem really is, what information can be used, what would count as a good answer, how to find that answer, and what evidence would justify trusting it.

These decisions matter even more now that an AI assistant can generate code in seconds.

## What you should gain from this lesson

By the end of this lesson, you should be able to:

- explain why a prediction can look accurate without solving the real problem;
- identify the questions that should come before choosing a model;
- describe the modelling process that will be repeated throughout this course;
- explain what AI can accelerate and what judgement remains your responsibility;
- describe the capabilities you should have developed by the end of the course.

## 1. A tempting question

Suppose we ask an AI assistant:

> Use historical stock-market data to predict tomorrow's closing price. Write the code and compare the predictions with what actually happened.

This sounds like a natural machine-learning project. It has data, a value to predict, and an outcome that many people care about. It is also a task for which an AI assistant can quickly produce a complete-looking solution.

The assistant can write code that downloads the data, prepares the inputs, builds a simple prediction rule from recent prices, and draws a graph. The assistant produces the implementation; the resulting model produces the predictions. We can even arrange the evaluation so that the model predicts each day without seeing that day's answer.

Now imagine that we plot several years of predictions against the actual market. The two lines are almost impossible to separate.

<!-- Figure task: Show the actual market level and the AI-generated next-day prediction over a long period. The reader should initially experience the result as highly convincing because the two lines almost overlap. -->

Has the model solved the problem?

Before answering, notice that several different claims have quietly become mixed together:

- the program runs;
- the model predicts the next recorded price with a small error;
- the model has learned useful information about tomorrow;
- the result can support an investment decision.

These claims are not equivalent.

## 2. A simple comparison changes the conclusion

Consider a rule that does not require training a model:

> Predict that today's closing price will be the same as yesterday's closing price.

This is a **baseline**: a simple and reasonable point of comparison. A baseline tells us what can already be achieved before introducing a more complicated method.

When the actual market and yesterday's price are plotted over the same long period, the two lines also appear almost identical.

<!-- Figure task: Compare the AI prediction and yesterday baseline over the same period and visual scale. -->

The first model's graph was not necessarily false. The problem is that the graph did not establish the conclusion we wanted.

Market levels usually change by much less from one day to the next than they change across several years. On a long-term chart, yesterday's value will therefore sit very close to today's value. A model can produce a visually impressive prediction without successfully anticipating the new movement that matters for a decision.

Look more closely at a short, fixed window and compare the changes from one day to the next. The prediction repeatedly moves in the wrong direction. Across all unseen days from 2020 to 2024, it identifies the direction correctly on only about 48 out of every 100 days. A rule that always says “up”, without examining the recent prices at all, is correct on about 54 out of 100 days in the same period. The impressive long-term overlap has not become useful foresight.

<!-- Figure task: Show the short-window daily changes together with the full-period direction comparison. -->

This leads to a more important question:

> Were we trying to predict the price level, predict the next change, or decide whether to take an action?

Until that is clear, there is no single meaning of “accurate”.

## 3. Machine learning begins before the model

The stock example reveals the main idea of this course: machine learning begins with a real problem, not with an algorithm.

We will repeatedly ask six questions.

### What are we trying to solve?

A vague wish such as “predict the market” is not yet a well-defined task. Do we want tomorrow's price, the direction of movement, the probability of a large change, or support for a particular decision? Different questions require different outputs and different evidence.

We should also ask whether machine learning is the right tool at all. Some questions can be answered by a simple rule, direct calculation, or better data collection.

### How will data represent the problem?

We must decide what one example represents, which information is available at the time of prediction, and what output the model should produce.

This step connects the real situation to data. A model cannot use information that was unavailable when the decision had to be made, and it cannot recover an important distinction that the data representation has removed.

### What counts as a good answer?

A model improves whatever standard we give it. That standard must match the real task.

A small error in the predicted price level may be useful for one purpose and almost irrelevant for another. If the intended decision depends on tomorrow's movement, an evaluation dominated by the overall price level may answer the wrong question.

### How will we find an answer?

We choose a family of possible rules or models and use data to find a better one. For some models, the best answer can be calculated directly. For others, the answer is improved step by step.

The important point is not simply which software function to call. We need to understand what kind of answer the method is allowed to produce and what it is trying to improve.

### What evidence would justify trusting it?

Code that runs is evidence that the implementation can execute. Good performance on data already used for fitting is evidence that the model can adapt to those examples. Neither is enough to show that the model will work in a new situation.

Useful evidence normally requires unseen data, a sensible baseline, and an evaluation that matches the claim we want to make.

### Where might it fail?

Every model relies on assumptions and has a limited domain of use. Data can change, important information can be missing, and a method that works for one group or period may fail for another.

Identifying a limitation is not an admission that learning has failed. In this course, limitations are what create the need for the next method.

<!-- Figure task: Present the recurring modelling process as one connected sequence: real problem, data representation, criterion for a good answer, solution, evidence, and limitations. The figure should communicate a reusable way of thinking, not a rigid checklist. -->

### Applying the six questions to the stock example

We can now apply the complete framework to the opening stock-market prediction.

**What are we really trying to solve?** The original ambition was to use a prediction to support an investment decision. The AI implementation actually predicted the next day's price level. These are not the same task.

**What information can be used?** Each prediction can use only information that already existed at that time. Data revealed after the prediction cannot be used as an input or to choose the model.

**What counts as a good answer?** If the real interest is tomorrow's new movement, producing a value close to the current price is not enough. The success criterion must reflect whether the model adds information that is useful for the decision.

**How will we find an answer?** The current model uses recent closing prices to find a rule for producing the next prediction. Later lessons will examine more carefully what forms such rules can take and how data are used to improve them.

**What evidence is needed?** We need to compare the model on days it has not seen with sensible simple rules, such as using yesterday's price or always predicting the more common direction. Any advantage must also be relevant to the original decision.

**What are the limits of the current conclusion?** The existing graph shows that the model can produce values close to the actual price level. It does not show that the model anticipated new market movements, and it does not show that following the model would produce investment returns.

This audit used no new algorithm. It simply separated questions, comparisons, and conclusions that had previously been mixed together. This is the capability that the rest of the course will repeatedly develop.

## 4. A high score can still be useless

Suppose a rare disease affects 10 people in every 10,000. A system gives everyone the same prediction: **healthy**.

It will be correct for 9,990 people, producing an accuracy of 99.9%. The number looks excellent. Yet the system has found none of the 10 people who need help.

The arithmetic is not wrong. The problem is that 99.9% answers “How often was the system correct overall?” but not “Can it find the people who need help?” A large number can hide the failure that matters most for the real task.

Now make two quicker judgements:

- We want to predict whether a student will pass before the final examination, but the final mark is included as an input. However impressive the result, that information does not exist when the real prediction must be made.
- We want to predict tomorrow's demand for shared bikes. If a complicated method cannot beat “use yesterday's demand” or “use the same day last week”, its extra complexity has not yet demonstrated value.

The contexts are different, but the questions are the same: What is the real purpose? What information is genuinely available? What simple method should we beat? What evidence supports the conclusion we care about?

## 5. What you should be able to do by the end of the course

By the end of the course, you should be able to take an unfamiliar real-world question and a new dataset through a defensible modelling process.

You should be able to:

1. turn a vague requirement into a clear learning task;
2. identify the observations, inputs, outputs, and important constraints;
3. establish a sensible baseline before adding complexity;
4. choose and train a model for a reason you can explain;
5. evaluate it on appropriate unseen data;
6. compare alternatives using evidence that matches the decision;
7. communicate what the result supports and where its limits lie.

This does not mean memorising every model or writing every implementation without help. It means being able to explain and defend the important choices, including when AI assists with the code.

## 6. How the course builds this capability

The course expands the kinds of problems you can handle in four stages.

### Predict a number

We begin with quantities such as demand, temperature, or price. This gives us the first complete modelling process: describe the available information, define the kind of prediction we want, measure its errors, improve it using data, and examine whether it works on new examples.

We then meet relationships that a simple straight-line pattern cannot describe, followed by flexible models that begin to follow accidental noise. These difficulties motivate better ways to represent the inputs and control unnecessary complexity.

### Support a classification decision

Many real tasks do not ask for a number. They ask whether an event will happen, which category an example belongs to, or how likely an outcome is.

Changing the output changes the model, what it is asked to improve, and the evidence we need. It also forces us to consider which kinds of mistakes matter most for the decision.

### Find structure without supplied answers

Sometimes the data do not come with a known answer. We may instead want a simpler view of a complicated dataset or groups of similar observations.

The modelling process still applies, but “a good answer” must be defined differently. The result depends on the structure we ask the method to find and the assumptions built into that request.

### Learn useful representations

Earlier models mainly depend on measurements selected or constructed by the analyst. Neural networks can learn useful intermediate descriptions of the data alongside the final prediction.

This allows us to work with more complex information, including images, while preserving the same fundamental questions about what counts as a good result, how an answer is found, what evidence supports it, and where it may fail.

<!-- Figure task: Show the four course stages as an expanding range of capabilities. The visual should emphasise what new kind of problem the student can handle at each stage, rather than listing algorithm names. -->

The course therefore does not move through methods arbitrarily. Each new topic appears because the previous capability is no longer enough.

## 7. How we will learn each week

Each topic will follow a recognisable pattern.

We will begin with a problem or with the failure of a method we already understand. We will formalise the new task, define what a good answer means, examine how the method finds an answer, and decide what evidence supports using it. We will then identify an important limitation that motivates the next development.

Different course materials support different parts of this work:

- the **lecture** establishes the problem and the main reasoning;
- the **note** provides the complete explanation and optional depth;
- a **demo** makes an important mechanism or comparison observable;
- the **lab** asks you to reconstruct the reasoning with data and code.

They are not four unrelated sets of content. They are four ways of developing the same capability.

## 8. Learning with AI

AI assistants are useful in this course. They can help you:

- explain unfamiliar code or terminology;
- suggest and debug an implementation;
- generate a small test case;
- compare alternative approaches;
- produce a first version of a graph or analysis.

You remain responsible for:

- stating the real problem and assumptions;
- checking whether the data and target are appropriate;
- deciding what comparison and evidence are needed;
- running and inspecting the code;
- explaining the result in your own words;
- limiting the conclusion to what the evidence supports.

A productive habit is:

> **Use, test, check, and explain.**

AI changes how quickly an implementation can be produced. It does not remove the need to understand what the implementation is doing or whether its result is useful.

## 9. Two levels, one learning journey

MPS311 and MPS439 share the same problems and core narrative. They differ in the depth of analysis, not in whether one group receives a completely separate course.

### MPS311

You should be able to use standard methods correctly, explain the main modelling choices, compare a model with a sensible baseline, interpret validation evidence, and communicate the main limitations.

### MPS439

In addition to the shared core, you will go further in two directions.

**Depth** means examining objectives and solution methods more closely, implementing or modifying important components, and testing assumptions, sensitivity, and robustness.

**Breadth** means exploring more advanced questions, modern tools, and research or professional practices when they genuinely deepen the topic. For example, later neural-network topics may use a framework such as PyTorch to move beyond high-level model calls.

The aim is not to collect additional software names. It is to become able to investigate a model more deeply and design stronger evidence.

## Optional MPS439 extension: redesigning the AI-generated solution

Return to the stock-market example and treat the AI-generated analysis as a proposal that needs to be redesigned. Your task is not only to identify its weaknesses, but to rewrite the original request so that it defines a more credible investigation.

The revised request should state:

1. the question the model is genuinely intended to answer;
2. the information that may be used at each prediction time;
3. the simple baseline the model should outperform;
4. what result would count as adding value;
5. the claims that the final evidence would be allowed to support.

For example, the original request—“predict tomorrow's stock price and plot it against the actual price”—could become:

> Using only information available before each prediction, investigate whether a model provides more information about the next day's change than simply using yesterday's price. Compare the methods on dates the model has not seen, and state clearly what the result does and does not support.

The point is not yet to select a more complicated model or a technical measure. It is to turn a plausible AI-generated request into an investigation whose question, information, comparison, evidence, and conclusion can be clearly explained and tested.

Broader directions that later topics may revisit include uncertainty, changing data distributions, information leakage, reproducibility, and model responsibility.

## 10. Takeaways

Machine learning is not simply the act of running an algorithm. It is a process for turning a real problem into a learnable task, finding an answer, and deciding what evidence supports using it.

A model can look accurate without solving the problem we care about. A sensible baseline can expose this quickly.

This course will progressively expand the kinds of problems you can handle, while repeating the same modelling questions in each new setting.

AI can accelerate implementation. You remain responsible for the problem, the evidence, and the conclusion.

Before finishing, check whether you can now:

- explain why an almost perfectly overlapping prediction graph may still be misleading;
- propose a sensible simple baseline for a more complicated model;
- state what information and evidence you would need before trusting the model.

Return to the opening stock-market graph. Before trusting it, what is the first question you would now ask?
