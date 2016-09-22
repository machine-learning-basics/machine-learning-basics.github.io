---
layout: page
mathjax: true
permalink: /classification/
---

These supplemental notes are meant to accompany [CS229's Lecture notes 1](http://cs229.stanford.edu/notes/cs229-notes1.pdf). This site is not affiliated with Stanford. It's purely a guide to help understand the CS229 notes better and to spread the joy of machine learning. Please comment if you find any errors.

- [Intro to Machine Learning](#intro)
- [Supervised Learning](#supervised-learning)
  - [Hypothesis](#hypothesis)
  - [Update Rule](#update-rule)
- [Further Reading](#reading)

<a name='intro'></a>

## Intro to Machine Learning

**Definition**. Machine learning is training a computer to perform a task by feeding it data. Machine learning can be separated into three main types of learning:

- **supervised learning** -- learning with input and **labeled output**
- **unsupervised learning** -- learning with **just** input. no output.
- **reinforcement learning** -- learning with input, **some output**, and the **output's grade**. Note the output isn't necessarily the desired output.

<a name='supervised-learning'></a>

## Supervised Learning

**Example**. Suppose you work at a tech giant called XYZ Corp. XYZ Corp's recruiters and engineering managers want to know how productive an incoming software engineer will be. Specifically, they want to know how many Git commits she will produce after she ramps up at XYZ Corp. The engineering managers notice that the number of Git commits a software engineer produces is proportional to the number of months of industry experience, so they gather data on XYZ Corp's currently employed engineers.

<div class="fig figcenter fighighlight">
  <img src="/assets/code_vs_industry_table.png">
</div>

You, a machine learning practioner at XYZ Corp, plot this data and notice that the relationship between industry experience and number of Git commits is roughly linear. You decide that it's best to use machine learning to build a mathematical model that will predict the number of Git commits a software engineer will produce given his/her industry experience.

<div class="fig figcenter fighighlight">
  <img src="/assets/code_vs_industry_experience.png">
</div>

In this example, the **input is number of months of industry experience** and the **output is number of Git commits.** Since there is output, that rules out the possiblity of this problem being an example of unsupervised learning. Furthermore, the data tells you exactly how many Git commits a software engineer will produce for a given amount of industry experience. This number, the number of Git commits, is the output the recruiters and engineering managers desire. This makes it a supervised learning problem. In contrast, this is not a reinforcement learning problem because the desired output, the number of Git commits, is given. We'll see examples of reinforcement learning problems later.

<a name='hypothesis'></a>

### Hypothesis
---

Because the relationship between the input and the output looks linear, we can draw a line through it. Now this begs the question: if months of industry experience is along the x-axis, and number of Git commits is along the y-axis, what's the **best-fit** line to draw through this data such that given an x-axis value, the line will tell you the corresponding y-axis value that's a good prediction of the real y-axis value (i.e., the real number of Git commits a person with $$x$$ months of industry experience can produce)? In more technical terms, we want to find a line such that given an **input/independent variable (or feature)**, the corresponding **output variable (or target/dependent variable)** will be the best prediction of the real target value based on the available data. That was a mouthful, so let's transform this problem from the language of English to the language of mathematics. Transforming this problem into mathematics makes it more concise and allows us to take advantage of mathematical properties discovered by mathematicians and scientists.

<div class="fig figcenter fighighlight">
  <img src="/assets/code_vs_industry_experience_fitted.png">
  <div class="figcaption">The best fit line through the training data. We'll discuss how to find this best fit line (hypothesis) with the update rule.</div>
</div>

Let's return to grade school math. We want to find a line. The equation for a line is

\begin{equation}
y = mx + b \quad (1)
\end{equation}

where $$y$$ is the target variable and $$x$$ is the feature. In grade school, $$m$$ is known as the slope, and $$b$$ is known as the y-intercept. In machine learning, the slope is known as the **parameter (or weight)**, and the y-intercept is known as the **intercept term (or bias)**. Let's rewrite this exact equation using the Greek letters with which machine learning practioners and researchers are familiar:

\begin{equation}
y = \theta_1x + \theta_0 \quad (2)
\end{equation}

How do we find $$\theta_1$$ and $$\theta_0$$ (i.e., the weight of the feature and the bias, respectively)? Intuitively, let's try drawing a line through the data and calculating a score for that line. Then let's draw another line, hopefully one that'll have a better score. Then another. And another. Let's keep drawing lines that fit the data until we find a line whose score is the best among all the lines we've drawn.

How do we compute this score? What would be the mathematical formula for it? Intuitively, we want a line that's as close as possible to all the **(feature, target)** pairs (or **training samples**) we have. Remember, in our example above, the recruiters and engineering managers have given us a list of engineers' industry experience in months and their corresponding number of Git commits.

How do we measure how *close* a training sample is to a line? Let's introduce more terminology before we move on. We call the result of $$\theta_1x + \theta_0$$ a **hypothesis**. In other words, going back to our grade school equation of a line, $$y = hypothesis$$. Let's rewrite our equation again:

\begin{equation}
h_\theta(x) = \theta_1x + \theta_0 \quad (3)
\end{equation}

We have a training sample $$(x^{(1)}, y^{(1)})$$. The superscript **(1)** doesn't denote an exponent. It simply means the **1st** training example. Note that we have multiple training samples $$(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \cdots, (x^{(m)}, y^{(m)})$$, where $$m$$ is the number of training samples we have.  The **hypothesis or predicted dependent variable** evaluated at $$x^{(1)}$$ is $$h_\theta(x^{(1)})$$, and $$y^{(1)}$$ is the **observed dependent variable**. The difference between the observed dependent variable and the hypothesis, $$y^{(1)} - h_\theta(x^{(1)})$$, is the **residual (or cost)**. If we compute the hypothesis for each of the $$m$$ training samples, that is, $$h_\theta(x^{(1)}), h_\theta(x^{(2)}), \cdots, h_\theta(x^{(m)})$$, we can compute $$m$$ residuals or costs. If we sum up these $$m$$ costs, we'll have the cost of a specific line. Note in $$(3)$$ above, the variables that define the line are $$\theta_1, \theta_0$$. These are the **parameters** of the line/hypothesis. Whew. That was a lot to take in. Let's review:

$$
\begin{array}{l|l}
(x^{(i)}, y^{(i)}) & i\text{-th training sample} \\
h_\theta(x^{(i)}) & \text{hypothesis or predicted dependent variable} \\
y^{(i)} & \text{observed dependent variable} \\
h_\theta(x^{(i)}) - y^{(i)} & i\text{-th residual or cost}
\end{array}
$$

Let's mathematically define the cost of a line to be the sum of the vertical distances between all $$h_\theta(x^{(i)})$$ and $$y^{(i)}$$

\begin{equation}
J(\theta) = \text{cost of a line} = \sum_{i = 1}^m (h_\theta(x^{(i)}) - y^{(i)})
\end{equation}

where $$\theta$$ is a $$n \times 1$$ column vector, $$n$$ being the number of parameters (e.g., **2** for our example).

<div class="fig figcenter fighighlight">
  <img src="/assets/classification_residual.jpg">
  <div class="figcaption">The cost of the line is the sum of the residuals.</div>
</div>

What's wrong with this cost function? You'll notice that some residuals may be **positive**, and some may be **negative**. This is bad. The positive residuals will sort of (or in the worst case, completely) cancel out of the negative residuals. As a result, the cost function above may result in low costs for lines that are a terrible fit for the data. To take care of this problem, we can take the absolute value of the residual.

\begin{equation}
J(\theta) = \sum_{i = 1}^m \left| h_\theta(x^{(i)}) - y^{(i)} \right|
\end{equation}

Another equation for the cost function could be the sum of the squared residuals
\begin{equation}
J(\theta) = \sum_{i = 1}^m \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2
\end{equation}

Actually, according to the [Gauss-Markov theorem](https://en.wikipedia.org/wiki/Gauss%E2%80%93Markov_theorem), this cost function will product the best fit line, so let's proceed with this function. This cost function ensures negative and postiive residuals ultimately contribute to the cost since the residuals are squared.

One thing you might notice is we set out to find a line that's close to as many training samples as possible. If that's the kind of line we're looking for, shouldn't we look for the perpendicular distance between a point and a line and not the vertical distance?

<div class="fig figcenter fighighlight">
  <img src="/assets/classification_ols_vs_tls.png">
</div>

If you look at the graphs above, on the left we see the vertical distance between a point and a line being used as the cost. On the right we see the perpendicular distance between a point and a line being used as the cost. The left is called **ordinary least squares (OLS)**, and the right is called **total least squares (TLS).** In our example, our input is known. That is, we know the engineer's months of industry experience. That is, if we're told an engineer has 24 months of industry experience, he has 24 months of industry experience. What we're trying to predict is the number of Git commits he has produced given his known industry experience. Thus, the error comes from the difference between the known and predicted values of number of Git commits. There is no error in independent variables. TLS allows for error in independent variables. That is why we use OLS, the cost function we mentioned above.

<a name='update-rule'></a>

### Update Rule
---

Great, now we have a cost function that will tell us how well a line fits the data. How do we go about finding more and more lines? In other words, how do we update $$\theta_1, \theta_0$$ so that we have more lines to test with? Thankfully, two talented and smart Stanford researchers came up with an **update rule** called the **Widrow-Hoff rule**.

\begin{equation}
\theta = \theta + \alpha \cdot (-\nabla_\theta J(\theta))
\end{equation}

There is a lot going on in that equation, so let's break it down, step by step. In single variable calculus, the derivative of a function tells you the rate of change of the function. For example, given a function $$f(x)$$, the derivative $$\frac{df}{dx}$$ tells you how fast $$f(x)$$ is changing at $$x$$. In multivariable calculus, the derivative is called a **gradient**, and it is denoted with $$\nabla_x f(x)$$. If $$x$$ is a vector, the gradient tells you how fast $$f(x)$$ is changing at the point $$(x_1, x_2, ..., x_n)$$.

Going back to the cost function, $$\nabla_\theta J(\theta)$$ is the gradient of the cost function $$J(\theta)$$ with respective to $$\theta$$. The cost function is a **scalar value**. On the other hand, the gradient of the cost function is a $$n \times 1$$ column vector, where $$n = $$ number of features.

In calculus, when the derivative (or gradient) is **0**, that means the function is not changing in value anymore. Pictorially, that means you're either at the top or bottom of a curve. For example, if $$f(x) = x^2$$, the equation of a parabola, when $$\frac{df}{dx} = 0$$, you're at the bottom of the parabola. If $$\frac{df}{dx} > 0$$, as $$x$$ increases, $$f(x)$$ will increase as well. If $$\frac{df}{dx} <0 0$$, as $$x$$ increases, $$f(x)$$ will decrease.

Given our cost function, we want successive values of $$\theta$$ such that $$J(\theta)$$ will decrease. Successive values of $$\theta$$ that result in decreasing $$J(\theta)$$ give us more and more lines that will fit the data better and better.

<a name='reading'></a>

#### Further Reading

Here are some (optional) links you may find interesting for further reading:

- [Gauss-Markov theorem](https://en.wikipedia.org/wiki/Gauss%E2%80%93Markov_theorem), learn why the least squares cost function results in the best fit line.
