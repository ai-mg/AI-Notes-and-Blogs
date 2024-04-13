# Unsupervised Learning

<!-- View the book with "<i class="fa fa-book fa-fw"></i> Book Mode". -->

<!-- Examples
---
- [Book example](/s/book-example)
- [Slide example](/s/slide-example)
- [YAML metadata](/s/yaml-metadata)
- [Features](/s/features)

Themes
---
- [Dark theme](/theme-dark?both)
- [Vertical alignment](/theme-vertical-writing?both)

###### tags: `Templates` `Book` -->

## Basics

Understanding the concepts of expectation, bias, and variance is crucial for grasping fundamental statistical principles in machine learning. Here’s a beginner-friendly explanation:

### 1. Expectation

The expectation (or expected value) is a core concept in probability theory and statistics, reflecting the average outcome one can expect from a random variable. Mathematically, it's the weighted average of all possible values that this random variable can take on, with the weights being the probabilities of each outcome.

- **For Discrete Random Variables**: If you have a discrete random variable $X$ that can take on values $x_1, x_2, ..., x_n$ with probabilities $P(X=x_1), P(X=x_2), ..., P(X=x_n)$, the expectation is given by:
$$ E[X] = \sum_{i=1}^{n}x_iP(X=x_i) $$

- **For Continuous Random Variables**: If $X$ is continuous with a probability density function $f(x)$, the expectation is:
$$E[X] = \int_{-\infty}^{\infty} x f(x) dx $$

### 2. Bias

Bias in machine learning refers to the error introduced by approximating a real-world problem, which may lead to systematic errors in predictions or estimations. For estimators (functions used to estimate parameters of a distribution), bias measures the difference between the expected value of the estimator and the true value of the parameter being estimated.

- **Mathematically**: If $\hat{\theta}$ is an estimator for the parameter $ \theta $, the bias of $\hat{\theta}$ is defined as:
$$ \text{Bias}(\hat{\theta}) = E[\hat{\theta}] - \theta $$

An estimator is called unbiased if its bias is 0 for all values of $\theta$, meaning on average, it accurately predicts the parameter.

### 3. Variance

Variance measures the spread of the random variable's values around its mean (expected value), indicating the variability from the average. In machine learning, variance captures how much the predictions for a given input vary between different realizations of the model.

- **Mathematically**: The variance of a random variable $X$ with mean $\mu = E[X]$ is given by:
$$ \text{Var}(X) = E[(X - \mu)^2] $$

In the context of estimators, variance measures how much the estimates $\hat{\theta}$ of the parameter $\theta$ would differ across different datasets drawn from the same distribution.

### 4. Mean Square Error (MSE)

The Mean Square Error (MSE) is a measure used to quantify the difference between the values predicted by a model and the actual values. Mathematically, it is defined as the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value.

For a set of predictions $\hat{y}_i$ and the observed values $y_i$, where $i$ indexes over $n$ observations, the MSE is given by:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$

In this formula:
- $\hat{y}_i$ represents the predicted value,
- $y_i$ represents the actual value,
- $n$ is the number of observations.

The MSE incorporates both the variance of the estimator (how spread out the predictions are) and its bias (how far the average prediction is from the actual value). Thus, it is a comprehensive measure that evaluates the quality of an estimator or a model in terms of both its precision and its accuracy.

In the context of vectors and matrices (especially in machine learning), the predictions $\hat{y}$ and actual values $y$ are often represented as vectors. The operation involving the difference between these vectors, squared, can be expressed using linear algebra notation, especially when dealing with multiple dimensions or multivariate regression. However, in the basic formulation of MSE given above, we directly compute the squared difference without explicitly mentioning vector transposition.

In some contexts, especially when dealing with matrix operations in multivariate regression or when predictions and actual values are matrices, you might see an expression like $(\hat{Y} - Y)^T(\hat{Y} - Y)$ for calculating a form of squared error. This notation incorporates matrix transposition ($^T$) and is used to perform matrix multiplication in a way that results in a scalar value representing the squared error. However, this is a more general or complex scenario than the basic MSE formula for a single predictive model.

The simple version of MSE provided initially is widely used in many machine learning tasks, including regression, where the goal is often to minimize this value to improve model performance.

## **Estimation Theory**

Estimation theory is concerned with estimating the values of parameters based on observed data. These parameters define the characteristics of a population, such as its average or variance. 

- **Key Idea**: If we have a dataset, how can we best estimate the underlying process that generated this data? For instance, given a set of measurements, how can we find the mean value that represents these measurements most accurately?


### 1. **Fisher Information Matrix**

The Fisher Information provides a measure of how much information an observable data sample carries about an unknown parameter of the model that generated the sample. It's essential for understanding the precision with which we can estimate these parameters.

- **Concept**: High Fisher Information indicates that the parameter estimate can be very precise. Low Fisher Information means the data provides little information about the parameter.

### 1.1 Likelihood
#### For univariate normal (Gaussian) distributions

To derive the log-likelihood formula for a dataset under the assumption that the data points are drawn from a normal (Gaussian) distribution, let's start with the probability density function (pdf) of the normal distribution. 


The pdf for a single data point $x$ given the mean $\mu$ and standard deviation $\sigma$ is:

$$p(x; \mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

For a dataset $X = \{x_1, x_2, ..., x_n\}$ consisting of $n$ independently and identically distributed (i.i.d.) observations drawn from this distribution, the likelihood function $L(\mu, \sigma | X)$ is the joint probability of observing all data points:

$$L(\mu, \sigma | X) = \prod_{i=1}^{n} p(x_i; \mu, \sigma)$$

Substituting the pdf into the likelihood function gives:

$$L(\mu, \sigma | X) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x_i-\mu)^2}{2\sigma^2}}$$

Taking the natural logarithm of the likelihood function to obtain the log-likelihood function $\ln L(\mu, \sigma | X)$ simplifies the multiplication into a sum:

$$\ln L(\mu, \sigma | X) = \ln \left( \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x_i-\mu)^2}{2\sigma^2}} \right)$$

$$\ln L(\mu, \sigma | X) = \sum_{i=1}^{n} \ln \left( \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x_i-\mu)^2}{2\sigma^2}} \right)$$

$$\ln L(\mu, \sigma | X) = \sum_{i=1}^{n} \left( -\ln(\sqrt{2\pi\sigma^2}) -\frac{(x_i-\mu)^2}{2\sigma^2} \right)$$

$$\ln L(\mu, \sigma | X) = -\frac{n}{2} \ln(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^{n} (x_i-\mu)^2$$

This is the log-likelihood function for a dataset under the assumption that the data points are drawn from a normal distribution, showing how the log-likelihood depends on the mean $\mu$ and standard deviation $\sigma$ of the distribution, as well as the observed data points $X$. This formula combines the constant terms and the sum of the squared differences between the observed data points and the mean, scaled by the variance. This expression is essential for methods like Maximum Likelihood Estimation (MLE), where we aim to find the parameters ($\mu$ and $\sigma$) that maximize this log-likelihood.

#### For multivariate normal (Gaussian) distributions

The PDF for a multivariate normal distribution for a random vector $X \in \mathbb{R}^d$ with mean vector $\mu \in \mathbb{R}^d$ and covariance matrix $\Sigma \in \mathbb{R}^{d \times d}$ is given by:

$$
p(x; \mu, \Sigma) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu)\right)
$$

Here, $|\Sigma|$ denotes the determinant of the covariance matrix $\Sigma$, and $(x-\mu)^{T}\Sigma^{-1}(x-\mu)$ is a quadratic form representing the Mahalanobis distance between the random vector $x$ and the mean vector $\mu$, weighted by the inverse of the covariance matrix.

To obtain the log-likelihood of observing a set of $n$ independent and identically distributed (i.i.d.) data points $X = \{x_1, x_2, ..., x_n\}$ from this distribution, we take the natural logarithm of the product of their probabilities:

$$
\ln L(\mu, \Sigma) = \sum_{i=1}^{n} \ln p(x_i; \mu, \Sigma)
$$

Substituting the PDF into the equation above and simplifying, we get:

$$
\ln L(\mu, \Sigma) = \sum_{i=1}^{n} \left[ -\frac{d}{2}\ln(2\pi) -\frac{1}{2}\ln(|\Sigma|) - \frac{1}{2}(x_i-\mu)^{T}\Sigma^{-1}(x_i-\mu) \right]
$$

This expression can be further simplified by aggregating constants and summing the quadratic forms across all observations:

$$
\ln L(\mu, \Sigma) = -\frac{nd}{2}\ln(2\pi) -\frac{n}{2}\ln(|\Sigma|) - \frac{1}{2}\sum_{i=1}^{n}(x_i-\mu)^{T}\Sigma^{-1}(x_i-\mu)
$$

The multivariate normal distribution is a generalization of the univariate normal distribution to multiple variables. It describes the behavior of a random vector in which all linear combinations of the components are normally distributed. This distribution is particularly useful in statistics and machine learning for modeling correlations between variables.

#### 1.1.1 Covariance Matrix ($\Sigma$)

- **Definition**: The covariance matrix, denoted as $\Sigma$, is a square matrix that encapsulates the covariance between each pair of elements in the random vector. For a random vector $X = [X_1, X_2, ..., X_n]$, the element at the $i$th row and $j$th column of $\Sigma$, denoted $\Sigma_{ij}$, is the covariance between $X_i$ and $X_j$. For diagonal elements, where $i = j$, $\Sigma_{ii}$ represents the variance of $X_i$.

- **Properties**:
    - $\Sigma$ is symmetric: $\Sigma = \Sigma^T$.
    - $\Sigma$ is positive semi-definite, meaning all its eigenvalues are non-negative. This property ensures that the quadratic form $(x-\mu)^T \Sigma^{-1} (x-\mu)$ in the log-likelihood equation is always non-negative.

#### 1.1.2 Inverse of the Covariance Matrix ($\Sigma^{-1}$)

- ***Role in Multivariate Normal Distribution***: The inverse of the covariance matrix, $\Sigma^{-1}$, plays a crucial role in the probability density function (pdf) of the multivariate normal distribution. It appears in the exponent of the pdf, contributing to measuring the "distance" of an observation $x$ from the mean $\mu$, taking into account the correlations between the variables.

- ***Geometric Interpretation***: The inverse of the covariance matrix can be seen as a transformation that "uncorrelates" the variables, mapping them into a new space where their covariance is zero except for their variances (on the diagonal). This is akin to stretching and rotating the coordinate axes so that the contours of equal probability density of the distribution become circles (in 2D) or spheres (in higher dimensions), rather than ellipses or ellipsoids.

- ***Mathematical Significance***: $\Sigma^{-1}$ adjusts the quadratic term $(x-\mu)^T \Sigma^{-1} (x-\mu)$ in the density function so that it accounts for both the variance and covariance of the variables. This term essentially measures the Mahalanobis distance, which is a generalized distance metric that considers the scale and correlation of the data dimensions.

Understanding the covariance matrix and its inverse is foundational for working with multivariate normal distributions in statistics and machine learning. They allow us to model complex relationships between multiple random variables, making the multivariate normal distribution a powerful tool for multivariate analysis, pattern recognition, and machine learning algorithms.

### 2. **Cramér-Rao Lower Bound (CRLB)**

The CRLB provides a theoretical lower limit on the variance of unbiased estimators. It tells us the best precision we can achieve with an unbiased estimator for estimating a parameter.

- **Understanding**: If you're estimating a parameter (like the mean of a population), the CRLB gives you the lowest variance possible for an unbiased estimator. It sets the benchmark for evaluating the efficiency of estimators.

### 3. Unbiasedness and Efficiency of an Estimator

In statistics, an estimator is a rule or formula that tells us how to calculate an estimate of a given quantity based on observed data. The quantity we're trying to estimate could be any parameter of the population from which the data was sampled, such as the population mean.

#### 3.1 The Arithmetic Mean as an Estimator

The arithmetic mean, often simply called the "mean," is one of the most basic estimators. It's used to estimate the central tendency or the average of a set of numbers.

#### 3.1.1 Unbiased Estimator

An unbiased estimator is a statistical technique used to estimate an unknown parameter of a distribution. It's called "unbiased" when the expected value of the estimation equals the true value of the parameter being estimated.

- **Mathematical Expression**: For a parameter $\theta$, an estimator $\hat{\theta}$ is unbiased if $E[\hat{\theta}] = \theta$, where $E[\cdot]$ denotes the expected value.

An estimator is called unbiased if, on average, it gives us the true value of the parameter we're trying to estimate. In more formal terms, an estimator is unbiased if its expected value—the long-run average of the estimates if we could repeat our sampling process an infinite number of times—is equal to the true parameter.

\
The arithmetic mean $\hat{\mu} = \sum_{i=1}^{n} \frac{1}{n} x_{i}$ is an unbiased estimator for the parameter $\mu$ because its expected value equals the true mean of the distribution from which the samples are drawn. Mathematically, this can be shown as follows:

$$ E[\hat{\mu}] = E\left[\frac{1}{n} \sum_{i=1}^{n} x_{i}\right] = \frac{1}{n} \sum_{i=1}^{n} E[x_{i}] $$

The expected value operator $E[\cdot]$ has a linear property, which means that the expected value of a sum of random variables is equal to the sum of their expected values

Since each $x_i$ is drawn from a distribution with mean $\mu$, $E[x_i] = \mu$ for all $i$. Thus,

$$E[\hat{\mu}] = \frac{1}{n} \sum_{i=1}^{n} \mu = \frac{1}{n} \cdot n \cdot \mu = \mu$$

This demonstrates that the arithmetic mean is an unbiased estimator for $\mu$, fulfilling the criteria for unbiasedness by having its expected value equal to the parameter it estimates.
   
#### Intuition

If you had a bag of marbles, with each marble having a number on it, and you wanted to know the average number, you could take a handful of marbles out and calculate their average by adding up the numbers and dividing by how many marbles you took. If you put them back and took a different handful many, many times, averaging each time, the average of all those averages would be very close to the average number on all marbles in the bag.

This is what it means for the arithmetic mean to be an unbiased estimator of the mean: if you could keep sampling and averaging, on average, you'd get the true mean of the whole population.


To understand the efficiency of the estimator $\hat{\mu}$ and how it relates to the Cramér-Rao Lower Bound (CRLB), let's go through the concepts step by step.

#### 3.1.2 Efficiency in Estimators

In the world of statistics, when we talk about the efficiency of an estimator, we are interested in how well the estimator performs in terms of its variance. We prefer estimators that give us results closer to the true parameter value more consistently, i.e., with less variance. Among all unbiased estimators, the one with the lowest variance is considered the most "efficient."

#### Variance of the Arithmetic Mean

The arithmetic mean estimator $\hat{\mu}$ of a set of observations has a variance. The variance measures how much the values of $\hat{\mu}$ would be spread out if we repeatedly took samples from the population and calculated their mean each time. For a normal distribution, the variance of the estimator $\hat{\mu}$ is given by $\frac{\sigma^2}{n}$, where:

- $\sigma^2$ is the variance of the population (a measure of how spread out the individual data points are around the true mean).
- $n$ is the number of samples we have taken.

For a normal distribution, the Fisher Information for the mean $\mu$ is $\frac{n}{\sigma^2}$. This tells us that more samples or less variability in our data both increase the amount of information we have about the mean.

The Cramér-Rao Lower Bound is a theoretical limit that tells us the best variance we could possibly achieve with an unbiased estimator of a parameter. In mathematical terms, it is the reciprocal of the Fisher Information: $\frac{1}{I(\mu)} = \frac{\sigma^2}{n}$. This formula says the lowest variance we can hope for decreases with more samples and increases with more variability in the data.

#### The Efficiency of $\hat{\mu}$

Now, if we compare the actual variance of $\hat{\mu}$ (which is $\frac{\sigma^2}{n}$) with the CRLB, we find they are the same. This means our arithmetic mean estimator is as good as it gets—it is the most efficient estimator we could use for the mean because it has the lowest variance that an unbiased estimator could possibly have according to the CRLB.

#### Intuitive Explanation

Think of the CRLB as the speed limit for estimators on the highway of statistics. It's the law that says "you cannot go below this variance while being an unbiased estimator." Now, if our arithmetic mean is cruising exactly at that speed limit, it means it's the fastest (or most efficient) estimator allowed by the "laws" of statistics for estimating the population mean.

In summary, the arithmetic mean is not just unbiased (giving us the right answer on average); it's also efficient (giving us the answer with as little random error as possible) when the data comes from a normal distribution. This dual quality of unbiasedness and efficiency makes the arithmetic mean a very powerful and commonly used estimator in statistics.

### Conclusion

Understanding these concepts forms the backbone of not only machine learning but also data science and statistics at large. They're essential for analyzing the behavior of algorithms and models, especially in understanding overfitting and underfitting, model selection, and for improving the predictive performance of machine learning models.

#### Additional Basics to Understand

To fully grasp the above concepts, you should be familiar with the following:

- **Probability Theory**: Understanding random variables, probability distributions, and their properties is crucial.
- **Basic Calculus and Algebra**: Derivatives and integrals are used extensively in deriving estimators and their properties. Linear algebra is essential for understanding data structures and manipulations in machine learning.
- **Statistical Measures**: Know how to calculate and interpret mean, median, variance, and standard deviation, as these are foundational in statistical analysis and inference.

By starting with these foundational topics and gradually exploring more complex concepts, you'll build a solid understanding of estimation theory and its applications in machine learning and data science.

### Resources and References on Basics

1. **"Introduction to Probability" by Joseph K. Blitzstein and Jessica Hwang**: This book provides a comprehensive introduction to probability, covering expectation, variance, and many other fundamental concepts with clarity.

2. **"The Elements of Statistical Learning" by Trevor Hastie, Robert Tibshirani, and Jerome Friedman**: A cornerstone text in statistical learning that explains the trade-off between bias and variance, among other concepts.

3. **Khan Academy’s Statistics and Probability**: Offers free online tutorials that cover basics and advanced concepts in statistics and probability, including expectation, bias, and variance.

4. **"Pattern Recognition and Machine Learning" by Christopher M. Bishop**: Provides detailed explanations of bias, variance, and expectation within the context of machine learning models.




### Resources for Learning Advanced Concepts

1. **Mathematics for Machine Learning** by Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong is an excellent resource that covers the mathematical underpinnings necessary for understanding these concepts in machine learning.

2. **Statistical Inference** by George Casella and Roger L. Berger provides a deep dive into estimation theory, unbiased estimators, and much more, giving you a thorough understanding of statistical theory.

3. **Online Courses**: Websites like Coursera, edX, and Khan Academy offer courses in statistics and machine learning that start from the basics and advance to more complex topics, including estimation theory and statistical inference.

