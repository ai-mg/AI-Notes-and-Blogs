# Theoretical Concepts of Machine Learning

[![hackmd-github-sync-badge](https://hackmd.io/2DM4d7wATx-tCNn5QQwm0A/badge)](https://hackmd.io/2DM4d7wATx-tCNn5QQwm0A)


## Central Limit Theorem

The Central Limit Theorem (CLT) is a fundamental result in the field of statistics and probability theory. It provides a foundation for understanding why many distributions in nature tend to approximate a normal distribution under certain conditions, even if the original variables themselves are not normally distributed. The theorem states that, given a sufficiently large sample size, the distribution of the sample means will be approximately normally distributed, regardless of the shape of the population distribution, provided the population has a finite variance.

The formula or the mathematical formulation of the CLT can be derived from the concept of convergence in distribution of standardized sums of independent random variables. Let's consider the classical version of the CLT to understand where the formula comes from:

Consider a sequence of $n$ independent and identically distributed (i.i.d.) random variables, $X_1, X_2, ..., X_n$, each with a mean $\mu$ and a finite variance $\sigma^2$. The sample mean $\bar{X}$ of these $n$ random variables is given by:

$$
\bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i
$$

The Central Limit Theorem tells us that as $n$ approaches infinity, the distribution of the standardized sample means (i.e., how many standard deviations away the sample mean is from the population mean) converges in distribution to a standard normal distribution. The standardized sample mean is given by:

$$
Z = \frac{\bar{X} - \mu}{\sigma / \sqrt{n}}
$$

This formula ensures that $Z$ has a mean of 0 and a standard deviation of 1:

- **Mean of \(Z\)**: $E(Z) = E\left(\frac{\bar{X} - \mu}{\frac{\sigma}{\sqrt{n}}}\right) = \frac{E(\bar{X}) - \mu}{\frac{\sigma}{\sqrt{n}}} = 0$

- **Variance of $Z$**: $Var(Z) = Var\left(\frac{\bar{X} - \mu}{\frac{\sigma}{\sqrt{n}}}\right) = \frac{Var(\bar{X})}{\left(\frac{\sigma}{\sqrt{n}}\right)^2} = 1$

Here, $Z$ converges in distribution to a standard normal distribution $N(0, 1)$ as $n$ becomes large.

<!-- This means:

$$
\lim_{n \to \infty} P\left(a < \frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} < b\right) = \Phi(b) - \Phi(a)
$$

where $P$ denotes probability, $\Phi$ is the cumulative distribution function (CDF) of the standard normal distribution, and $a$ and $b$ are any two points on the real line with $a < b$. -->

The CLT is derived from the properties of characteristic functions or moment-generating functions of probability distributions. In essence, the CLT can be proven by showing that the characteristic function of $Z$ converges to the characteristic function of a standard normal distribution as $n$ approaches infinity ([see Appendix A.3](#A.3.-Derivation-of-normal-distribution-from-central-limit-theorem)).

The significance of the CLT lies in its ability to justify the use of the normal distribution in many practical situations, including hypothesis testing, confidence interval construction, and other inferential statistics procedures, even when the underlying population distribution is unknown or non-normal.

### 1. Why $\sigma$ is divided by $\sqrt{n}$:

We know that that Z-Score or Standardization is the deviation of the data point from mean in units of standard deviation ([see Appendix A.2. on Standardization](#A.2.-Standardization-or-$Z$-Score)). Here, the deviation is of the sample mean ($\bar{X}$) from the population mean ($\mu$). Therefore, we derive the standard deviation of the sample mean ($\bar{X}$) as follows.

#### Variance of the Sample Mean $\bar{X}$

The variance of the sample mean $\bar{X}$ is derived as follows. Since $\bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i$, its variance is:

$$
Var(\bar{X}) = Var\left(\frac{1}{n} \sum_{i=1}^{n} X_i\right)
$$

Because the $X_i$ are i.i.d., the variances add up, and we get:

$$
Var(\bar{X}) = \frac{1}{n^2} \sum_{i=1}^{n} Var(X_i) = \frac{1}{n^2} \cdot n \cdot \sigma^2 = \frac{\sigma^2}{n}
$$

This shows that the variance of the sample mean decreases as the sample size $n$ increases.


#### Standard Deviation of $\bar{X}$

The standard deviation is the square root of the variance. Therefore, the standard deviation of the sample means, also known as the standard error of the mean (SEM), is:

$$
SEM = \sqrt{Var(\bar{X})} = \sqrt{\frac{\sigma^2}{n}} = \frac{\sigma}{\sqrt{n}}
$$

#### Mathematical Explanation for Dividing by $\sqrt{n}$

- **Reducing Spread**: Dividing by $\sqrt{n}$ reduces the spread of the sampling distribution of the sample mean as the sample size increases. This reflects the fact that larger samples are likely to yield means closer to the population mean ($\mu$), thus decreasing variability among the sample means.
  
- **Normalization**: The process of dividing the population standard deviation ($\sigma$) by $\sqrt{n}$ normalizes the scale of the sample means' distribution. This normalization ensures that no matter the sample size, the scale (spread) of the distribution of sample means is consistent and comparable.

#### Role in the Central Limit Theorem

The CLT states that as $n$ approaches infinity, the distribution of the standardized sample means:

$$
Z = \frac{\bar{X} - \mu}{\frac{\sigma}{\sqrt{n}}}
$$

converges to a standard normal distribution $N(0, 1)$. Here, $\bar{X}$ is the mean of a sample of size $n$, $\mu$ is the population mean, and $\sigma$ is the population standard deviation. The denominator $\frac{\sigma}{\sqrt{n}}$ standardizes the distribution of $\bar{X}$ by adjusting for the size of the sample, allowing the theorem to hold across different sample sizes and population variances.

#### Conclusion

Mathematically, dividing by $\sqrt{n}$ in the calculation of the SEM and the standardization of sample means under the CLT ensures that the variability among sample means decreases with increasing sample size. This adjustment is fundamental to the convergence of the distribution of sample means to a normal distribution, a cornerstone of statistical inference.

### 2. Derivation of the Variance of $Z$

Given the definition of $Z$, to find its variance, we use the property that the variance operator $Var(aX) = a^2 Var(X)$ for any random variable $X$ and constant $a$ (see proof in Appendix A.1.). Applying this to the definition of $Z$, we get:

$$
Var(Z) = Var\left(\frac{\bar{X} - \mu}{\sigma / \sqrt{n}}\right)
$$

Since $\mu$ is a constant, subtracting it from $\bar{X}$ does not affect the variance, so we focus on the scaling factor. Applying the variance operator:

$$
Var(Z) = \left(\frac{1}{\sigma / \sqrt{n}}\right)^2 Var(\bar{X}) = \left(\frac{\sqrt{n}}{\sigma}\right)^2 \cdot \frac{\sigma^2}{n} = \frac{n}{\sigma^2} \cdot \frac{\sigma^2}{n} = 1
$$

This calculation shows that the variance of $Z$ is 1. Here's the breakdown:

- $\left(\frac{1}{\sigma / \sqrt{n}}\right)^2$ is the square of the inverse of the standard deviation of $\bar{X}$, which is $\sigma / \sqrt{n}$.
- $Var(\bar{X}) = \frac{\sigma^2}{n}$ is the variance of the sample mean.
- Multiplying these together, the $\sigma^2$ and $n$ terms cancel out, leaving $Var(Z) = 1$.

The derivation shows that the process of standardizing the sample mean $\bar{X}$ results in a new variable $Z$ with a variance of 1. This is a crucial step in the application of the CLT because it ensures that $Z$ is scaled appropriately to have a standard normal distribution with mean 0 and variance 1 as $n$ becomes large. This standardization allows us to use the properties of the standard normal distribution for statistical inference and hypothesis testing.

#### Properties of Variance

Variance is a fundamental statistical measure that quantifies the spread or dispersion of a set of data points or a random variable's values around its mean. Understanding the properties of variance is crucial for statistical analysis, as these properties often underpin the manipulation and interpretation of statistical data. Here are some key properties of variance:

#### 1. Non-negativity
Variance is always non-negative ($Var(X) \geq 0$). This is because variance is defined as the expected value of the squared deviation from the mean, and a square is always non-negative.

#### 2. Variance of a Constant
The variance of a constant ($c$) is zero ($Var(c) = 0$). Since a constant does not vary, its spread around its mean (which is the constant itself) is zero.

#### 3. Scaling Property
Scaling a random variable by a constant factor scales the variance by the square of that factor: $Var(aX) = a^2 Var(X)$, where $a$ is a constant and $X$ is a random variable. This property was detailed in a previous explanation.

#### 4. Variance of a Sum of Random Variables
For any two random variables $X$ and $Y$, the variance of their sum is given by $Var(X + Y) = Var(X) + Var(Y) + 2Cov(X, Y)$, where $Cov(X, Y)$ is the covariance of $X$ and $Y$. If $X$ and $Y$ are independent, $Cov(X, Y) = 0$, and the formula simplifies to $Var(X + Y) = Var(X) + Var(Y)$.

#### 5. Linearity of Variance (for Independent Variables)
While the expectation operator is linear ($E[aX + bY] = aE[X] + bE[Y]$), variance is not linear except in specific cases. For independent random variables $X$ and $Y$, and constants $a$ and $b$, $Var(aX + bY) = a^2 Var(X) + b^2 Var(Y)$. However, for dependent variables, you must also consider the covariance term.

#### 6. Variance of the Difference of Random Variables
Similar to the sum, the variance of the difference of two random variables is $Var(X - Y) = Var(X) + Var(Y) - 2Cov(X, Y)$. For independent variables, this simplifies to $Var(X - Y) = Var(X) + Var(Y)$, as their covariance is zero.

#### 7. Zero Variance Implies a Constant
If a random variable $X$ has a variance of zero ($Var(X) = 0$), then $X$ is almost surely a constant. This is because no variation from the mean implies that $X$ takes on its mean value with probability 1.

These properties are widely used in statistical modeling, data analysis, and probability theory, especially in the derivation of statistical estimators, hypothesis testing, and in the study of the distributional properties of sums and transformations of random variables.


### 3. CLT simulation in Python for a centered continous uniform random distribution

```python=9

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.stats import norm

%matplotlib inline

# Set the random seed for reproducibility
np.random.seed(42)

# Parameters
sample_sizes = [1, 2, 10]
num_realizations = 100000
a, b = -np.sqrt(3), np.sqrt(3)

# Plot setup
fig, axes = plt.subplots(len(sample_sizes), 1, figsize=(10, 6), sharex=True)
plt.subplots_adjust(hspace=0.5)

# Perform simulations and plotting
for ax, n in zip(axes, sample_sizes):
    # Generate realizations of Z
    Z = (1 / np.sqrt(n)) * np.sum(np.random.uniform(a, b, size=(num_realizations, n)), axis=1)
    print(Z.shape)
    # Plot histogram of the realizations
    sns.histplot(Z, bins=30, kde=True, ax=ax, stat='density', label=f'n={n}')
    ax.set_title(f'Histogram of Z with n={n}')
    
    # Overlay the standard normal density for comparison
    x = np.linspace(min(Z), max(Z), 100)
    ax.plot(x, norm.pdf(x), 'r--', label='Standard Normal')
    ax.legend()

plt.suptitle('Density/Histogram of Z and Standard Normal Distribution')
plt.show()

```

![Screenshot 2024-03-15 at 3.50.42 AM](https://hackmd.io/_uploads/r1KmON-0T.png)

The plots demonstrate the Central Limit Theorem (CLT) in action for different sample sizes $n \in \{1, 2, 10\}$. Each subplot shows the distribution of $Z = \frac{1}{\sqrt{n}} \sum_{i=1}^n X_i$ for 100,000 realizations, where $X_i$ are drawn from the uniform distribution over $[-\sqrt{3},\sqrt{3})$. This distribution is overlayed with the density of the standard normal distribution (dotted red curve) for comparison.

Here's what we observe:

- For $n=1$, the histogram of $Z$ resembles the uniform distribution itself, since there's no averaging involved, and the transformation simply scales the distribution.
- For $n=2$, the histogram starts to show a more bell-shaped curve, indicating the beginning of the convergence towards a normal distribution as predicted by the CLT.
- For $n=10$, the histogram closely resembles the standard normal distribution, showcasing a significant convergence towards normality. This demonstrates the CLT, which states that the sum (or average) of a large number of independent, identically distributed random variables, regardless of the original distribution, will tend towards a normal distribution.

These observations align with the CLT, highlighting its power: as the number of samples increases, the distribution of the sum (or mean) of these samples increasingly approximates a normal distribution, even when the original variables are not normally distributed.

#### How does the number of realizations affect the histogram?

The number of realizations, or samples, you choose for generating a histogram affects the smoothness and accuracy of the representation of the underlying distribution. This principle applies not just to histograms of samples from a standard normal distribution, but to histograms of samples from any distribution. Here's how changing the number of realizations impacts the histogram:

#### More Realizations

- **Smoothness**: Increasing the number of realizations tends to produce a smoother histogram. This is because more data points allow for a more detailed and accurate approximation of the distribution's shape.
- **Accuracy**: With more samples, the histogram better approximates the true underlying distribution (in this case, the standard normal distribution). You're more likely to observe the classic bell curve shape of the normal distribution with a mean of 0 and a standard deviation of 1.
- **Stability**: The histogram becomes more stable with respect to random fluctuations. This means that if you were to repeat the experiment multiple times, the shape of the histogram would be more consistent across trials.

#### Fewer Realizations

- **Roughness**: Decreasing the number of realizations can lead to a rougher, more jagged histogram. This is because there are fewer data points to outline the distribution's shape, leading to a less accurate representation.
- **Inaccuracy**: With fewer samples, the histogram might not accurately capture the true characteristics of the underlying distribution. It might miss certain features or exaggerate others due to the limited data.
- **Instability**: The histogram becomes more susceptible to random fluctuations. Small sample sizes can lead to significant variability in the histogram's shape across different realizations of the experiment.

#### Practical Implications

In practical terms, choosing the number of realizations depends on the balance between computational resources and the need for accuracy. For exploratory data analysis or when computational resources are limited, a smaller number of realizations might suffice. However, for precise statistical analysis or when the goal is to closely approximate the underlying distribution, using a larger number of realizations is preferable.

It's also important to note that while increasing the number of realizations improves the approximation to the underlying distribution, it does not change the distribution itself. The Central Limit Theorem (CLT) ensures that, given enough samples, the distribution of sample means will approximate a normal distribution, independent of the number of realizations used to construct each individual histogram.

### 4. Distribution of Variances
The Central Limit Theorem (CLT) primarily applies to the distribution of sample means, stating that the distribution of the sample mean of a sufficiently large number of independent and identically distributed (i.i.d.) random variables, each with a finite mean and variance, will approximate a normal distribution, regardless of the shape of the original distribution.

For variances, the situation is slightly different. While the CLT does imply that sums (and by extension, averages) of i.i.d. random variables tend toward a normal distribution as the sample size increases, the distribution of sample variances follows a different path. Specifically, the distribution of sample variances (scaled appropriately) of a population is described by the Chi-square ($\chi^2$) distribution when the population is normally distributed ([see Appendix A.4. for details on $\chi^2$ square distribution](#A.4.-$\chi^2$-Square-Distribution)).

#### For Normally Distributed Data:

- When you draw samples from a normally distributed population, the sample variances (when properly scaled) follow a Chi-square distribution, not a normal distribution. The scaling involves dividing the sample variance by the population variance and then multiplying by the degrees of freedom ($n-1$ for a sample of size $n$). This scaled sample variance $\frac{(n-1)S^2}{\sigma^2}$ follows a Chi-square distribution with $n-1$ degrees of freedom, where $S^2$ is the sample variance, and $\sigma^2$ is the population variance.

#### For Non-Normally Distributed Data:

- The distribution of sample variances from non-normal populations does not necessarily follow a Chi-square distribution. The behavior of sample variances in this case can be more complicated and depends on the underlying distribution. However, for large sample sizes, various theorems (like the Central Limit Theorem for variances or similar results) suggest that the distribution of the variance estimator can approach normality under certain conditions, largely due to the fact that the variance itself can be considered as a sum of squared deviations, which are random variables.

#### General Consideration:

- The distribution of the sample variance is inherently related to the distribution of squared deviations from the mean. Since these squared deviations are not symmetric around zero (unlike the deviations themselves, which could be symmetric for a normal distribution), the distribution of sample variances does not inherently tend toward normality with increasing sample size in the same straightforward manner as the sample mean does.

In summary, while the CLT provides a basis for expecting the sample mean to be normally distributed for large sample sizes regardless of the population distribution, the sample variance follows a Chi-square distribution for normally distributed data and may approach normality under certain conditions for large sample sizes in non-normal populations, but this is not as directly assured as it is for sample means.

### Further links and resources

- On convergence speeds for different common distributions in CLT: https://david-salazar.github.io/posts/fat-vs-thin-tails/2020-05-30-central-limit-theorem-in-action.html
- Video on CLT from the well-known '3Blue1Brown' Youtube channel we all go to, when Professor's lecture flies above the head: https://www.youtube.com/watch?v=zeJD6dqJ5lo

# Fischer Information

The Fisher Information provides a measure of how much information an observable data sample carries about an unknown parameter of the model that generated the sample. It's essential for understanding the precision with which we can estimate these parameters.

- **Concept**: High Fisher Information indicates that the parameter estimate can be very precise. Low Fisher Information means the data provides little information about the parameter.


## 1. Likelihood


The likelihood function represents the plausibility of a parameter value given specific observed data. Unlike a probability function, which provides the probability of observing data given certain parameter values, the likelihood function considers the parameter (e.g., mean, variance) as variable and the data as fixed.

#### Example 1:

Suppose we are studying the heights of chicks in a particular town. We collect height data from a random sample of $n$ birds, and we wish to estimate the mean height μ for the entire population of chicks in the town. Assume that the heights are normally distributed, which is a reasonable assumption for biological measurements like height. In this example, the parameter μ represents the mean height we are trying to estimate. The likelihood function measures how "likely" or "plausible" different values of μ are given the observed data. The MLE is particularly powerful because it selects the value of μ that makes the observed data most probable under the assumed statistical model.

#### Example 2:
Suppose you have a coin, and you want to determine whether it's fair. You flip the coin ten times, and it comes up heads six times. The likelihood function in this scenario would help you evaluate how plausible different probabilities of flipping heads (let's denote this probability as $p$) are, given that you observed 6 heads out of 10 flips.

#### Mathematical Description

If you have a set of independent and identically distributed (i.i.d.) data points $X_1, X_2, \ldots, X_n$ from a probability distribution $f(x; \theta)$, where $\theta$ represents the parameters of the distribution, then the likelihood function $\mathcal{L}(\theta; X)$ is defined as the product of the probabilities (or probability densities for continuous data) of observing each specific $X_i$:

$$
\mathcal{L}(\theta; X) = f(X_1; \theta) \times f(X_2; \theta) \times \cdots \times f(X_n; \theta)
$$

In practice, especially with many data points, it's more convenient to work with the logarithm of the likelihood function, known as the log-likelihood function:

$$
\ln \mathcal{L}(\theta; X) = \sum_{i=1}^n \ln f(X_i; \theta)
$$

This transformation is useful because it turns the product into a sum, simplifying both computation and differentiation.

### 1.1 Relation to Fisher Information

Fisher Information quantifies how much information an observable random variable, sampled from a distribution, carries about an unknown parameter upon which the probability depends. Mathematically, Fisher Information is defined as the expected value of the squared gradient (first derivative) of the log-likelihood function [see Appendix A.5](#A.5.-Fischer-Information-as-Variance-of-Score-Function), or equivalently, as the negative expectation of the second derivative of the log-likelihood function with respect to the parameter:

$$
I_F(\theta) = -E\left[\frac{d^2}{d\theta^2} \ln f(X; \theta)\right]
$$

This definition implies that Fisher Information measures the steepness or curvature of the log-likelihood function around the parameter $\theta$. A steeper curve suggests that the parameter can be estimated with higher precision since small changes in $\theta$ lead to larger changes in the likelihood, making the maximum more distinct and easier to pinpoint accurately.

![image](https://hackmd.io/_uploads/Sy9y_BugR.png)


#### Interpretation and Application

- **Maximum Likelihood Estimation (MLE)**: In practice, we often use the likelihood function to find the parameter estimates that maximize the likelihood of observing the given data. These estimates are known as maximum likelihood estimates.
- **Sensitivity and Precision**: Higher Fisher Information for a parameter means that the data is more sensitive to changes in that parameter, implying that the parameter can be estimated with greater precision.

The likelihood function serves as a bridge between observed data and theoretical models, helping statisticians make inferences about unknown parameters. Fisher Information, derived from the likelihood function, plays a crucial role in assessing the quality and precision of these inferences.

### 1.2 Why is Fischer information negative of expectation E?

The negative sign in the Fisher Information formula, where Fisher Information is defined as the negative expectation of the second derivative of the log-likelihood function with respect to the parameter $\theta$, is a critical aspect to understand. The reason for this sign arises from the curvature of the log-likelihood function and its implications for parameter estimation.

#### Mathematical Perspective

1. **Second Derivative of the Log-Likelihood Function**: The second derivative of the log-likelihood function, $\frac{\partial^2}{\partial \theta^2} \ln \mathcal{L}(\theta)$, typically measures the curvature of the log-likelihood function at a particular point $\theta$. This curvature is crucial in determining the nature of the extremum (maximum or minimum) at that point.

2. **Convexity and Concavity**:
   - A **positive** second derivative at a point indicates that the function is **concave up** at that point, resembling a bowl. This shape generally implies a minimum point in a typical mathematical function.
   - A **negative** second derivative indicates that the function is **concave down** at that point, resembling a cap. This shape is typical of a maximum point in a standard function (e.g., **second derivative of $x^2$ - a concave curve - is 1 (positive) and vice versa for $-x^2$**).

Since we are often interested in maximizing the log-likelihood function to find the maximum likelihood estimators, the point of interest (where the first derivative is zero) will generally have a negative second derivative if it is a maximum.

#### Statistical Rationale

- **Maximization of Log-Likelihood**: In the context of likelihood estimation, we are interested in points where the log-likelihood function is maximized with respect to the parameter $\theta$. At these points, the curvature (second derivative) is negative, reflecting the concavity of the log-likelihood function.

- **Negative Expectation**: Given that the second derivative at the maximum point of the log-likelihood is negative, taking the negative of this expectation (i.e., the negative of a generally negative number) results in a positive value. Fisher Information must be positive as it quantifies the amount of information the data carries about the parameter, where higher values imply more information or precision in the estimation.

#### Why is Positive Fisher Information Important?

- **Variance of Estimators**: Positive Fisher Information is crucial because it relates directly to the precision of estimators through the Cramér-Rao Lower Bound. According to this bound, the variance of any unbiased estimator is at least the reciprocal of the Fisher Information:
  
  $$
  \text{Var}(\hat{\theta}) \geq \frac{1}{I_F(\theta)}
  $$

  If Fisher Information were not positive (and substantial), this fundamental relationship, which guarantees a lower bound on the variance of estimators, would not hold, undermining the statistical inference process.

In summary, the negative sign in the definition of Fisher Information as the negative of the expected value of the second derivative of the log-likelihood function is necessary to ensure that Fisher Information is a positive quantity, reflecting the concavity of the log-likelihood at its maximum and the precision achievable in parameter estimation.

## 2. Fischer Information of $n$ i.i.d. Samples
To calculate the Fisher Information $I_F^{\mathcal L}(\theta)$ of the likelihood function for $n$ i.i.d samples $\{x_1, \ldots, x_n\}$ from the given probability mass function (pmf) $f(x; \theta)$, we start with the given likelihood function:

$$
\mathcal{L}(\{x_1, \ldots, x_n\}; \theta) = \prod_{i=1}^{n} f(x_i; \theta)
$$

First, we need to compute the logarithm of the likelihood function, known as the log-likelihood function:

$$
\ln \mathcal{L}(\{x_1, \ldots, x_n\}; \theta) = \ln \left(\prod_{i=1}^{n} f(x_i; \theta)\right) = \sum_{i=1}^{n} \ln f(x_i; \theta)
$$

Next, we differentiate the log-likelihood function twice with respect to $\theta$ and then compute the expectation (the Fisher Information of the likelihood function):

$$
I_F^{\mathcal L}(\theta) = -\mathbb{E}_X\left[\frac{\partial^2}{\partial \theta^2} \ln \mathcal{L}(\{x_1, \ldots, x_n\}; \theta)\right]
$$

Since the second derivative of the sum of the log-likelihoods is the sum of the second derivatives of the individual log-likelihoods, and knowing that $I_F(\theta)$ is the Fisher Information of one sample ([see Appendix A.6 on property below](#A.6.-Property-of-Expectation)):

$$
I_F^{\mathcal L}(\theta) = -\mathbb{E}_X\left[\sum_{i=1}^{n} \frac{\partial^2}{\partial \theta^2} \ln f(x_i; \theta)\right] = \sum_{i=1}^{n} -\mathbb{E}_X\left[\frac{\partial^2}{\partial \theta^2} \ln f(x_i; \theta)\right]
$$

Given that $I_F(\theta) = -\mathbb{E}_X\left[\frac{\partial^2}{\partial \theta^2} \ln f(x; \theta)\right]$, we can substitute this into the equation for each $i$, recognizing that each term in the sum is just $I_F(\theta)$:

$$
I_F^{\mathcal L}(\theta) = \sum_{i=1}^{n} I_F(\theta) = n \cdot I_F(\theta)
$$


The Fisher Information $I_F^{\mathcal L}(\theta)$ of the likelihood function for $n$ i.i.d samples is $n$ times the Fisher Information $I_F(\theta)$ of a single sample. This result indicates that the amount of information about the parameter $\theta$ contained in $n$ i.i.d samples is $n$ times the information contained in a single sample. Essentially, as the sample size $n$ increases, the total information about $\theta$ increases linearly with $n$, implying that larger sample sizes provide more precise estimates of the parameter $\theta$, as reflected in the decrease in the variance of the estimator.

## Further links and resources

- A detailed analysis on Fischer Information: https://awni.github.io/intro-fisher-information/
- A sane introduction to maximum likelihood estimation (MLE) and maximum a posteriori (MAP): https://blog.christianperone.com/2019/01/mle/
- On Estimators, Loss Functions, Optimizers: https://towardsdatascience.com/estimators-loss-functions-optimizers-core-of-ml-algorithms-d603f6b0161a
- Visualization based explanation on MLE: https://www.youtube.com/watch?v=sguol03tfWo


# Appendix

### A.1. Derivation of $Var(Y)=a^2 Var(X)$

The variance of a random variable measures the dispersion of that variable's values around its mean. The formula for the variance of a random variable $X$ is defined as:

$$
Var(X) = E[(X - E[X])^2]
$$

where $E[X]$ is the expected value (or mean) of $X$, and $E$ denotes the expectation operator.

Now, let's consider a new random variable $Y = aX$, where $a$ is a constant. We want to derive the variance of $Y$, denoted as $Var(Y)$ or $Var(aX)$.

#### Step 1: Define $Y = aX$

Given $Y = aX$, we apply the variance formula:

$$
Var(Y) = E[(Y - E[Y])^2]
$$

Since $Y = aX$, we have $E[Y] = E[aX]$.

#### Step 2: Calculate $E[Y]$

The expected value of $Y$ is:

$$
E[Y] = E[aX] = aE[X]
$$

This is because the expectation operator is linear, and the constant $a$ can be factored out of the expectation.

#### Step 3: Plug $E[Y]$ into the Variance Formula

Substituting $Y = aX$ and $E[Y] = aE[X]$ into the variance formula, we get:

$$
Var(Y) = E[((aX) - aE[X])^2] = E[(a(X - E[X]))^2]
$$

#### Step 4: Simplify the Expression

Since $a$ is a constant, we can factor it out of the squared term:

$$
Var(Y) = E[a^2(X - E[X])^2] = a^2 E[(X - E[X])^2]
$$

Noting that $E[(X - E[X])^2]$ is the definition of $Var(X)$, we have:

$$
Var(Y) = a^2 Var(X)
$$


This derivation shows that the variance of $Y = aX$, where $a$ is a constant, is $a^2$ times the variance of $X$. The key takeaway is that scaling a random variable by a constant $a$ scales its variance by $a^2$, reflecting the squared nature of variance as a measure of dispersion.

### A.2. Standardization or $Z$-Score

Standardization is a statistical method used to transform random variables into a standard scale without distorting differences in the ranges of values. The process converts original data into a format where the mean of the transformed data is 0 and the standard deviation is 1. This transformation is achieved by subtracting the expected value (mean) from each data point and then dividing by the standard deviation.

#### The Formula

The formula for standardizing a random variable $X$ is:

$$
Z = \frac{X - \mu}{\sigma}
$$

It is basically the deviation of data point from mean (i.e. how far the data point is from the mean) per unit standard deviation. For e.g, $Z = 2$ means that the data point is 2 standard deviation away from the mean. 

where:
- $X$ is the original random variable,
- $\mu$ is the mean of $X$,
- $\sigma$ is the standard deviation of $X$, and
- $Z$ is the standardized variable.

#### Why Use Standardization?

The rationale behind standardization and the specific form of the standardization formula involves several key statistical principles:

1. **Comparability**: Standardization allows data from different sources or distributions to be compared directly. Because the standardized data has a mean of 0 and a standard deviation of 1, it removes the units of measurement and normalizes the scale, making different datasets or variables comparable.

2. **Normalization**: Many statistical methods and machine learning algorithms assume or perform better when the data is normally distributed or similarly scaled. Standardization can help meet these assumptions or improve performance by giving every variable an equal weight, preventing variables with larger scales from dominating those with smaller scales.

3. **Understanding Z-scores**: The standardized value, or **Z-score, tells you how many standard deviations away from the mean a data point is.** This can be useful for identifying outliers, understanding the distribution of data, and performing statistical tests.

4. **Mathematical Foundation**: The formula is grounded in the properties of the normal distribution. In a standard normal distribution, the mean ($\mu$) is 0, and the standard deviation ($\sigma$) is 1. The standardization process transforms the data so that it can be described in terms of how far each observation is from the mean, in units of the standard deviation. This transformation is particularly useful in the context of the Central Limit Theorem, which states that the distribution of the sample means tends towards a normal distribution as the sample size increases, regardless of the shape of the population distribution.

#### Conclusion

The act of subtracting the mean and dividing by the standard deviation in standardization serves to "normalize" the scale of different variables, enabling direct comparison, simplifying the interpretation of data, and preparing data for further statistical analysis or machine learning modeling. This process leverages the fundamental statistical properties of mean and standard deviation to achieve a standardized scale, where the effects of differing magnitudes among original data values are neutralized.

### A.3. Derivation of Normal Distribution from Central Limit Theorem

Deriving the normal distribution mathematically from the Central Limit Theorem (CLT) in a simple, non-technical explanation is challenging due to the advanced mathematical concepts involved, particularly the use of characteristic functions or moment-generating functions. However, I'll outline a basic approach using characteristic functions to give you a sense of how the derivation works. This explanation simplifies several steps and assumes some familiarity with concepts from probability theory.

#### Step 1: Understanding Characteristic Functions

The characteristic function $\phi_X(t)$ of a random variable $X$ is defined as the expected value of $e^{itX}$, where $i$ is the imaginary unit and $t$ is a real number:

$$
\phi_X(t) = E[e^{itX}]
$$

Characteristic functions are powerful tools in probability theory because they uniquely determine the distribution of a random variable, and they have properties that make them particularly useful for analyzing sums of independent random variables.

#### Step 2: The Characteristic Function of the Sum of Independent Variables

Consider $n$ independent and identically distributed (i.i.d.) random variables $X_1, X_2, ..., X_n$, each with mean $\mu$ and variance $\sigma^2$. Let $S_n = X_1 + X_2 + ... + X_n$ be their sum. The characteristic function of $S_n$ is:

$$
\phi_{S_n}(t) = \left(\phi_X(t)\right)^n
$$

This is because the characteristic function of a sum of independent variables is the product of their individual characteristic functions.


#### Step 3: Standardizing $S_n$

First, standardize $S_n$ to get $Z_n$:

$$
Z_n = \frac{S_n - n\mu}{\sigma\sqrt{n}}
$$

#### Characteristic Function of $Z_n$

We want to find the characteristic function of $Z_n$, $\phi_{Z_n}(t)$. The characteristic function for $S_n$ is $\phi_{S_n}(t) = (\phi_X(t))^n$, since the variables are i.i.d.

Given the Taylor expansion of $\phi_X(t)$ around 0:

$$
\phi_X(t) \approx 1 + it\mu - \frac{t^2\sigma^2}{2} + \text{higher-order terms}
$$

#### Adjusting for $Z_n$

To adjust this for $Z_n$, note that we're interested in the effect of $t$ on $Z_n$, not on the original variables. The transformation involves a shift and scaling of $t$, considering the definition of $Z_n$. So, we replace $t$ with $\frac{t}{\sigma\sqrt{n}}$ to reflect the scaling in $Z_n$ and consider the subtraction of $n\mu$, which shifts the mean to 0:

$$
\phi_{Z_n}(t) = E\left[e^{it\frac{S_n - n\mu}{\sigma\sqrt{n}}}\right] = \left(\phi_X\left(\frac{t}{\sigma\sqrt{n}}\right)\right)^n \cdot e^{-it\mu\sqrt{n}/\sigma}
$$

Substituting the approximation for $\phi_X\left(\frac{t}{\sigma\sqrt{n}}\right)$ and simplifying, we aim to show that this converges to $e^{-t^2/2}$ as $n \rightarrow \infty$.

#### Applying the Approximation and Taking the Limit

When you substitute the Taylor expansion into the expression for $\phi_{Z_n}(t)$ and simplify, focusing on terms up to the second order, you essentially deal with:

$$
\left(1 + i\frac{t}{\sigma\sqrt{n}}\mu - \frac{\left(\frac{t}{\sigma\sqrt{n}}\right)^2\sigma^2}{2}\right)^n
$$

Since $\mu$ is the mean of the original distribution, and we're considering the sum $S_n$ minus $n\mu$, adjusted by $\sigma\sqrt{n}$, this simplifies to:

$$
\left(1 - \frac{t^2}{2n}\right)^n
$$

As $n \rightarrow \infty$, this expression converges to $e^{-t^2/2}$, by the limit definition of the exponential function:

$$
\lim_{n \to \infty} \left(1 - \frac{t^2}{2n}\right)^n = e^{-t^2/2}
$$

#### Final Step: Connection to the Standard Normal Distribution

This result, $e^{-t^2/2}$, is the characteristic function of a standard normal distribution $N(0,1)$ (i.e., a mean $(\mu$) of 0 and a standard deviation ($\sigma$) of 1). The inverse Fourier transform (or the characteristic function inversion theorem) tells us that the probability density function corresponding to this characteristic function is the PDF of the standard normal distribution:

$$
f(z) = \frac{1}{\sqrt{2\pi}} e^{-\frac{z^2}{2}}
$$

This formula describes the distribution of values that $Z$ can take, where $Z$ represents the number of standard deviations away from the mean a particular observation is. The factor $\frac{1}{\sqrt{2\pi}}$ normalizes the area under the curve of the PDF to 1, ensuring that the total probability across all possible outcomes is 1, as required for any probability distribution.

This demonstrates how, under the Central Limit Theorem, the distribution of the standardized sum (or average) of a large number of i.i.d. random variables, regardless of their original distribution, converges to a normal distribution, provided the original variables have a finite mean and variance.

This derivation, while not delving into the full technical rigor of the proofs involving characteristic functions, provides a conceptual bridge from the CLT to the emergence of the normal distribution.

### A.4. $\chi^2$ Square Distribution

The Chi-square distribution is a widely used probability distribution in statistical inference, particularly in hypothesis testing and in constructing confidence intervals. It arises primarily in contexts involving the sum of squared independent, standard normal variables.


The Chi-square distribution with $k$ degrees of freedom is defined as the distribution of a sum of the squares of $k$ independent standard normal random variables. Mathematically, if $Z_1, Z_2, \ldots, Z_k$ are independent and identically distributed (i.i.d.) standard normal random variables ($N(0,1)$), then the random variable

$$
Q = Z_1^2 + Z_2^2 + \ldots + Z_k^2
$$

follows a Chi-square distribution with $k$ degrees of freedom, denoted as $Q \sim \chi^2(k)$.

#### Probability Density Function (PDF)

The probability density function of the Chi-square distribution for $x \geq 0$ and $k$ degrees of freedom is given by:

$$
f(x; k) = \frac{1}{2^{k/2} \Gamma(k/2)} x^{k/2 - 1} e^{-x/2}
$$

where:
- $x$ is the variable,
- $k$ is the degrees of freedom,
- $\Gamma(\cdot)$ is the Gamma function, which generalizes the factorial function with $\Gamma(n) = (n-1)!$ for integer $n$.

#### Characteristics

- **Mean and Variance**: The mean of the Chi-square distribution is $k$ (equal to the degrees of freedom), and the variance is $2k$.
- **Skewness**: The distribution is positively skewed, and the skewness decreases as the degrees of freedom increase. As $k$ becomes large, the Chi-square distribution approaches a normal distribution due to the Central Limit Theorem, as it is the sum of $k$ squared normal variables.
- **Applications**: The Chi-square distribution is used extensively in hypothesis testing (especially for tests of independence in contingency tables and goodness-of-fit tests) and in estimating variances of normal distributions in confidence intervals and tests.

The Chi-square function is crucial in fields like biology, finance, and physics, where it helps in decision-making processes involving uncertainty and variability.

![Screenshot 2024-04-13 at 3.10.25 PM](https://hackmd.io/_uploads/SkVF4WugA.png)

Here's the plot showing Chi-square distributions for various degrees of freedom (df). The degrees of freedom were chosen as 2, 5, and 10 for this illustration. As you can see:

- With **df=2**, the distribution starts off high at the lower end and quickly tapers off, indicating a high probability of low values and rapidly diminishing probability for higher values.
- As the **degrees of freedom increase (df=5 and df=10)**, the peak of the distribution shifts to the right, and the shape becomes more symmetrical. This shift reflects the increasing mean of the distribution with higher degrees of freedom, and the distribution becomes less skewed.
- The **spread** of the distribution also increases with the degrees of freedom, showing greater variability in potential chi-square values as df increases.

This visualization helps in understanding how the shape of the Chi-square distribution is influenced by its degrees of freedom, with greater degrees of freedom leading to a more pronounced and symmetrical shape.

### A.5. Fischer Information as Variance of Score Function

The Fisher Information is a crucial concept in statistical inference, intimately linked to the variability in the score function of the log-likelihood, where the score function represents the first derivative of the log-likelihood with respect to the parameter $\theta$. Here, Fisher Information is portrayed not just as the variance of the score function, but also through a deeper mathematical connection to the curvature of the log-likelihood function.

**Note**: This score function is different from Z-Score in Appendix A.3. Z-Score is applied to normalize the data, whereas the purpose of score function here is to find the extrema (maximum) of log-likelihood.

### Score Function

The score function $U(\theta)$ is formally defined as the derivative of the log-likelihood function $\ln \mathcal{L}(\theta)$ with respect to the parameter $\theta$:

$$
U(\theta) = \frac{\partial}{\partial \theta} \ln \mathcal{L}(\theta)
$$

This function measures the sensitivity of the likelihood function to changes in the parameter $\theta$, thus indicating how much information the observed data provide about $\theta$.

#### Fisher Information and Its Mathematical Basis

Fisher Information, $I_F(\theta)$, can be understood as the variance of the score function, expressed mathematically as:

$$
I_F(\theta) = \mathbb{E}\left[ \left( U(\theta) \right)^2 \right] = \mathbb{E}\left[ \left( \frac{\partial}{\partial \theta} \ln \mathcal{L}(\theta) \right)^2 \right]
$$

However, a more profound expression of Fisher Information comes from the second derivative of the log-likelihood function:

$$
I_F(\theta) = -\mathbb{E}\left[\frac{\partial^2}{\partial \theta^2} \ln \mathcal{L}(\theta)\right]
$$

This formulation shows that Fisher Information is the negative expectation of the second derivative (or the Laplacian) of the log-likelihood function. It characterizes the curvature of the log-likelihood function around the parameter $\theta$. A sharper curvature (higher absolute value of the second derivative) implies more information about $\theta$ is available from the data, suggesting that estimates of $\theta$ can be made more precisely.

#### Interpretation and Practical Implications

- **Information Content**: Fisher Information quantifies the amount of information that the sample data ($X$) provides about the parameter ($\theta$). A higher Fisher Information suggests that small changes in $\theta$ produce substantial changes in the likelihood, indicating that the data is highly informative regarding $\theta$.

- **Precision of Estimators**: Fisher Information is inversely related to the variance of any unbiased estimator of $\theta$. This relationship is crystallized in the Cramér-Rao Lower Bound, which states that the variance of any unbiased estimator of $\theta$ must be at least as great as the reciprocal of the Fisher Information:

$$
\text{Var}(\hat{\theta}) \geq \frac{1}{I_F(\theta)}
$$

Thus, greater Fisher Information leads to a lower bound on the variance of the estimator, enhancing the precision with which $\theta$ is estimated.

In essence, the Fisher Information not only provides a measure of the expected sharpness of the log-likelihood function's peak but also fundamentally ties to how confidently parameters can be estimated from given data. This relationship underscores the vital role of Fisher Information in both theoretical statistics and practical data analysis.


### A.6. Property of Expectation

$$
I_F^{\mathcal L}(\theta) = -\mathbb{E}_X\left[\sum_{i=1}^{n} \frac{\partial^2}{\partial \theta^2} \ln f(x_i; \theta)\right] = \sum_{i=1}^{n} -\mathbb{E}_X\left[\frac{\partial^2}{\partial \theta^2} \ln f(x_i; \theta)\right]
$$

The expectation of a sum of random variables is equal to the sum of their expectations. This is why the expectation of the sum of the second derivatives can be represented as the sum of the expectations of the second derivatives. $\sum_{i=1}^{n} -\mathbb{E}_X\left[\frac{\partial^2}{\partial \theta^2} \ln f(x_i; \theta)\right]$ makes explicit that each term in the sum is processed individually through the expectation operator. This emphasizes that Fisher Information for the entire dataset can be viewed as the sum of the individual Fisher Informations from each data point.

#### Why This Formulation?

This formulation shows how the Fisher Information for a model based on multiple i.i.d. observations aggregates information from each observation. Since the observations are i.i.d., the information they provide about $\theta$ is additive. The Fisher Information from each observation contributes to the total information available in the sample about the parameter $\theta$.

**Note: The statement "The expectation of a sum of random variables is equal to the sum of their expectations" is a fundamental property in probability theory known as the **linearity of expectation**. This property holds regardless of whether the random variables are independent or not.** 

Let's describe this mathematically:


Suppose $X_1, X_2, \ldots, X_n$ are random variables. Then, the expectation of their sum is:

$$
\mathbb{E}\left[\sum_{i=1}^n X_i\right]
$$

By the linearity of expectation, this can be written as:

$$
\mathbb{E}\left[\sum_{i=1}^n X_i\right] = \sum_{i=1}^n \mathbb{E}[X_i]
$$

#### Proof (For the General Case)

To see why this is true, consider the definition of the expected value for discrete random variables (the proof is similar for continuous random variables, using integrals instead of sums). The expected value of a random variable $X$ is given by:

$$
\mathbb{E}[X] = \sum_{x} x \cdot P(X = x)
$$

Where the sum is taken over all possible values $x$ that $X$ can take, and $P(X = x)$ is the probability that $X$ takes the value $x$.

For the sum of two random variables $X_1$ and $X_2$, the expectation is:

$$
\mathbb{E}[X_1 + X_2] = \sum_{x_1, x_2} (x_1 + x_2) \cdot P(X_1 = x_1 \text{ and } X_2 = x_2)
$$

Using the distributive property of multiplication over addition, this becomes:

$$
\mathbb{E}[X_1 + X_2] = \sum_{x_1, x_2} x_1 \cdot P(X_1 = x_1 \text{ and } X_2 = x_2) + \sum_{x_1, x_2} x_2 \cdot P(X_1 = x_1 \text{ and } X_2 = x_2)
$$


> Here, $P(X_1 = x_1)$ is the marginal probability of $X_1$, which is obtained by summing the joint probabilities over all possible values of $X_2$:
>    $$
>    P(X_1 = x_1) = \sum_{x_2} P(X_1 = x_1 \text{ and } X_2 = x_2)
>    $$
> 
> Using the marginal probability, we rewrite the expectation:
>    $$
>    \mathbb{E}[X_1] = \sum_{x_1} x_1 \left(\sum_{x_2} P(X_1 = x_1 \text{ and } X_2 = x_2)\right) = \sum_{x_1, x_2} x_1 \cdot P(X_1 = x_1 \text{ and } X_2 = x_2)
>    $$
>    Here, $x_1$ is factored out of the inner sum because it does not depend on $x_2$, making the expression equivalent to summing $x_1$ times the joint probability over all $x_1, x_2$ pairs.


Now, each of these sums can be separated into the sums over $x_1$ and $x_2$ respectively, which simplifies to:

$$
\mathbb{E}[X_1 + X_2] = \mathbb{E}[X_1] + \mathbb{E}[X_2]
$$

This simplification relies on the fact that you can rearrange the sums because the sum of the joint probabilities over one variable $x_1$ or $x_2$ for all values of the other yields the marginal probability of $x_1$ or $x_2$.

#### Extension to $n$ Variables

The argument for two variables can be extended inductively to any finite number $n$ of random variables. For $n$ variables $X_1, X_2, \ldots, X_n$:

$$
\mathbb{E}\left[\sum_{i=1}^n X_i\right] = \sum_{i=1}^n \mathbb{E}[X_i]
$$

This linearity property is extremely useful because it simplifies calculations of expectations in complex situations involving sums of random variables and is a cornerstone in fields such as statistics, finance, and other areas of applied mathematics and engineering.
