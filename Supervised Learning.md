# Supervised Learning

[![hackmd-github-sync-badge](https://hackmd.io/UYVlCTTJTkGVYQa5O9x6lg/badge)](https://hackmd.io/UYVlCTTJTkGVYQa5O9x6lg)


**Note:** See Sect. 3 of cl_en.pdf in Computational Intelligence course for introduction on Linear Regression.
## 1. Linear Regression

Linear regression is a statistical method used in supervised learning where the goal is to predict the value of a dependent variable $t$ based on the values of one or more independent variables $x$. The relationship between the dependent variable and the independent variables is assumed to be linear, i.e., it can be described using a linear function.

The general form of a linear regression model is given by:

$$
\mathbf{t} = \mathbf{w}^\top \mathbf{x} + \epsilon
$$

where:
- $\mathbf{x}$ is the vector of input features.
- $\mathbf{w}$ is the vector of weights that needs to be learned.
- $\epsilon$ represents the noise or error term, which accounts for the fact that the relationship between inputs and outputs may not be perfectly linear or may be affected by other variables not included in the model.

### 1.1 Design Matrix

A design matrix, often used in statistics and machine learning, is a matrix of data in which rows represent individual observations and columns represent the features or predictors associated with those observations. This matrix is crucial in various modeling techniques, particularly in regression analysis, where it helps to organize the data for computational efficiency and clarity in the formulation of models. Here's a deeper look into its components and uses:

#### Components of a Design Matrix

1. **Observations**: Each row in the design matrix corresponds to a single observation or data point in the dataset.
2. **Features**: Each column represents a feature or a predictor variable. For polynomial regression, these features could be various powers of a single variable, and for multiple regression, these would be different variables influencing the outcome.
3. **Intercept Term**: Often, a column of ones is included as the first column of the design matrix to account for the intercept term in linear models. This column enables the linear equation to have a constant term, which shifts the line or hyperplane up or down.

#### Uses in Regression Analysis

In regression analysis, the design matrix $\Phi$ is used to map the relationship between the input features (independent variables) and the target (dependent variable) through a linear predictor function. The general form of a linear regression model can be written as:

$$ \mathbf{t} = \Phi \mathbf{w} + \mathbf{\epsilon} $$

Where:
- $\mathbf{t}$ is the vector of target values.
- $\Phi$ is the design matrix.
- $\mathbf{w}$ is the vector of weights or coefficients, including the intercept if an intercept column is present.
- $\mathbf{\epsilon}$ is the vector of errors or residuals.


#### 1.1.1 How is the design matrix based on Sigmoidal basis different from the design matrix on Polynomial basis?

- **Polynomial Basis**: In a polynomial basis design matrix, each column represents a power of the input feature. For a single input feature $x$ and a polynomial degree $D$, the columns of the design matrix $\Phi$ are $[1, x, x^2, x^3, ..., x^D]$. This representation models the relationship between the input and the output as a polynomial equation. The higher the degree, the more complex the curves that the model can fit, but it also increases the risk of overfitting.

- **Sigmoidal Basis**: A sigmoidal basis design matrix uses sigmoid functions, typically logistic functions, to transform the input feature. Each column in the design matrix $\Phi$ is of the form $\sigma\left(\frac{x - c_i}{s}\right)$, where $c_i$ is the center of the $i$-th sigmoid function and $s$ is its width. This basis function is particularly useful for capturing the presence of thresholds or transitions at specific points in the input range, leading to models that can represent step-like features or sudden changes in behavior ([see Appendix A.1. for details on Sigmoidal Function](#A.1.-Sigmoidal-Function))

#### **Behavior and Applications**

- **Polynomial Basis**:
  - **Global Influence**: Each polynomial term affects the entire range of the input feature. A change in one parameter of the polynomial affects the model predictions over the entire range of the input feature.
  - **Risk of Oscillations**: High-degree polynomials can lead to oscillatory behavior, especially near the boundaries of the data range, known as Runge's phenomenon.
  - **Best for**: Modeling smooth, continuous, and globally varying phenomena without abrupt changes.

- **Sigmoidal Basis**:
  - **Local Influence**: Sigmoid functions are mostly influenced by values near their center $c_i$; they transition from 0 to 1 around this point, making their impact local rather than global.
  - **Step-like Features**: They are excellent at modeling features that exhibit threshold effects or are expected to change behavior abruptly.
  - **Best for**: Problems where the influence of input features changes drastically at certain values, common in classification thresholds and in certain types of segmented regression models.

#### **Flexibility and Fitting**

- **Polynomial Basis**: While flexible, polynomials can become unwieldy and prone to overfitting as their degree increases, especially if the data does not inherently follow a polynomial trend.

- **Sigmoidal Basis**: Offers fine-grained control over where the model should be sensitive to changes in the input features, allowing for more nuanced modeling of complex behaviors. However, determining the right placement and scale of sigmoid functions can be challenging and may require domain knowledge or experimentation.

#### **Computation and Interpretation**

- **Polynomial Basis**: Easier to compute and interpret in terms of traditional curve fitting. The coefficients directly relate to the influence of powers of the input feature.

- **Sigmoidal Basis**: Computation can be more intensive due to the nature of the exponential function in the sigmoid. The interpretation is more about the importance and impact of specific ranges of input values rather than overall trends.

In summary, the choice between a polynomial and a sigmoidal basis in constructing a design matrix largely depends on the nature of the data and the specific characteristics of the phenomenon being modeled. Each offers unique advantages and challenges that make them suited to different types of regression problems.

### 1.2 Computation of $\mathbf{w}$ with and without Regularization

The optimal weight vector computed with regularization (often referred to in machine learning and statistics as ridge regression when using L2 regularization) differs from the one computed without regularization in how it manages the complexity and potential overfitting of the model. Let’s explore these differences and the role of the regularization parameter:

#### Optimal Weight Vector Without Regularization
The optimal weight vector without regularization is computed using the simple least squares approach. This method minimizes the sum of the squared differences between the observed targets and the targets predicted by the linear model. Mathematically, it is computed as:
$$ \mathbf{w}^* = (\Phi^T \Phi)^{-1} \Phi^T \mathbf{y} $$
Where:
- $\Phi$ is the design matrix.
- $\mathbf{y}$ is the vector of target values.
- $\mathbf{w}^*$ represents the set of coefficients (weights) that best fit the model to the data.

#### Optimal Weight Vector With Regularization
When regularization is introduced, the equation is modified to include a regularization term $\lambda$ that penalizes the magnitude of the weight coefficients. This approach is aimed at reducing overfitting by discouraging overly complex models:
$$ \mathbf{w}^*_{\text{reg}} = (\Phi^T \Phi + \lambda I)^{-1} \Phi^T \mathbf{y} $$
Here:
- $\lambda$ is the regularization parameter.
- $I$ is the identity matrix of appropriate dimension.
- The term $\lambda I$ adds a penalty for large weights to the minimization problem, thus constraining them.

#### How Regularization Affects the Model
1. **Bias-Variance Trade-off**: Regularization helps manage the bias-variance tradeoff. By introducing bias into the model (since it does not fit the training data perfectly), regularization helps to reduce variance and thus improves the model’s performance on unseen data.

2. **Prevention of Overfitting**: By penalizing large weights, regularization effectively limits the model's ability to fit the noise in the training data, focusing instead on capturing the underlying patterns.

3. **Impact on Coefficients**: Regularized models typically have smaller absolute values of coefficients compared to non-regularized models. This often results in a smoother model function that is less likely to capture high-frequency fluctuations (noise) in the data.

#### Definition and Selection of the Regularization Parameter ($\lambda$)
- **Definition**: The regularization parameter $\lambda$ controls the strength of the penalty applied to the size of the coefficients. A larger $\lambda$ increases the penalty, thus forcing coefficients to be smaller, whereas a smaller $\lambda$ approaches the non-regularized least squares solution.
- **Selection**: Choosing the right value for $\lambda$ is critical and is usually done via:
  - **Cross-validation**: A common method where different values of $\lambda$ are tested, and the one that results in the best performance on a validation set (or through cross-validation scores) is chosen.
  - **Analytical Criteria**: Techniques such as AIC (Akaike Information Criterion), BIC (Bayesian Information Criterion), or other regularization paths (e.g., Lasso path) may also be used.

Regularization is a powerful concept in machine learning and statistics, helping to create more generalizable models, especially when dealing with high-dimensional data or situations where the number of features might approach or exceed the number of observations.

### Minimizing the Cost Function (Sum of Squared Errors)

The primary objective in linear regression is to find the weight vector $\mathbf{w}$ that minimizes the difference between the predicted outputs and the actual outputs in the training data. This difference is quantified using a cost function, commonly the sum of squared errors:

$$
E(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^N (t_n - \mathbf{w}^\top \mathbf{x}_n)^2
$$

where:
- $t_n$ is the actual output value for the nth sample.
- $\mathbf{x}_n$ is the input feature vector for the nth sample.
- $N$ is the total number of samples.

**In many contexts within regression analysis, especially in the formulation of the error function, a factor of $\frac{1}{2}$ is often included.** This factor simplifies the derivation of the gradient and optimization steps because it cancels out the 2 that appears when differentiating squared terms. The inclusion of this factor does not affect the location of the minimum; it merely scales the error value, making derivative computations slightly cleaner.

#### Adjusted Sum of Squared Errors

If we incorporate the $\frac{1}{2}$ factor, the adjusted sum of squared errors (SSE) formula becomes:

$$ \text{SSE}(\mathbf{w}) = \frac{1}{2} \sum_{i=1}^N (t_i - \mathbf{x}_i^\top \mathbf{w})^2 $$

#### Adjusted Regularized Sum of Squared Errors

Similarly, the regularized sum of squared errors with the $\frac{1}{2}$ factor included would look like:

$$ \text{SSE}_{\text{reg}}(\mathbf{w}) = \frac{1}{2} \sum_{i=1}^N (t_i - \mathbf{x}_i^\top \mathbf{w})^2 + \frac{\lambda}{2} \|\mathbf{w}\|^2 $$




### Probabilistic Perspective

From a probabilistic perspective, assuming the noise $\epsilon$ is normally distributed with mean zero and variance $\sigma^2$, the likelihood of observing the target values given the inputs and the model parameters (weights) can be modeled by the normal distribution:

$$
p(\mathbf{t} | \mathbf{X}, \mathbf{w}, \sigma^2) = \prod_{n=1}^N \mathcal{N}(t_n | \mathbf{w}^\top \mathbf{x}_n, \sigma^2)
$$

Here, $\mathcal{N}$ denotes the normal distribution, $\mathbf{t}$ is the vector of all target values, and $\mathbf{X}$ is the matrix of input features.

#### Solution

The weights $\mathbf{w}$ that maximize this likelihood (equivalently minimize the negative log-likelihood) can be found analytically by solving the normal equations:

$$
\mathbf{w} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{t}
$$

This solution involves computing the pseudo-inverse of the matrix $\mathbf{X}^\top \mathbf{X}$, which assumes that this matrix is non-singular (i.e., has full rank).

### Conclusion

Linear regression provides a straightforward method for modeling the linear relationships between variables. It is widely used due to its simplicity, interpretability, and the fact that it can be applied to a variety of real-world problems. However, it assumes that the relationship between the dependent and independent variables is linear, which might not always hold, potentially limiting its effectiveness in complex scenarios where relationships might be non-linear.

## 3. Decision Tree

- Hyperparameters in Decision Tree: https://ken-hoffman.medium.com/decision-tree-hyperparameters-explained-49158ee1268e

# Appendix

## A.1 Sigmoidal Function

The sigmoid function is a mathematical function that produces a characteristic "S"-shaped curve, also known as a sigmoid curve. This function is widely used in various fields, especially in statistics, machine learning, and artificial intelligence, mainly because of its ability to map a wide range of input values to a small and specific range, typically between 0 and 1. This makes it very useful for tasks like transforming arbitrary real-valued numbers into probabilities, which is crucial in logistic regression and neural networks.

#### Basic Form of the Sigmoid Function

The basic form of the sigmoid function, often referred to as the logistic function, is defined as:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

Here’s what happens in this function:

- **$x$** is the input value.
- **$e$** is the base of the natural logarithm, approximately equal to 2.71828.
- **$e^{-x}$** is the exponential function that rapidly decreases as $x$ increases, making the denominator $1 + e^{-x}$ grow.

As $x$ increases, $e^{-x}$ decreases, making the overall value of $\sigma(x)$ increase towards 1. Conversely, as $x$ decreases, $e^{-x}$ increases, pushing $\sigma(x)$ towards 0.

#### Characteristics of the Sigmoid Function

- **Output range**: The output of the sigmoid function is always between 0 and 1, which makes it particularly useful for applications that require a probabilistic interpretation.
- **Shape**: The function smoothly transitions from 0 to 1, with an inflection point at $x = 0$, where $\sigma(x) = 0.5$. This inflection point is where the output growth rate changes from increasing to decreasing.
- **Asymptotes**: The function approaches the limits of 0 and 1 as $x$ moves towards negative and positive infinity, respectively.

#### Sigmoid Function in the Context of Sigmoidal Basis Functions

In the context of using sigmoid functions as basis functions in regression or neural networks, the sigmoid function can be adjusted to better fit specific data patterns through two parameters: center $c$ and width $s$. This modified sigmoid function is often written as:

$$
\phi(x) = \sigma\left(\frac{x - c}{s}\right)
$$

Where:
- **$c$** (center) determines the point along the $x$-axis where the sigmoid function transitions from being close to 0 to being close to 1.
- **$s$** (scale or width) affects the steepness of the transition. A smaller $s$ makes the transition sharper, while a larger $s$ makes it smoother.

#### Example Plot of the Sigmoid Function
``` Python=9
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Range of x values from -10 to 10 for a clear view of the sigmoid behavior
x_values = np.linspace(-10, 10, 400)

# Different centers and widths for demonstration
centers = [0, 2, -2]  # Shifts the sigmoid along the x-axis
widths = [1, 0.5, 2]  # Changes the steepness of the curve

plt.figure(figsize=(10, 6))

for c, s in zip(centers, widths):
    y_values = sigmoid((x_values - c) / s)
    plt.plot(x_values, y_values, label=f'Center = {c}, Width = {s}')

plt.title('Effect of Center and Width on Sigmoid Function')
plt.xlabel('X')
plt.ylabel('Sigmoid(X)')
plt.legend()
plt.grid(True)
plt.show()

```
![image](https://hackmd.io/_uploads/ry8hcejZC.png)

### Choice of Width
The choice of width ($s$) for the sigmoid functions in a design matrix heavily depends on the desired sensitivity and overlap of the basis functions across the input range. The two formulas you're comparing are strategies for defining how quickly the sigmoid function transitions from its lower asymptote near 0 to its upper asymptote near 1, across the input domain defined from $x_{\text{min}}$ to $x_{\text{max}}$. Here's an explanation of each approach:

#### Formula 1: $(x_{\text{max}} - x_{\text{min}}) / (\text{nbf} - 1) / 2$
Note: nbf refers to nr_basis_functions

This formula effectively sets the width of each sigmoid to half the average distance between the centers of adjacent sigmoid functions. The rationale behind dividing by 2 is to ensure that each sigmoid function reaches about halfway to its midpoint at the location of the next center. This approach has several implications:

- **Overlap**: There is significant overlap between adjacent sigmoid functions, which ensures smoother transitions and a more gradual change in contributions from one basis function to the next across the range of input values.
- **Sensitivity**: The functions are less sensitive to small changes in input values because the transition region is broader. This can be beneficial when you want to avoid rapid changes in the output of the model due to small fluctuations in input.

#### Formula 2: $(x_{\text{max}} - x_{\text{min}}) / (\text{nbf} - 1)$

This version sets the width equal to the average distance between the centers of adjacent sigmoid functions. Here, each sigmoid will transition from near 0 to near 1 right around the point where the next sigmoid center is located, assuming minimal overlap at the steepest part of each sigmoid. Key characteristics include:

- **Reduced Overlap**: Each sigmoid function has less overlap with its neighbors, making each function more distinct and localized in its effect.
- **Increased Sensitivity**: The model becomes more sensitive to changes in input values because each sigmoid function covers a narrower range of input values. This can be advantageous when the model needs to react more distinctly to changes in specific regions of the input space.

#### Choosing the Right Width

The choice between these two (or other variations) depends on the specific requirements of the modeling task:

- **Generalization vs. Fitting**: A narrower width (closer to the second formula) might fit the training data more tightly, but it could lead to overfitting where the model captures noise instead of underlying patterns. A broader width (like the first formula) might generalize better on unseen data by smoothing out noise and minor fluctuations.
- **Data Characteristics**: If the input features exhibit clear, sharp transitions or thresholds that should trigger changes in output, a narrower width might be appropriate. Conversely, broader widths may be better for smoothly varying or more gradually changing relationships.

Ultimately, the width should be chosen based on experimental validation or domain-specific knowledge that informs how features are expected to influence the output. It's common to adjust these parameters during the model development process based on performance metrics on validation datasets.

### Example of Use

Imagine you have data on how likely individuals are to respond to a particular stimulus, and you believe that the probability of a response changes sharply at a specific value of some variable (like dosage of a drug). Using a sigmoid function as a basis function allows you to model this behavior, capturing the probability transitioning from very low to very high near a particular dosage level.

This flexibility to fit complex, non-linear patterns in data makes sigmoid functions invaluable in fields that require detailed probability estimates and classifications based on real-world, continuous input data.

When constructing a design matrix using sigmoidal basis functions for linear regression or other machine learning models, the choice of the width function for each sigmoidal basis function is crucial for capturing the variability and structure of the data effectively. The width, often denoted as $s$, determines how quickly the sigmoid function transitions from 0 to 1 across the input domain. Different choices for calculating this width can significantly influence the model's performance, especially in how it generalizes to unseen data.

### More Width types
A few approaches to define the width of sigmoidal basis functions, which can be tailored based on the specific characteristics of the dataset or the desired properties of the model:

#### i. **Fixed Width**
A simple and straightforward approach is to use a fixed width for all sigmoid functions across the input space. This method is easy to implement and interpret but may not be flexible enough to model data with varying scales or complexities effectively.

- **Formula**: 
  $$s = \text{constant}$$

#### ii. **Proportional to the Range of Data**
Adjusting the width in proportion to the overall range of the input data ensures that the sigmoid functions are scaled appropriately relative to the variability in the data.

- **Formula**:
  $$s = \frac{x_{\max} - x_{\min}}{k}$$
  where $k$ is a scaling factor that can be tuned based on the data distribution.

#### iii. **Proportional to the Distance Between Centers**
Another common approach is to set the width relative to the distance between the centers of adjacent sigmoid functions. This method helps to control the overlap between the functions, ensuring smooth transitions and sufficient coverage across the input domain.

- **Formula**:
  $$s = \frac{c_{i+1} - c_i}{k}$$
  where $c_i$ are the centers and $k$ is a factor controlling the degree of overlap (typically $k = 2$ to $k = 4$).

#### iv. **Based on Percentile Ranges**
In datasets with outliers or heavy tails, using percentile ranges (such as interquartile range) to determine the width can help in focusing the sigmoid functions on the most dense parts of the data distribution, thereby avoiding undue influence from extreme values.

- **Formula**:
  $$s = \frac{\text{percentile}(X, 75) - \text{percentile}(X, 25)}{k}$$
  This method uses the interquartile range scaled by a factor $k$ to adjust sensitivity.

#### v. **Adaptive Width**
Adaptive width strategies involve dynamically adjusting the width of each sigmoid function based on local data density or other criteria. This can be complex to implement but allows the model to adapt more finely to the underlying data structure.

- **Method**: Implementing this might involve techniques like kernel density estimation to assess data density and set widths accordingly.

#### Implementation Example

Here is a Python function to set up a sigmoidal design matrix with widths proportional to the distance between centers:

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def design_matrix_sigmoid(data, nr_basis_functions, x_min, x_max, k=2):
    centers = np.linspace(x_min, x_max, nr_basis_functions)
    width = np.min(np.diff(centers)) / k

    Phi_sig = np.zeros((data.shape[0], nr_basis_functions))
    for i in range(nr_basis_functions):
        Phi_sig[:, i] = sigmoid((data[:, 0] - centers[i]) / width)
    
    return Phi_sig
```
