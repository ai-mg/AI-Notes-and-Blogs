

# 1. Fully Recurrent Networks: Overview

In this Section, we will organize and summarize key concepts about fully recurrent networks, the forward pass, derivatives, and visualization of computational graphs. We'll also provide mathematical equations, explanations, and illustrations for better understanding.

A **fully recurrent network (RNN)** is a type of neural network designed for processing sequences of data, such as time-series data or text. Unlike traditional feed-forward networks, RNNs have **connections that loop back on themselves**, allowing the network to **remember information** from previous time steps. This property makes RNNs well-suited for sequential tasks like language modeling and speech recognition.

The basic equations for a fully recurrent network are:

$s(t) = W x(t) + R a(t-1)$

$a(t) = \tanh(s(t))$

$z(t) = V a(t)$

$\hat{y}(t) = \sigma(z(t))$

- **$s(t)$**: Intermediate value combining the current input $x(t)$ and the previous activation $a(t-1)$.
- **$a(t)$**: Hidden state at time step $t$, computed using the **tanh** activation function.
- **$z(t)$**: Logit value representing the output of the hidden state at time step $t$.
- **$\hat{y}(t)$**: Predicted output at time step $t$, computed using the **sigmoid** activation function.

### Key Components

- **Weight Matrices**:
  - **$W$**: Weights for the input to the hidden state.
  - **$R$**: Weights for the recurrent (hidden to hidden) connection.
  - **$V$**: Weights for the hidden state to the output.
- **Activation Functions**:
  - **$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$**: Used to keep the activations stable between -1 and 1.
  - **$\sigma(x) = \frac{1}{1 + e^{-x}}$**: Sigmoid function used to convert logits into probabilities between 0 and 1.

## 1.1 Forward Pass Explained

The **forward pass** is the process of passing an input sequence through the network to get an output. In an RNN, this involves processing each time step sequentially, computing the intermediate values, hidden activations, and predictions.

### Step-by-Step Explanation

The forward pass for the RNN involves the following steps:

1. **Initialization**: Start with zero values for the hidden state at the initial time step.

2. **Loop Through Time Steps**: For each time step $t$:
   - **Compute $s(t)$**:
     
     $$
     s(t) = W x(t) + R a(t-1)
     $$
     This combines the **current input** and the **memory** from the previous time step.
   - **Compute $a(t)$**:
     
     $$
     a(t) = \tanh(s(t))
     $$
     This calculates the **hidden state** for the current time step, adding non-linearity to the model.

3. **Output Calculation** (Final Time Step):
   - **Compute $z(T)$**:
     
     $$
     z(T) = V a(T)
     $$
     This calculates the **logit value** at the final time step.
   - **Compute Predicted Output $\hat{y}(T)$**:
     
     $$
     \hat{y}(T) = \sigma(z(T))
     $$
     The sigmoid function converts the logit into a **probability**.

4. **Loss Calculation**: Use **binary cross-entropy** to calculate the loss:

   $$
   L(z, y) = -y \log(\hat{y}) - (1 - y) \log(1 - \hat{y})
   $$

### Code Representation

```python
# Example of the forward pass in Python
def forward(model, x, y):
    T, D = x.shape
    I = model.W.shape[0]
    K = model.V.shape[0]

    # Initialize hidden activations and logit
    model.a = np.zeros((T, I))
    model.z = np.zeros(K)

    for t in range(T):
        if t == 0:
            a_prev = np.zeros(I)
        else:
            a_prev = model.a[t - 1]

        # Compute s(t)
        s_t = np.dot(model.W, x[t]) + np.dot(model.R, a_prev)
        # Compute a(t)
        model.a[t] = np.tanh(s_t)

    # Compute z(T) and predicted output
    model.z = np.dot(model.V, model.a[-1])
    y_hat = sigmoid(model.z)

    # Compute binary cross-entropy loss
    loss = -y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)

    return loss
```

## 1.2 Numerical Stability of Binary Cross-Entropy Loss

### Problem of Numerical Instability

The **binary cross-entropy loss** function can suffer from **numerical instability** due to the use of logarithms and exponentials. There are two main problems to consider:

1. **Overflow**: When $z$ becomes very large (positive), $e^z$ can grow extremely quickly, leading to **overflow** issues where the value exceeds the range that can be represented numerically.
When $z$ is large positive
   - $e^z \rightarrow \infty$
   - Example: $e^{100} \approx 2.688117141 × 10^{43}$
   
3. **Underflow**: When $z$ becomes very negative, $e^{-z}$ becomes very small, leading to **underflow** issues where the value becomes too tiny for the computer to represent, effectively resulting in zero.
When $z$ is large negative
   - $e^{-z} \rightarrow 0$
   - Example: $e^{-100} \approx 3.720075976 × 10^{-44}$
5. **Logarithm of Near-Zero Values**: The logarithm function $log(x)$ approaches negative infinity as $x$ approaches zero, which means that if the predicted probability ($\hat{y}$) is very close to 0 or 1, the cross-entropy can become extremely large.
When $\hat{y}$ approaches 0 or 1
   - $\log(\hat{y}) \rightarrow -\infty$
   - Example: $\log(10^{-308}) \approx -709.78$

### Solution: Numerical Stability through Log-Sum-Exp Trick

To address these problems, it's better to work with the **logits** directly rather than the probabilities. Instead of calculating the sigmoid and then applying the cross-entropy loss, you combine the sigmoid and loss calculations into a single expression to maintain stability:

Using the **log-sum-exp trick**:

$$
\log(1 + e^z) = \log(1 + e^{-|z|}) + \max(0, z)
$$

This formulation helps ensure that the exponential terms do not grow too large or too small, which helps avoid overflow and underflow.

### Derivative of the Binary Cross-Entropy Loss

The derivative of the binary cross-entropy loss function $L(z, y)$ with respect to the logit $z$ is:

$$
\frac{dL(z, y)}{dz} = \sigma(z) - y
$$

This result is **unexpectedly simple** because it directly relates the **predicted probability** $\sigma(z)$ to the **true label** $y$. The simplicity makes the optimization process efficient and easy to interpret.

## 1.3 Speeding Up the Forward Pass

### Loop-Free Computation

The forward pass as described earlier uses a loop to iterate over each time step in the sequence. This can be inefficient, especially for long sequences. To **speed up** the forward pass, you can reconfigure the computation to avoid the loop:

- **Matrix Multiplication**: Instead of iterating through each time step, you can represent the entire sequence as a matrix and use matrix multiplication to calculate the intermediate states for all time steps at once.
We can reshape the input and perform batch matrix operations:

```python
def forward_optimized(model, x, y):
    # Reshape x to (batch_size, sequence_length, input_dim)
    X = x.reshape(-1, T, D)
    
    # Compute all hidden states at once
    S = np.dot(X, model.W.T) + np.dot(model.a_prev, model.R.T)
    A = np.tanh(S)
    
    # Compute final output
    z = np.dot(A[:, -1, :], model.V.T)
    return compute_stable_bce_loss(z, y)
```
- **Removing Non-linearities**:
In cases where linear relationships are sufficient, we can remove the tanh activation:

```python
def forward_linear(model, x, y):
    X = x.reshape(-1, T, D)
    A = np.dot(X, model.W.T) + np.dot(model.a_prev, model.R.T)
    z = np.dot(A[:, -1, :], model.V.T)
    return compute_stable_bce_loss(z, y)
```
- **Batch Processing**: You can process multiple sequences at the same time, making use of efficient tensor operations with frameworks like **NumPy**, **TensorFlow**, or **PyTorch**.

## 1.4 Computational Graph of the RNN

The **computational graph** provides a visual representation of how the inputs, activations, and outputs are connected across time steps in an RNN.

### Creating and Visualizing the Graph

The following code creates a computational graph using **NetworkX** and **Matplotlib** to represent the operations happening at each time step.

```python
import networkx as nx
import matplotlib.pyplot as plt

def create_rnn_graph():
    G = nx.DiGraph()
    
    # Add nodes for time steps t = 1, 2, 3
    for t in range(1, 4):
        G.add_node(f'x({t})', pos=(t, 4))
        G.add_node(f'a({t})', pos=(t, 3))
        G.add_node(f'z({t})', pos=(t, 2))
        G.add_node(f'L({t})', pos=(t, 1))
    
    # Add edges showing dependencies
    for t in range(1, 4):
        G.add_edge(f'x({t})', f'a({t})', label='W')
        G.add_edge(f'a({t})', f'z({t})', label='V')
        G.add_edge(f'z({t})', f'L({t})', label='Loss')
        if t > 1:
            G.add_edge(f'a({t-1})', f'a({t})', label='R')
    
    return G

def draw_rnn_graph(G):
    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=3000, arrowsize=20, font_size=10, 
            font_weight='bold')
    
    # Add edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title("Computational Graph of Fully Recurrent Network (t = 1, 2, 3)", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Create and draw the graph
G = create_rnn_graph()
draw_rnn_graph(G)
```

### Graph Description
- **Nodes** represent key components at each time step: input ($x(t)$), hidden activation ($a(t)$), logit ($z(t)$), and loss ($L(t)$).
- **Edges** represent the flow of computation, such as how the input influences the hidden state and how the previous hidden state affects the next one.
- The **recurrent connections** show how information flows from one time step to the next, allowing the RNN to retain memory.

### Computational Graph and PyTorch

In frameworks like **PyTorch**, the computational graph is automatically constructed during the **forward pass**. Each operation on tensors creates nodes and edges in the computational graph, which PyTorch uses to calculate **gradients** during the **backward pass**.

- **Backward Pass**: The backward pass involves calculating the **gradient** of the loss with respect to all parameters in the network using **backpropagation**. PyTorch records all the operations during the forward pass and uses them to compute gradients efficiently.
- **Importance of Computational Graph**: Understanding the computational graph is crucial for estimating gradients manually and for debugging. It helps visualize how different operations depend on each other and ensures that the network is learning correctly.

#### PyTorch's Automatic Differentiation

PyTorch builds a dynamic computational graph during the forward pass:

```python
class RNN(nn.Module):
    def forward(self, x):
        # PyTorch records operations
        s_t = torch.matmul(x, self.W.t()) + torch.matmul(self.a_prev, self.R.t())
        a_t = torch.tanh(s_t)  # Operation recorded
        z = torch.matmul(a_t, self.V.t())  # Operation recorded
        return z
```

During backward pass:
1. Computes $\frac{\partial L}{\partial z}$
2. Uses chain rule to compute gradients for all parameters
3. Updates weights using computed gradients

### Manual Gradient Computation

Understanding the computational graph helps in manual gradient computation:

```python
def manual_backward(model, loss_grad):
    # Gradient of loss w.r.t. z
    dz = sigmoid(model.z) - y  # Simple due to BCE loss
    
    # Gradient of loss w.r.t. V
    dV = np.outer(dz, model.a[-1])
    
    # Gradient of loss w.r.t. hidden states
    da = np.zeros_like(model.a)
    da[-1] = np.dot(model.V.T, dz)
    
    for t in reversed(range(T-1)):
        da[t] = np.dot(model.R.T, da[t+1]) * (1 - model.a[t]**2)
    
    return dV, da
```

## 1.5 Advanced Optimization Techniques

### Parallelization Strategies

1. **Batch Processing**
   - Process multiple sequences simultaneously
   - Utilize GPU acceleration
   - Implement mini-batch gradient descent

2. **Sequence Chunking**
   - Split long sequences into manageable chunks
   - Process chunks in parallel
   - Maintain state between chunks

### Alternative Loss Functions

#### Least Squares Error
- More stable for regression tasks
- Less sensitive to outliers
- Linear gradient behavior

```python
def least_squares_loss(y_pred, y_true):
    return 0.5 * np.sum((y_pred - y_true)**2)
```

### Linearization Techniques

1. **ReLU Instead of Tanh**
   - Faster computation
   - No saturation
   - Sparse activation

2. **Linear Approximation**
   - Replace non-linear functions with piecewise linear approximations
   - Faster forward and backward passes
   - Trade-off between accuracy and speed


## 1.6 Summary

- **Fully Recurrent Networks** are designed to process sequential data by having loops that carry information from one time step to the next.
- The **forward pass** involves computing hidden activations and predictions sequentially, ultimately calculating a loss that measures how far off the prediction is from the true label.
- **Numerical stability** is crucial in calculating the binary cross-entropy loss, and combining the sigmoid and cross-entropy into one formulation helps avoid overflow and underflow.
- The **derivative** of the binary cross-entropy loss with respect to the logit is simply $\sigma(z) - y$, making gradient-based optimization efficient.
- The **computational graph** visually represents the relationships and dependencies between inputs, activations, and outputs across multiple time steps, which is important for understanding backpropagation and gradient computation.

These concepts are foundational for understanding how recurrent neural networks learn to process sequences, remember important information, and make predictions based on past inputs. The computational graph is particularly important for implementing backpropagation, whether done manually or by using frameworks like PyTorch.


## 1.7 References

1. Goodfellow, I., et al. (2016). Deep Learning. MIT Press.
2. Pascanu, R., et al. (2013). On the difficulty of training recurrent neural networks.
3. PyTorch Documentation: https://pytorch.org/docs/stable/autograd.html

