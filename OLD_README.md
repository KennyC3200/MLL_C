## Gradient
For a function $f:\mathbb{R}^n\to\mathbb{R}$, the gradient of an input $\mathbf{x}=(x_1,\dots,x_n)\in\mathbb{R}^n$ is defined as:
$$
\nabla f=
\begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
$$
Where $\nabla f:\mathbb{R}^n\to\mathbb{R}^n$, and $\nabla f(\mathbf{x})$ corresponds to a vector of steepest ascent on the curve.

## Jacobian Matrix
Let $\mathbf{f}:\mathbb{R}^n\to\mathbb{R}^m$ be a function such that each of its first-order partial derivatives exists on $\mathbb{R}^n$. For an input vector $\mathbf{x}=(x_1,\dots,x_m)\in\mathbb{R}^n$, the output $\mathbf{f(x)}=(f_1(\mathbf{x}),\dots,f_m(\mathbf{x}))\in\mathbb{R}^m$. Then the Jacobian matrix of $\mathbf{f}$, denoted $\mathbf{J_f}$, is the $m\times n$ matrix whose $(i,j)$ entry is $\frac{\partial f_i}{\partial x_j}$.
$$
\begin{align*}
\mathbf{J_f}&=\left[\frac{\partial\mathbf{f}}{\partial x_1}\frac{\partial \mathbf{f}}{\partial x_2}\dots\frac{\partial\mathbf{f}}{\partial x_n}\right]\\
&=
\begin{bmatrix}
    \nabla^T f_1 \\
    \nabla^T f_2 \\
    \vdots \\
    \nabla^T f_m
\end{bmatrix}\\
&=
\begin{bmatrix}
    \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \dots & \frac{\partial f_1}{\partial x_n} \\
    \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \dots & \frac{\partial f_2}{\partial x_n} \\
    \vdots & \vdots & \ddots & \vdots \\
    \frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \dots & \frac{\partial f_m}{\partial x_n}
    \end{bmatrix}_{m\times n}
\end{align*}
$$
Where $\nabla^T f_i$ is the transpose of the gradient of the $i$-th component. This matrix basically stores the partial derivatives of $f_i$ for $i\in[1,m]$ with respect to $x_j$ for $j\in[1,n]$.

## JVP and VJP
The JVP (Jacobian-vector product) refers to the product of a Jacobian matrix $J$ and a vector $v$:
$$
J\cdot v
$$
And similarly, VJP (vector-Jacobian product) refers to:
$$
v\cdot J
$$

### The V in JVP
The Jacobian matrix tells us how changes in $x_j$ affect $f_i$. For example, changing $x_1$ by $1$ while keeping the other $x_j$ values constant, the change in the $f_i$ values is:
$$
J\cdot
\begin{bmatrix}
1 \\
0 \\
\vdots \\
0
\end{bmatrix}
=
\begin{bmatrix}
\frac{\partial f_1}{\partial x_1} \\
\frac{\partial f_2}{\partial x_1} \\
\vdots \\
\frac{\partial f_m}{\partial x_1}
\end{bmatrix}
$$
In general, if each $x_j$ changes by $\epsilon_j$, represented as $\mathbf{v}=[\epsilon_1,\dots,\epsilon_n]^T$, the corresponding change to $f_i$ is:
$$
J\cdot v=J\cdot 
\begin{bmatrix}
\epsilon_1 \\
\epsilon_2 \\
\vdots \\
\epsilon_n
\end{bmatrix}
$$
Which is the JVP. Each row (index $i$) tells us the effect that the changes in the $x_j$ values, described by $\mathbf{v}$, had on the corresponding function $f_i$.

### The V in VJP
Sometimes the output $f_i$ might be passed into another function, perhaps $g(f_1,\dots,f_m)$, which could be a loss function that returns a scalar (the loss). The derivative of $g$, $\frac{\partial g}{\partial f_i}$ tells us how much the loss changes with respect to $f_i$. To determine how changes in $x_j$ affect $g$, let
$$
v=
\begin{bmatrix}
\frac{\partial g}{\partial f_1} & \frac{\partial g}{\partial f_2} & \dots & \frac{\partial g}{\partial f_m}
\end{bmatrix}
$$
And computing VJP, we notice that through the chain rule we actually determine how changes in $x_j$ affects $g$:
$$
\begin{align*}
v\cdot J&=
\begin{bmatrix}
\frac{\partial g}{\partial f_1} & \frac{\partial g}{\partial f_2} & \dots & \frac{\partial g}{\partial f_m}
\end{bmatrix}
\begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \dots & \frac{\partial f_1}{x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \dots & \frac{\partial f_2}{x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \dots & \frac{\partial f_m}{x_n}
\end{bmatrix}\\
&=
\begin{bmatrix}
\displaystyle\sum_i\frac{\partial g}{\partial f_i}\frac{\partial f_i}{\partial x_1} & \displaystyle\sum_i\frac{\partial g}{\partial f_i}\frac{\partial f_i}{\partial x_2} & \dots & \displaystyle\sum_i\frac{\partial g}{\partial f_i}\frac{\partial f_i}{\partial x_n}
\end{bmatrix}\\
&=
\begin{bmatrix}
\frac{\partial g}{\partial x_1} & \frac{\partial g}{\partial x_2} & \dots & \frac{\partial g}{\partial x_n}
\end{bmatrix}
\end{align*}
$$

## The Jacobian is a Function
The Jacobian matrix of a function $\mathbb{f}\in\mathbb{R}^m$ consists of functions, not fixed values. The input of the Jacobian is a function $\mathbf{x}\in\mathbb{R}^n$
$$
J(\mathbf{x})=
\begin{bmatrix}
\frac{\partial f_1(\mathbf{x})}{\partial x_1} & \frac{\partial f_1(\mathbf{x})}{\partial x_2} & \dots & \frac{\partial f_1(\mathbf{x})}{\partial x_n} \\
\frac{\partial f_2(\mathbf{x})}{\partial x_1} & \frac{\partial f_2(\mathbf{x})}{\partial x_2} & \dots & \frac{\partial f_2(\mathbf{x})}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m(\mathbf{x})}{\partial x_1} & \frac{\partial f_m(\mathbf{x})}{\partial x_2} & \dots & \frac{\partial f_m(\mathbf{x})}{\partial x_n}
\end{bmatrix}
$$

### Example 1. Jacobian of $Wx$
Consider
$$
\mathbf{f(x)}=W\cdot x
$$
where
$$
\mathbf{x}=
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
\quad
\mathbf{W}=
\begin{bmatrix}
w_{1,1} & w_{1,2} \\
w_{2,1} & w_{2,2}
\end{bmatrix}
$$
Then
$$
\begin{align*}
\mathbf{f(x)}&=
\begin{bmatrix}
w_{1,1} & w_{1,2} \\
w_{2,1} & w_{2,2}
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}\\
&=
\begin{bmatrix}
w_{1,1}x_1 + w_{1,2}x_2 \\
w_{2,1}x_1 + w_{2,2}x_2
\end{bmatrix}\\
f_1(\mathbf{x})&=w_{1,1}x_1 + w_{1,2}x_2 \\
f_2(\mathbf{x})&=w_{2,1}x_1 + w_{2,2}x_2
\end{align*}
$$
Therefore
$$
\begin{align*}
\mathbf{J_f(x)}&=
\begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2}
\end{bmatrix}\\
&=
\begin{bmatrix}
w_{1,1} & w_{1,2} \\
w_{2,1} & w_{2,2}
\end{bmatrix}
\end{align*}
$$

### Example 2. Jacobian of $x^TWx$
Consider
$$
\begin{align*}
\mathbf{f(x)}&=x^TWx\\
&=
\begin{bmatrix}
x_1 & x_2
\end{bmatrix}
\begin{bmatrix}
w_{1,1} & w_{1,2} \\
w_{2,1} & w_{2,2}
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}\\
&=
\begin{bmatrix}
w_{1,1}x_1+w_{2,1}x_2+w_{1,2}x_1+w_{2,2}x_2
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}\\
&=
\begin{bmatrix}
w_{1,1}x_1^2+(w_{1,2}+w_{2,1})x_1x_2+w_{2,2}x_2^2
\end{bmatrix}
\end{align*}
$$
Since there is only one row, the Jacobian is simply
$$
\begin{align*}
\mathbf{J_f(x)}&=
\begin{bmatrix}
\frac{\partial f(\mathbf{x})}{\partial x_1} & \frac{\partial f(\mathbf{x})}{\partial x_2}
\end{bmatrix}\\
&=
\begin{bmatrix}
2w_{1,1}x_1^2+(w_{1,2}+w_{2,1})x_2 & 2w_{2,2}x_2+(w_{1,2}+w_{2,1})x_1
\end{bmatrix}
\end{align*}
$$

## Softmax Function
The softmax function is a vector function that grabs a vector of numbers and turns them into probabilities. Because the softmax function revolves around probabilities, the codomain is in the range $(0,1)$
$$
\sigma:\mathbb{R}^K\to(0,1)^K,\quad K>1
$$
The function takes in a tuple $\mathbf{z}=(z_1,\dots,z_k)\in\mathbb{R}^K$, computing each component of vector $\sigma(\mathbf{z})\in(0,1)^K$ with
$$
\sigma(\mathbf{z})_i=\frac{e^{z_i}}{\displaystyle\sum_{j=1}^k e^{z_j}}
$$

### Jacobian of Softmax
Since $\sigma:\mathbb{R}^n\to(0,1)^n$, the derivative of the softmax function is a Jacobian matrix:
$$
\mathbf{J_\sigma}=
\begin{bmatrix}
\frac{\partial \sigma_1}{\partial z_1} & \dots & \frac{\partial \sigma_1}{\partial z_n} \\  
\vdots & \ddots & \vdots \\
\frac{\partial \sigma_n}{\partial z_1} & \dots & \frac{\partial \sigma_n}{\partial z_n}
\end{bmatrix}_{n\times n}
$$
Since the denominator of the softmax function always contains $z_i$, the off-diagonal elements of the Jacobian are nonzero.

Since the softmax function outputs only positive values, we can apply a trick: instead of taking the partial derivative of the output, take the logarithmic derivative of the output instead:
$$
\begin{align*}
\frac{\partial}{\partial z_j}\ln(\sigma_i)&=\frac{1}{\sigma_i}\frac{\partial \sigma_i}{\partial z_j}\\
\Rightarrow\frac{\partial\sigma_i}{\partial z_j}&=\sigma_i\cdot\frac{\partial}{\partial z_j}\ln(\sigma _i)
\end{align*}
$$
To compute $\frac{\partial}{\partial z_j}\ln(\sigma_i)$, we first compute $\ln(\sigma_i)$:
$$
\begin{align*}
\ln(\sigma_i)&=\ln\left(\frac{e^{z_i}}{\sum_{k=1}^ne^{z_k}}\right)\\
&=z_i-\ln\left(\sum_{k=1}^ne^{z_k}\right)
\end{align*}
$$
Taking the partial derivative WRT $z_j$:
$$
\frac{\partial}{\partial z_j}\ln(\sigma_i)=\frac{\partial z_i}{\partial z_j}-\frac{\partial}{\partial z_j}\ln\left(\sum_{k=1}^ne^{z_k}\right)
$$
For the first term on the right-hand side, notice:
$$
\frac{\partial z_i}{\partial z_j}=
\begin{cases}
    1 & \text{if } i=j\\
    0  & \text{otherwise}
\end{cases}
$$
And we use the chain rule to evaluate the second term on the right side:
$$
\begin{align*}
\frac{\partial}{\partial z_j}\ln\left(\sum_{k=1}^n e^{z_k}\right)&=\frac{1}{\sum_{k=1}^ne^{z_k}}\frac{\partial}{\partial z_j}\left(\sum_{k=1}^ne^{z_k}\right)\\
&=\frac{1}{\sum_{k=1}^ne^{z_k}}(e^{z_j})\\
&=\sigma_j
\end{align*}
$$
So we get
$$
\frac{\partial}{\partial z_j}\ln(\sigma_i)=1\{i=j\}-\sigma_j\\
$$
Substituting this back into the original equation:
$$
\frac{\partial \sigma_i}{\partial z_j}=\sigma_i(1\{i=j\}-\sigma_j)
$$
This is the formula for all the elements in the Jacobi matrix. As an example, for $n=4$:
$$
\mathbf{J_\sigma}=
\begin{bmatrix}
\sigma_1 \cdot (1 - \sigma_1) & -\sigma_1 \cdot \sigma_2 & -\sigma_1 \cdot \sigma_3 & -\sigma_1 \cdot \sigma_4 \\
-\sigma_2 \cdot \sigma_1 & \sigma_2 \cdot (1 - \sigma_2) & -\sigma_2 \cdot \sigma_3 & -\sigma_2 \cdot \sigma_4 \\
-\sigma_3 \cdot \sigma_1 & -\sigma_3 \cdot \sigma_2 & \sigma_3 \cdot (1 - \sigma_3) & -\sigma_3 \cdot \sigma_4 \\
-\sigma_4 \cdot \sigma_1 & -\sigma_4 \cdot \sigma_2 & -\sigma_4 \cdot \sigma_3 & \sigma_4 \cdot (1 - \sigma_4)
\end{bmatrix}
$$
Notice how the diagonal elements differ from the non-diagonal elements. An interesting property is that $\mathbf{J}_\sigma=\mathbf{J}_\sigma^T$.

## Categorical Cross-Entropy Loss
Softmax function is related to categorical cross-entropy loss since it is usually only used with networks with a softmax layer at the output. Classification problems can be broken into two separate categories:
- Multi-class classification: each sample belongs to only one class (mutually exclusive)
- Multi-label classification: each sample belongs to multiple classes (or to no class)

Categorical cross-entropy loss is used in multi-class classification tasks, where one sample belongs exactly to one of the $C$ classes. The true label assigned to each sample consists of an integer value between 0 and $C-1$. The label is represented by one-hot encoded vector of size $C$, which has a $1$ for the correct class and $0$ everywhere else. For instance, if $\text{label}=2$, the one-hot encoded vector would be:
$$
\mathbf{y}=
\begin{bmatrix}
0 \\ 0 \\ 1 \\ 0
\end{bmatrix}
$$

Cross-entropy takes in two probability distributions and outputs a single scalar that represents the similarity between the two distributions.
$$
\mathcal{L}(\mathbf{p},\mathbf{q})=-\sum_{i=1}^Cp_i\ln(q_i)
$$
Where $C$ is the number of different classes, $\mathbf{p}$ contains the expected probability distributions, and $\mathbf{q}$ contains the actual probability distribution. Intuitively, the smaller $\mathcal{L}$ is, the more similar the two probability distributions are. 

When cross-entropy is the loss function, $\mathbf{p}$ is fed with the one-hot encoded labels and the probabilities generated by the softmax function $\mathbf{\sigma}$ are used as $\mathbf{q}$.

To begin the backpropagation process, we need to calculate the derivative of the loss $\mathcal{L}$ WRT the weighted input $\mathbf{z}$ of the output layer
$$
\begin{align*}
\frac{\partial\mathcal{L}}{\partial z_j}&=-\frac{\partial}{\partial z_j}\sum_{i=1}^Cp_i\ln(q_i)\\
&=-\sum_{i=1}^Cp_i\frac{\partial\ln(q_i)}{\partial z_j}\\
&=-\sum_{i=1}^C\frac{p_i}{q_i}\frac{\partial q_i}{\partial z_j}
\end{align*}
$$
Since the actual probability distribution is the softmax function $\sigma$:
$$
\frac{\partial\mathcal{L}}{\partial z_j}=-\sum_{i=1}^C\frac{p_i}{\sigma_i}\frac{\partial \sigma_i}{\partial z_j}
$$
Since we uncovered $\frac{\partial \sigma_i}{\partial z_j}$ previously:
$$
\begin{align*}
\frac{\partial\mathcal{L}}{\partial z_j}&=-\sum_{i=1}^C\frac{p_i}{\sigma_i}\sigma_i(1\{i=j\}-\sigma_j)\\
&=-\sum_{i=1}^Cp_i(1\{i=j\}-\sigma_j)\\
&=\sum_{i=1}^Cp_i\sigma_j-\sum_{i=1}^Cp_i\{i=j\}\\
&=\sum_{i=1}^Cp_i\sigma_j-p_j\\
&=\sigma_j\sum_{i=1}^Cp_i-p_j
\end{align*}
$$
Here, notice that $\displaystyle\sum_{i=1}^Cp_i=1$ because $\mathbf{p}$ is a one-hot encoded vector, meaning the sum of all its elements equal 1. Hence:
$$
\begin{align*}
\frac{\partial\mathcal{L}}{\partial z_j}&=\sigma_j-p_j\\
\Rightarrow\frac{\partial\mathcal{L}}{\partial\mathbf{z}}&=\mathbf{\sigma}-\mathbf{p}
\end{align*}
$$

## Xavier Initialization
This is a technique used to initialize weights in a neural network by setting the initial weights of a model in a way that enables the activations and gradients to flow efficiently during both the forward pass and backpropagation. In this technique, the key factor is the number of inputs and outputs in the layers.

This method solves some notable problems that occur:
1. Vanishing and exploding gradient
2. Overfitting
3. Saturation

### Uniform Xavier Initialization
We can initialize weight $w$ by drawing them from a random uniform distribution within a specific range, $w\in[-x,x]$ where:
$$
x=\sqrt{\frac{6}{n_\text{inputs}+n_\text{outputs}}}
$$

### Normal Xavier Initialization
The initialization sets the initial weights by drawing them from a Gaussian distribution with a mean of 0 and a specific standard deviation $\sigma$, where
$$
\sigma=\sqrt{\frac{2}{n_\text{inputs}+n_\text{outputs}}}
$$

## Forward Pass
For a batch that has $k$ samples, let layer $l$ have $n_{l}$ neurons and layer $l+1$ have $n_{l+1}$ neurons. For layer $l$, we can express the activated values for each neuron across all $k$ samples simultaneously as a $k\times n_l$ matrix:
$$
\begin{align*}
A^{(l)}&=
\begin{bmatrix}
    \mathbf{a^{(l)}_1} \\
    \mathbf{a^{(l)}_2} \\
    \vdots \\
    \mathbf{a^{(l)}_k} \\
\end{bmatrix}
\\
&=
\begin{bmatrix}
a^{(l)}_{1,1} & a^{(l)}_{1,2} & \dots & a^{(l)}_{1,n_l} \\
a^{(l)}_{2,1} & a^{(l)}_{2,2} & \dots & a^{(l)}_{2,n_l} \\
\vdots & \vdots & \ddots & \vdots \\
a^{(l)}_{k,1} & a^{(l)}_{k,2} & \dots & a^{(l)}_{k,n_l} \\
\end{bmatrix}_{k\times n_l}
\end{align*}
$$
Where there are $k$ rows, and each row stores $n_l$ values, corresponding to the number of neurons in the layer.

As for calculating the activated values for the neurons in the next layer, we recall that each neuron's weighted value can be calculated as
$$
z^{(l+1)}_i=\Sigma_j w^{(l+1)}_{(i,j)}a^{(l)}_{(i,j)}+b^{(l+1)}_i
$$
Before we use this value to calculate the activated value:
$$
a^{(l+1)}_i=f(z^{(l+1)}_i)
$$
When building out the computation graph, we start with calculating the multiplications (weight multiplied by previous activation value), then adding (weight sum and adding bias), and then the activation function.

Starting with calculating the multiplications, we can first express the weights for all the neurons as a matrix:
$$
\begin{align*}
W^{(l+1)}&=
\begin{bmatrix}
\mathbf{w^{(l+1)}_{1}} & \mathbf{w^{(l+1)}_{2}} & \dots & \mathbf{w^{(l+1)}_{n_{l+1}}} \\
\end{bmatrix}\\
&=
\begin{bmatrix}
    w^{(l+1)}_{1,1} & w^{(l+1)}_{1,2} & \dots & w^{(l+1)}_{1,n_{l+1}} \\
    w^{(l+1)}_{2,1} & w^{(l+1)}_{2,2} & \dots & w^{(l+1)}_{2,n_{l+1}} \\ 
    \vdots & \vdots & \ddots & \vdots \\
    w^{(l+1)}_{n_l,1} & w^{(l+1)}_{n_l,2} & \dots & w^{(l+1)}_{n_l,n_{l+1}}
\end{bmatrix}
\end{align*}
$$
Where there are $n_{l+1}$ columns for the $n_{l+1}$ neurons in the $A^{(l+1)}$ layer and there are $n_l$ rows for the $n_l$ neurons in the $A^{(l)}$ layer. 

For the biases, since each neuron only gets one bias, the matrix is a lot simpler:
$$
\begin{align*}
B^{(l+1)}&=
\begin{bmatrix}
    \mathbf{b^{(l+1)}_1} & \mathbf{b^{(l+1)}_2} & \dots & \mathbf{b^{(l+1)}_{n_{l+1}}}
\end{bmatrix}\\
&=
\begin{bmatrix}
    b^{(l+1)}_1 & b^{(l+1)}_2 & \dots & b^{(l+1)}_{n_{l+1}} \\
    b^{(l+1)}_1 & b^{(l+1)}_2 & \dots & b^{(l+1)}_{n_{l+1}} \\
    \vdots & \vdots & \ddots & \vdots \\
    b^{(l+1)}_1 & b^{(l+1)}_2 & \dots & b^{(l+1)}_{n_{l+1}}
\end{bmatrix}_{k\times n_{l+1}}
\end{align*}
$$
Where there are $k$ rows for the number of sample tests and for each row there are $n_{l+1}$ biases because there are $n_{l+1}$ neurons to add the individual biases to.

So the weighted sum can be written as:
$$
\begin{align*}
Z^{(l+1)}&=A^{(l)}W^{(l+1)} + B^{(l+1)} \\
&=\begin{bmatrix}
    a^{(l)}_{1,1} & a^{(l)}_{1,2} & \dots & a^{(l)}_{1,n_{l}} \\
    a^{(l)}_{2,1} & a^{(l)}_{2,2} & \dots & a^{(l)}_{2,n_{l}} \\
    \vdots & \vdots & \ddots & \vdots \\
    a^{(l)}_{k,1} & a^{(l)}_{k,2} & \dots & a^{(l)}_{k,n_{l}}
\end{bmatrix}
\begin{bmatrix}
    w^{(l+1)}_{1,1} & w^{(l+1)}_{1,2} & \dots & w^{(l+1)}_{1,n_{l+1}} \\
    w^{(l+1)}_{2,1} & w^{(l+1)}_{2,2} & \dots & w^{(l+1)}_{2,n_{l+1}} \\ 
    \vdots & \vdots & \ddots & \vdots \\
    w^{(l+1)}_{n_l,1} & w^{(l+1)}_{n_l,2} & \dots & w^{(l+1)}_{n_l,n_{l+1}}
\end{bmatrix}
+
\begin{bmatrix}
b^{(l+1)}_1 & b^{(l+1)}_2 & \dots & b^{(l+1)}_{n_{l+1}} \\
b^{(l+1)}_1 & b^{(l+1)}_2 & \dots & b^{(l+1)}_{n_{l+1}} \\
\vdots & \vdots & \ddots & \vdots \\
b^{(l+1)}_1 & b^{(l+1)}_2 & \dots & b^{(l+1)}_{n_{l+1}} \\
\end{bmatrix}_{k\times n_{l+1}}\\
&=
\begin{bmatrix}
z^{(l+1)}_{1,1} & z^{(l+1)}_{1,2} & \dots & z^{(l+1)}_{1,n_{l+1}} \\
z^{(l+1)}_{2,1} & z^{(l+1)}_{2,2} & \dots & z^{(l+1)}_{2,n_{l+1}} \\
\vdots & \vdots & \ddots & \vdots \\
z^{(l+1)}_{k,1} & z^{(l+1)}_{k,2} & \dots & z^{(l+1)}_{k,n_{l+1}} \\
\end{bmatrix}_{k\times n_{l+1}}
\end{align*}
$$
And again, there are $k$ rows for the $k$ samples and each row contains $n_{l+1}$ neurons for that layer. Then, we can apply the activation functo for each element to get
$$
A^{(l+1)}=
\begin{bmatrix}
a^{(l+1)}_{1,1} & a^{(l+1)}_{1,2} & \dots & a^{(l+1)}_{1,n_{l+1}} \\
a^{(l+1)}_{2,1} & a^{(l+1)}_{2,2} & \dots & a^{(l+1)}_{2,n_{l+1}} \\
\vdots & \vdots & \ddots & \vdots \\
a^{(l+1)}_{k,1} & a^{(l+1)}_{k,2} & \dots & a^{(l+1)}_{k,n_{l+1}} \\
\end{bmatrix}_{k\times n_{l+1}}
$$ 
And this process repeats until it reaches the output layer.

## Backpropagation
This involves going backwards in the computation graph and using the chain rule to calculate the gradients for the weights and biases. For activation function $f$, weighted sum $z$, weight $w$, and bias $b$
$$
\begin{align*}
a&=f(z)\\
df&=\frac{\partial L}{\partial f}\\
dz&=\frac{\partial L}{\partial f}\frac{\partial f}{\partial z}=df\frac{\partial f}{\partial z}\\
dw&=\frac{\partial L}{\partial z}\frac{\partial z}{\partial w}=dz\frac{\partial z}{\partial w}\\
db&=\frac{\partial L}{\partial z}\frac{\partial z}{\partial b}=dw\frac{\partial z}{\partial b}
\end{align*}
$$
Notice that due to the ordering of the computation graph, we first utilize the $df$ to calculate $dz$ which is then used to calculate gradients $dw$ and $db$.

### Weighted Sum Gradient ($\frac{\partial L}{\partial Z}$)
Let the activation $a_{i,j}=f(z_{i,j})$ and assuming that subsequent layers has provided $\frac{\partial L}{\partial A}$, we first compute $\frac{\partial L}{\partial Z}$
$$
\frac{\partial L}{\partial z_{i,j}}=\frac{\partial L}{\partial a_{i,j}}\cdot f'(z_{i,j})
$$
Extending this to matrices, this is an element-wise product:
$$
\frac{\partial L}{\partial Z}=\frac{\partial L}{\partial A}\odot f'(Z)
$$

### Weight Gradient ($\frac{\partial L}{\partial W}$)
From the computation graph, once we calculate $\frac{\partial L}{\partial Z}$ we can calculate $\frac{\partial L}{\partial W}$. A single weighted sum $z_{i,k}$ (sample $i$, output feature $j$) for layer $l$ is given by
$$
z^{(l)}_{i,j}=\displaystyle\sum_{m=1}^{n_{l-1}}a^{(l-1)}_{i,m}w^{(l)}_{m,j}+b^{(l)}_j
$$
Taking the partial derivative of $L$ with respect to weight $w_{m,j}$:
$$
\frac{\partial L}{\partial w^{(l)}_{m,j}}=\sum_{i=1}^k\frac{\partial L}{\partial z^{(l)}_{i,j}}\frac{\partial z^{(l)}_{i,j}}{\partial w^{(l)}_{m,j}}
$$
**Note:** A single weight $w_{m,j} connects input feature $m$ to output feature $j$. We do the sum from $i=1$ to $i=k$ because in a batch of $m$ samples, $w_{m,j}$ is reused $k$ times--once for every single sample $i$ in the batch. The multivariable chain rule says we must sum the contributions of $w_{m,j}$ across all $k$ samples.

And since $\frac{\partial z_{i,j}}{\partial w^{(l)}_{m,j}}=a^{(l-1)}_{i,m}$:
$$
\begin{align*}
\frac{\partial L}{\partial w^{(l)}_{m,j}}&=\sum_{i=1}^k\frac{\partial L}{\partial z^{(l)}_{i,j}}a^{(l-1)}_{i,m}\\
&=\sum_{i=1}^ka^{(l-1)}_{i,m}\frac{\partial L}{\partial z^{(l)}_{i,j}}
\end{align*}
$$
To make things clear for what the indices represent:
- $i$ is the sample index (there are $k$ total samples)
- $m$ is the weight index, or also the neuron index in layer $l-1$
- $j$ is the neuron index in layer $l$

Upon rearrangement, it becomes apparent that for $a^{(l-1)}_{i,m}$, if you swap the indices we can replace this summation with matrix multiplication:
$$
\begin{align*}
\frac{\partial L}{\partial w^{(l)}_{m,j}}&=
\begin{bmatrix}
    a^{(l-1)}_{m,1} & a^{(l-1)}_{m,2} & \dots & a^{(l-1)}_{m,k}
\end{bmatrix}
\begin{bmatrix}
    \frac{\partial L}{\partial z^{(l)}_{1,j}} \\
    \frac{\partial L}{\partial z^{(l)}_{2,j}} \\
    \vdots \\
    \frac{\partial L}{\partial z^{(l)}_{k,j}}
\end{bmatrix}\\
&=
\begin{bmatrix}
    a^{(l-1)}_{1,m} \\
     a^{(l-1)}_{2,m} \\
     \vdots \\
      a^{(l-1)}_{k,m}
\end{bmatrix}^T
\begin{bmatrix}
    \frac{\partial L}{\partial z^{(l)}_{1,j}} \\
    \frac{\partial L}{\partial z^{(l)}_{2,j}} \\
    \vdots \\
    \frac{\partial L}{\partial z^{(l)}_{k,j}}
\end{bmatrix}
\end{align*}
$$

Hence, for $m\in[1,n_{l-1}]$, we can express this as one matrix multiplication:
$$
\frac{\partial L}{\partial W}=(A^{(l-1)})^T\frac{\partial L}{\partial Z}
$$

### Input Gradient ($\frac{\partial L}{\partial A}$)
To pass the gradient to the previous layer, we need $\frac{\partial L}{\partial A}$, which is what we will be deriving. By the chain rule:
$$
\frac{\partial L}{\partial a^{(l-1)}_{i,m}}=\sum_{j=1}^{n_l}\frac{\partial L}{\partial z^{(l)}_{i,j}}\frac{\partial z^{(l)}_{i,j}}{\partial a^{(l-1)}_{i,m}}
$$
**Note:** An input element $a_{i,m}$ belongs to one sample $i$ and one input feature $m$. During the matrix multiplication, the specific $a_{i,m}$ is multiplied by every weight in row $m$ of the weight matrix to produce output features $j=1,\dots,n_l$ for that specific sample $i$. According to the multivariable chain rule, we must sum over all $n_l$ channels.

And since $\frac{\partial z^{(l)}_{i,j}}{\partial a^{(l)}_{i,m}}=w^{(l)}_{m,j}$:
$$
\frac{\partial L}{\partial a^{(l)}_{i,m}}=\sum_{j=1}^{n_l}\frac{\partial L}{\partial z^{(l)}_{i,j}}w^{(l)}_{m,j}
$$
And once again notice that if we reverse the indices for $w^{(l)}_{m,j}$ we can express the sum as a matrix product. Therefore, we can express this as:
$$
\frac{\partial L}{\partial A^{(l-1)}}=\frac{\partial L}{\partial Z}W^T
$$

### Bias Gradient ($\frac{\partial L}{\partial b}$)
For bias $b_j$:
$$
\frac{\partial L}{\partial b^{(l)}_j}=\sum_{i=1}^{k}\frac{\partial L}{\partial z^{(l)}_{i,j}}\frac{\partial z^{(l)}_{{i,j}}}{\partial b^{(l)}_{j}}
$$
But $\frac{\partial z^{(l)}_{{i,j}}}{\partial b^{(l)}_{j}}=1$:
$$
\frac{\partial L}{\partial b^{(l)}_j}=\sum_{i=1}^{k}\frac{\partial L}{\partial z^{(l)}_{i,j}}
$$
This sums the gradient across all $k$ samples along columns:
$$
\frac{\partial L}{\partial b^{(l)}}=\sum_{i=1}^{k}\left(\frac{\partial L}{\partial Z^{(l)}}\right)_{i,:}
$$
**Note:** The reason we take the sum for $k$ rows is because each neuron has only one bias and each bias has a gradient per sample. Since there are $k$ samples, we need to take the sum of the $k$ gradients of the biases.

## Articles Used
- https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant
- https://wangkuiyi.github.io/jacobian.html
- https://www.math.uwaterloo.ca/~hwolkowi/henry/reports/kronthesisschaecke04.pdf
- https://medium.com/data-science/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
- https://en.wikipedia.org/wiki/Softmax_function
- https://365datascience.com/tutorials/machine-learning-tutorials/what-is-xavier-initialization/
- https://www.geeksforgeeks.org/deep-learning/xavier-initialization/