# **Neural Networks from Scratch** 🕸️
### These notebooks explore Neural Networks, from the very basic Perceptron and its ability to classify linearly, to a DNN. All "from scratch", based on awesome tutorials from Guillaume Saint-Cirgue (Machine Learnia youtube channel) and more (see links below).<br></br>

## <u>Notebook names</u> :
### 1. Basic Perceptron
### 2. Neural Network with 1 hidden layer
### 3. Neural Network with 1 hidden layer and a multiclass output - softmax activation function
### 4. Neural Network with N layers
### 5. Neural Network with N layers and a multiclass output - softmax activation function
### 6. Neural Network with N layers as Class Objects, binary and multinomial classification<br></br>

## <u>Python script :</u>
### `DNN_from_scratch.py`
<br></br>
## ***Sources*** :
<h3>



- for the mathematical notations and equations :
  - [Machine Learnia's Youtube channel, deeplearning playlist](https://www.youtube.com/watch?v=XUFLq6dKQok&list=PLO_fdPEVlfKoanjvTJbIbd9V5d9Pzp8Rw)

- for the multiclass classification :
  I based the `DL_2layers_multiclass` notebook on previous notebooks (`DL_basics_1neuron` & `DL_2layers`) and different sources from internet :
    - [an example of multiclass DNN python code](https://stackabuse.com/creating-a-neural-network-from-scratch-in-python-multi-class-classification/)
    - [Detailed equations & python code](http://kkms.org/index.php/kjm/article/view/1275/673)
    - [Cross-Entropy calculation](https://stats.stackexchange.com/questions/378274/how-to-construct-a-cross-entropy-loss-for-general-regression-targets)
    - [Courses notes on deeplearning](https://physique.cmaisonneuve.qc.ca/svezina/mat/note_mat/MAT_Chap%206.2.pdf)

</h3>


## **The equations used for the N-layer Neural Network are defined above :**<br> </br>

## *Input and output data :*<br> </br>
$\Large
X =
\begin{bmatrix}
x_1^{1} & x_1^{2} & ... & x_1^{m} \\
x_2^{1} & x_2^{2} & ... & x_2^{m}
\end{bmatrix}
$
&emsp;
$\Large X \in \mathbb{R}^{n^{[0]} \times m}$
&emsp;
$;$ &emsp; &emsp;
$\Large
y =
\begin{bmatrix}
y^{1} & y^{2} & ... & y^{m}
\end{bmatrix}
$
&emsp;
$\Large y \in \mathbb{R}^{1 \times m}$<br> </br>

## *Z and W matrices for each layer (forward propagation) :*<br> </br>
$\Large
\left\{
    \begin{array}{ll}
        Z^{[1]} = W^{[1]}.X + b^{[1]} & \text{: \, first layer} \\
        \\
        Z^{[2]} = W^{[2]}.A^{[1]} + b^{[2]} & \text{: \, second layer} \\
        \\
        ... & ...\\
        \\
        Z^{[N]} = W^{[N]}.A^{[N-1]} + b^{[N]} & \text{: \, N-th layer} \\
    \end{array}
\right.
$
<br></br>

&emsp; $ \Large \text{and}$ &emsp; $\Large A^{[N]} = \dfrac{1}{1+e^{-Z^{[N]}}}$ &ensp; $\large \text{the sigmoid activation function}$ <br></br>
&emsp; &emsp; $ \Large \text{or}$ &emsp; $\Large A^{[N]} = max(0, Z^{[N]})$ &ensp; $\large \text{the ReLU activation function}$<br></br>
&emsp; &emsp; $ \Large \text{or}$ &emsp; $\Large A^{[N]} = tanh(Z^{[N]})$ &ensp; $\large \text{the tanh activation function}$ <br></br>

## *The dimensions for these matrices :* <br></br>

$ \Large Z^{[1]},A^{[1]} \in \mathbb{R}^{n^{[1]} \times m}$ &emsp; $\Large ;$ &emsp; $\Large Z^{[2]},A^{[2]} \in \mathbb{R}^{n^{[2]} \times m}$ &emsp;  $\Large ...$ &emsp; $\Large Z^{[N]},A^{[N]} \in \mathbb{R}^{n^{[N]} \times m}$ <br></br>

$\Large b^{[1]} \in \mathbb{R}^{n^{[1]} \times 1}$ &emsp; $\Large ;$ &emsp; $\Large b^{[2]} \in \mathbb{R}^{n^{[2]} \times 1}$ &emsp;  $\Large ...$ &emsp; $\Large b^{[N]} \in \mathbb{R}^{n^{[N]} \times 1}$ <br></br>

$\Large W^{[1]} \in \mathbb{R}^{n^{[1]} \times n^{[0]}}$ &emsp; $\Large ;$ &emsp; $\Large W^{[2]} \in \mathbb{R}^{n^{[2]} \times n^{[1]}}$ &emsp;  $\Large ...$ &emsp; $\Large W^{[N]} \in \mathbb{R}^{n^{[N]} \times n^{[N-1]}} $ <br></br>


## *Cost function to minimize using Gradient Descent :*
  - ## *binomial classification : Log-Loss function*
  &nbsp; &nbsp; &nbsp; $\Large L=-\dfrac{1}{m} \sum_{i=0}^{m} y_i log(A^{[N]}) + (1-y_i) log(1-A^{[N]}) $
  - ## *multinomial classification : Cross-Entropy function*
  &nbsp; &nbsp; &nbsp; $\Large L(y,\hat{y}) = - \sum_{i} y_i log(\hat{y_i}) $ <br></br>

## *Back propagation :* <br></br>
$\text{N-th layer} \ \ \ \, \Large
\left\{
    \begin{array}{ll}
        \frac{\partial L}{\partial W^{[N]}} = \frac{1}{m} dz_N . A^{[N-1]^{T}} \\
        \\
        \frac{\partial L}{\partial b^{[N]}} = \frac{1}{m} \sum_{_{axe1}} dz_N
    \end{array}
\right.
$
&emsp; $\Large with \ \ \ $ &ensp; $\Large dz_N = (A^{[N]}-y) $

$\text{(N-1)th layer}\Large
\left\{
    \begin{array}{ll}
        \frac{\partial L}{\partial W^{[N-1]}} = \frac{1}{m} dz_{N-1} . A^{[N-2]^{T}} \\
        \\
        \frac{\partial L}{\partial b^{[N-1]}} = \frac{1}{m} \sum_{_{axe1}} dz_{N-1}
    \end{array}
\right.
$
&emsp; $\Large with \ \ \ $ &ensp; $\Large dz_{N-1} = W^{[N]^{T}}.dz_N \times A^{[N-1]}(1 - A^{[N-1]}) $
 $
.\\
.\\
.\\
$
$\text{1st layer} \ \ \ \ \ \ \ \ \Large
\left\{
    \begin{array}{ll}
        \frac{\partial L}{\partial W^{[1]}} = \frac{1}{m} dz_1 . X^{T} \\
        \\
        \frac{\partial L}{\partial b^{[1]}} = \frac{1}{m} \sum_{_{axe1}} dz_1
    \end{array}
\right.
$
&emsp; $\Large with \, \ \ $ &ensp; $\Large dz_1 = W^{[2]^{T}}.dz_2 \times A^{[1]}(1 - A^{[1]}) $
