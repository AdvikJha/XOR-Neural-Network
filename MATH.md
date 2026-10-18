## Math
This is the math used for this neural network. 

# Mathematical Prerequisites
Knowledge of these topics is required to understand the math involved in this model.
- Precalculus: Summations, Sigma Notation, Sigmoid
- Calculus I: Derivatives and the Chain Rule
- Multivariable Calculus: Partial Derivatives

# Sigmoid Activation Function
The sigmoid function turns any real number into a number between 0 and 1.

$$
\sigma(z) = \frac{1}{1+e^{-z}}
$$

## Forward Propogation
The formulas for each neuron

Hidden Neuron 1:

$$
z_1 = w_1 x_1 + w_2 x_2 + b_1
$$

$$
a_1 = \sigma(z_1)
$$

Hidden Neuron 2:

$$
z_2 = w_3 x_1 + w_4 x_2 + b_2
$$

$$
a_2 = \sigma(z_2)
$$

Output Neuron 1:

$$
z_3 = w_5 a_1 + w_6 a_2 + b_3
$$

$$
\hat{y} = \sigma(z_3)
$$

## Mean Squared Error
Using a squared error emphasizes large errors while diminishing small errors

$$
L = \frac{1}{4} \sum^{4}_{i=1} (\hat{y} - y)^{2}
$$

## Backpropagation
For this step, we calculate the derivatives of Loss with respect to every weight and bias one at a time.

Output Layer:

$$
\delta_3 = 2(\hat{y} -y)\hat{y}(1-\hat{y})
$$

$$
\frac{\partial L}{\partial w_5} = \delta_3 a_1
$$

$$
\frac{\partial L}{\partial w_6} = \delta_3 a_2
$$

$$
\frac{\partial L}{\partial b_3} = \delta_3
$$

Hidden Layer:

$$
\delta_1 = \delta_3 w_5 a_1(1-a_1)
$$

$$
\delta_2 = \delta_3 w_6 a_2(1-a_2)
$$

$$
\frac{\partial L}{\partial w_1} = \delta_1 x_1
$$

$$
\frac{\partial L}{\partial w_2} = \delta_1 x_2
$$

$$
\frac{\partial L}{\partial w_3} = \delta_2 x_1
$$

$$
\frac{\partial L}{\partial w_4} = \delta_2 x_2
$$

$$
\frac{\partial L}{\partial b_1} = \delta_1
$$

$$
\frac{\partial L}{\partial b_2} = \delta_2
$$

## Gradient Descent
This time I chose to keep the learning rate ($\alpha$) as 1 after experimenting and finding that 1 had the highest accuracy without risking sigmoid saturation. In this equation theta represents any of the weights or biases.

$$
\theta_{new} \leftarrow \theta_{old} - \alpha \frac{\partial L}{\partial \theta}
$$
