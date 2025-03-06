---
title: "Linear Regression using Gradient Descent Part-1"
meta_title: "Linear Regression"
description: "Blog about understanding the theory behind Gradient Ddescent how it's used in parameter optimization in Linear Regression."
date: 2025-03-06T18:29:37+05:30
image: "/images/gallery/post-1/img1.png"
categories: ["Machine Learning", "Maths"]
author: "Raunak Wete"
tags: ["ML", "Linear Regression", "Gradient Descent"]
draft: false
---


## Introduction

Linear Regression is one of the simplest and most fundamental algorithms in **Machine Learning** and **Statistics**. It is used to model the relationship between a dependent variable (**target**) and one or more independent variables (**features**). The primary objective is to find the best-fitting line that minimizes the error between the actual and predicted values.

One of the most commonly used optimization techniques in Linear Regression is **Gradient Descent**. In this blog, we will delve into the **mathematical foundation** of Gradient Descent and its application in optimizing the parameters of a Linear Regression model.

## Understanding Linear Regression

### The Hypothesis Function

A **Linear Regression model** assumes that the relationship between the input variables \( X \) and the output variable \( Y \) is linear. The hypothesis function (prediction function) for simple linear regression (with one feature) is given by:

\[ h(\theta) = \theta_0 + \theta_1 x \]

For multiple linear regression (with multiple features), the hypothesis function is extended as:

\[ h(\theta) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n \]

Or in vector notation:

\[ h(\theta) = X\theta \]

where:

- \( X \) is the matrix of input features (including a column of ones for \( \theta_0 \)),
- \( \theta \) is the vector of parameters (coefficients),
- \( h(\theta) \) is the predicted output.

### The Cost Function

To measure how well our hypothesis function fits the data, we use the **Mean Squared Error (MSE)**, also known as the **Cost Function (Loss Function)**:

\[ J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x_i) - y_i)^2 \]

where:

- \( m \) is the number of training examples,
- \( h_{\theta}(x_i) \) is the predicted value for the \( i \)-th example,
- \( y_i \) is the actual output.

Our goal is to minimize \( J(\theta) \), which means finding the optimal parameters \( \theta \).

## Gradient Descent: Optimization Algorithm

### Concept of Gradient Descent

Gradient Descent is an iterative optimization algorithm used to find the minimum of a function. In our case, it helps find the optimal values of \( \theta \) that minimize the cost function \( J(\theta) \).

The idea is to start with initial values of \( \theta \) and update them iteratively using the gradient of the cost function until convergence. The update rule is given by:

\[ \theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j} \]

where:

- \( \alpha \) is the **learning rate** (a small positive number that controls the step size),
- \( \frac{\partial J(\theta)}{\partial \theta_j} \) is the **gradient** (partial derivative of the cost function with respect to \( \theta_j \)).

### Computing the Gradient

The partial derivatives of the cost function are calculated as follows:

\[ \frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x_i) - y_i) x_i^{(j)} \]

Thus, the update rule becomes:

\[ \theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x_i) - y_i) x_i^{(j)} \]

This is applied iteratively until convergence.

## Implementing Gradient Descent for Linear Regression

### Algorithm Steps

1. **Initialize parameters**: Set initial values for \( \theta \) (often set to zeros or small random values).
2. **Compute the cost function**: Calculate \( J(\theta) \) using the given data.
3. **Compute the gradient**: Determine the partial derivatives.
4. **Update parameters**: Adjust \( \theta \) using the update rule.
5. **Repeat** until the cost function converges.

### Choosing the Learning Rate

The learning rate \( \alpha \) is crucial:

- If \( \alpha \) is **too small**, convergence is slow.
- If \( \alpha \) is **too large**, the algorithm may not converge.

## Convergence of Gradient Descent

Gradient Descent converges when the change in the cost function is minimal (i.e., when \( J(\theta) \) no longer decreases significantly). This can be monitored using:

\[ | J(\theta_{new}) - J(\theta_{old}) | < \epsilon \]

where \( \epsilon \) is a small threshold value.

## Conclusion

In this blog, we explored **Linear Regression** and its optimization using **Gradient Descent**. We covered the hypothesis function, cost function, gradient descent update rule, and implementation details.

In **Part-2**, we will implement Linear Regression with Gradient Descent using Python and visualize the learning process.

Stay tuned!

---
