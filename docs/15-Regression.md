# Linear Regression

In this chapter we consider estimation and inference of a random response, conditional on another observed variable---the independent variable, also called predictor or covariate.  <br><br>

## Simple linear regression

The simple linear regression model says that a random response $Y_i$ has conditional mean $\beta_0 + \beta_1 x_i$ given $X_i = x_i$ where $X_i$ may be a random variable.  Moreover, the responses $Y_i$, $i = 1, \ldots, n$, are independent, with common variance $\sigma^2$, and are normally distributed.  This is commonly written using the following statistical notation:
\[Y_i = \beta_0 + \beta_1 x_i + \epsilon_i, \quad \epsilon_i\stackrel{iid}{\sim} N(0, \sigma^2)\]
where $\epsilon_i$ is the "random residual"---what is left over after subtracting the mean from the response.  <br><br>

The model is "linear" because the mean $E(Y_i|x_i) = \beta_0 + \beta_1 x_i$ is a linear function (a line in two dimensions $y$ and $x$).  The model is "simple" because it is the simplest line (two dimensions).  Later we will expand the model by adding covariates, i.e., $x_{1i}, x_{2i}, \ldots$, to the linear conditional mean function and call it *multiple linear regression*.

### Estimation

We estimate $(\beta_0, \beta_1)$ simultaneously using the method of "least squares".  The method of least squares defines the line $y = \beta_0 + \beta_1 x$ that is "closest" to the points $(y_i, x_i), i=1, \ldots, n$ is the one minimizing the sum of square vertical distances (residuals) from the points to the line:
\[(\hat\beta_0, \hat\beta_1) = \arg\min_{(\beta_0, \beta_1)}\sum_{i=1}^n(y_i - \beta_0 - \beta_1x_i)^2.\]
The plot below shows these residuals for three points (1.0,1.1), (1.5,2.1), and (2.0,1.8), compared to the line $y=x$.


```r
plot(c(1,1.5,2),c(1.1,2.1,1.8), xlab = 'x', ylab = 'y', main = '', xlim = c(0.5,2.5), ylim = c(0,3))
lines(c(0.5,2.5),c(0.5,2.5))
lines(c(1,1), c(1,1.1), lty = 3)
lines(c(1.5,1.5), c(1.5,2.1), lty = 3)
lines(c(2,2), c(2,1.8), lty = 3)
```

![](15-Regression_files/figure-epub3/unnamed-chunk-1-1.png)<!-- -->

We can determine the estimators $(\hat\beta_0, \hat\beta_1)$ by minimizing the sum of squared residuals using calculus:

$$
\begin{aligned}
\frac{\partial}{\partial\beta_0} \sum_{i=1}^n(y_i - \beta_0 - \beta_1x_i)^2 \\
& = -2\sum_{i=1}^n(y_i - \beta_0 - \beta_1x_i)\\
\text{set  }0 &= -2\sum_{i=1}^n(y_i - \beta_0 - \beta_1x_i)\\
&\Rightarrow \hat\beta_0 = n^{-1}\sum_{i=1}^n(y_i - \beta_1x_i)\\
& = \bar y - \beta_1\bar x
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial}{\partial\beta_1} \sum_{i=1}^n(y_i - \beta_0 - \beta_1x_i)^2 \\
& = -2 \sum_{i=1}^n x_i(y_i - \beta_0 - \beta_1x_i)\\
\text{set  }0 &= -2 \sum_{i=1}^n x_i(y_i - \beta_0 - \beta_1x_i)\\
&\Rightarrow 0 = \sum_{i=1}^n (y_ix_i - x_i(\bar y - \beta_1 \bar x) - \beta_1 x_i^2)\quad \text{by substituting }\beta_0\\
& \beta_1\sum(x_i^2 - x_i\bar x) = \sum_{i=1}^n(y_ix_i - \bar y x_i)\\
&\Rightarrow \hat\beta_1 = \frac{\sum_{i=1}^n (y_ix_i) - n\bar y\bar x}{\sum_{i=1}^n 
(x_i^2) - n\bar x^2} = \frac{\sum_{i=1}^n [(y_i - \bar y)(x_i - \bar x)]}{\sum_{i=1}^n (x_i - \bar x)^2}
\end{aligned}
$$
### LSEs are unbiased

A nice property of the least squares method is that it produces unbiased estimators.  Consider the expectation $E(\hat\beta_1)$.  Since the $x_i'$s are non-random, treat these as constants, and find $\hat\beta_1$ is unbiased:

$$
\begin{aligned}
E(\hat\beta_1) & = \frac{\sum_{i=1}^n E((y_i - \bar y))(x_i - \bar x)}{\sum_{i=1}^n (x_i - \bar x)^2}\\
& = \frac{\sum_{i=1}^n (\beta_0 + \beta_1 x_i - n^{-1}\sum_{i=1}^n [\beta_0 +\beta_1 x_i])(x_i - \bar x)}{\sum_{i=1}^n (x_i - \bar x)^2}\\
& = \frac{\beta_1\sum_{i=1}^n ( x_i - \bar x)(x_i - \bar x)}{\sum_{i=1}^n (x_i - \bar x)^2}\\
& = \beta_1.
\end{aligned}
$$

Similarly, the estimator of the intercept is unbiased:

$$
\begin{aligned}
E(\hat\beta_0) & = E(\bar y - \hat\beta_1 \bar x)\\
& = n^{-1}\sum_{i=1}^n (\beta_0 + \beta_1x_i) - \beta_1 \bar x\\
& = \beta_0 + \beta_1 \bar x - \beta_1 \bar x\\
& = \beta_0
\end{aligned}
$$

### Estimation of the common variance

Estimating $\sigma^2$ in the regression model is similar to the method used in ANOVA.  Define the observed/fitted residuals $\hat e_i = y_i - \hat y_i = y_i - \hat\beta_0 - \hat\beta_1 x_i$.  Since $(\hat\beta_0, \hat\beta_1)$ are unbiased, we have $E(\hat e_i) = E(Y_i - \hat\beta_0 - \hat\beta_1 x_i) = 0$.  Therefore, $V(\hat e_i) = E(\hat e_i ^2)$.  And, the method of moments suggests we estimate the variance(in this case second moment) by the sample variance:

\[\hat\sigma^2 = \frac{1}{n-2}\sum_{i=1}^n (y_i - \hat\beta_0 - \hat\beta_1 x_i)^2,\]
where we divide by $n-2$ so that the resulting estimator is unbiased.  (I'll leave that as a challenging exercise for the reader).

### Inference for regression parameters

There are two main inference (and prediction) problems of interest in regression.  The first is inference on $\beta_1$, and, in particular, testing $\beta_1 = 0$.  If $\beta_1=0$ then there is no linear relationship between the covariate $x$ and the response $Y$, and $Y$ has a constant mean (so is iid, rather than independent).  Testing $\beta_1=0$ is similar to the ANOVA F test for categorical $x$, rather than continuous $x$ in regression.   
<br><br>

For inference on $\beta_1$ we need the sampling distribution of $\hat\beta_1$.  Recognize that we can write this estimator as 
\[\hat\beta_1 = \frac{\sum_{i=1}^n Y_i(x_i - \bar x)}{\sum_{i=1}^n (x_i - \bar x)^2}\] because $\sum_{i=1}^n \bar Y(x_i - \bar x) = 0$.  Then, we see that $\hat\beta_1$ is a linear combination of $Y_i$, i.e., $\hat\beta_1 = \sum_{i=1}^nc_i Y_i$ for non-random $c_i$.  By a MGF argument we have used before, linear combinations of normal random variables are also normally-distributed.  This means, 
\[\hat\beta_1 \sim N\left(\beta_1, \sigma_2\sum_{i=1}^n c_i^2\right).\]

Furthermore, by essentially the same argument as in the proof of Student's Theorem,
\[\frac{(n-2)\hat\sigma^2}{\sigma^2}\sim \chi^2(n-2)\]
so that the studentized slope estimator has a Student's $t$ distribution:
\[t = \frac{\hat\beta_1 - \beta_1}{\sqrt{\hat\sigma^2 \left[\sum_{i=1}^n (x_i - \bar x)^2\right]^{-1}}}\sim t_{n-2}\]

<br>

A test of $H_0:\beta_1 = b$ versus $H_a:\beta_1 \ne b$ rejects the null if
\[\frac{|\hat\beta_1 - b|}{\sqrt{\hat\sigma^2 \left[\sum_{i=1}^n (x_i - \bar x)^2\right]^{-1}}} > t_{1-\alpha/2, n-2}\]

Similarly, a $100(1-\alpha)\%$ CI for $\beta_1$ is given by 
\[\left(\hat\beta_1 \pm t_{1-\alpha/2, n-2}\sqrt{\hat\sigma^2 \left[\sum_{i=1}^n (x_i - \bar x)^2\right]^{-1}}\right).\]

















