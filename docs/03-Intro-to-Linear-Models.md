# Introduction to Linear Models

In this section we define linear models, provide simple examples, and analyze linear models for one- and two-sample problems.

## Defining the linear model

Every linear model defines a linear relationship between an independent variable $Y$ and a dependent variable $X$, including a random term $\epsilon$:
\begin{equation}
Y = X\beta + \epsilon
  (\#eq:linmod)
\end{equation} 
Usually, $X$ is a fixed or non-random variable, while $\epsilon$ is a random variable representing variation due to a random sampling mechanism, so that $Y$ is a random outcome.  Further, in \@ref(eq:linmod) $Y = (Y_1, \ldots, Y_n)^\top$ is an $n\times 1$ vector of outcomes/responses, $X$ is an $n\times p$ matrix of fixed variables/covariates ($p<n$), $\epsilon = (\epsilon_1, \ldots, \epsilon_n)^\top$ is an $n\times 1$ vector of random variables, and $\beta = (\beta_1, \ldots, \beta_p)^{\top}$ is a $p\times 1$ coefficient vector of unknown parameters.    

The *least-squares model* or just called the *linear model* is the above model with few or no additional assumptions although, to estimate the unknown parameter $\beta$, which characterizes the relationship between $X$ and $Y$, assuming $E(\epsilon_i) = 0$ is very helpful and usually reasonable.

The *Gauss-Markov model*---which we will tacitly use throughout the course and examine in detail at the end of the semester---makes the assumptions $E(\epsilon_i) = 0$, $E(\epsilon_i^2) = \sigma^2$, which means the ``error" term $\epsilon$ has the same variance for each random sample (homogeneous or constant variance), and $E(\epsilon_i\epsilon_j)=0$.  Or, in other words, the last two assumptions may be written $Cov(\epsilon) = \sigma^2 I_{n}$ where $I_n$ is the $n\times n$ identity matrix.  

## Gauss-Markov model for one-sample

Let $P$ be a normal population with mean $\beta$ and variance $\sigma^2$.  Let $Y_i\stackrel{iid}{\sim}P$.  Then, we may write
\begin{equation}
\begin{aligned}
Y_i &= \beta x_i + \epsilon_i, \quad i = 1,\ldots, n, \,\,\text{or}\\
Y &= X\beta + \epsilon
\end{aligned}
\end{equation}
where $x_i = 1$, so that $X = (1, 1, ..., 1)^{\top}$ is an $n\times 1$ vector of ones.

