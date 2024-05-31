\[ 
X = \begin{pmatrix} 
x_{1.} \\ 
\vdots \\ 
x_{n.}
\end{pmatrix} 
\quad \implies \quad 
X \in \mathbb{R}^{n \times p} 
\]

Only assumption on $X$ is that its mean vector and covariance matrix exist.

Let $\delta = (\delta_1, \ldots, \delta_p)'$ be called the weighting vector. Then the weighted average is:
\[ 
\delta' X = \sum_{j=1}^{p} \delta_j X_j \quad \text{such that} \quad \sum_{j=1}^{p} \delta_j = 1 
\]

In order to properly choose $\delta$, we will maximize the variance:
\[ 
\max_{\delta \in M} \text{Var}(\delta' X) = \max_{\delta \in M} \delta' \text{Var}(X) \delta 
\]
\[ 
M = \{\delta : \|\delta\| = 1\}
\]

We call $\Sigma = \text{Var}(X)$.

\[ 
\max_{\delta \in M} \delta' \Sigma \delta 
\]

This is a quadratic convex maximization problem with nonlinear constraints.

\[ 
\mathcal{L} (\delta, \lambda) = \delta' \Sigma \delta - \lambda (\delta' \delta - 1) 
\]

- $\nabla_{\delta} \mathcal{L} = 0 : \Sigma \delta = \lambda \delta$

- $\lambda$: eigenvalue 
- $\delta$: eigenvector of $\lambda$
- $\delta' \Sigma \delta = \lambda \delta' \delta = \lambda$


Thus,
\[ 
\max_{\delta} \lambda \quad \text{subject to} \quad \Sigma \delta = \lambda \delta, \|\delta\| = 1 
\]
So we call $\delta$ the first principal component.

Let $\delta^1, \ldots, \delta^j$ fit $j$ principal components. The $j+1$ principal component is the result of:
\[ 
\max \delta' \Sigma \delta 
\]
subject to
\[ 
\delta' \delta^i = 0 \quad \forall i \in \{1, \ldots, j\} 
\]
\[ 
\delta' \delta = 1 
\]

The $j$-th principal component is an eigenvector associated to the $j$-th eigenvalue of $\Sigma$.

**Sources:** 
- 2015_Book_AppliedMultivariateStatistical.pdf
- ADM_PCA.pdf