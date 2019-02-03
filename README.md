# tensor_methods
Implementation of Optimal Tensor Methods for convex optimization

In this project we implement two tensor methods for the minimization of p-th order smooth convex functions. We compare the performance of the accelerated tensor method proposed in (Nesterov,2018), and the optimal tensor method proposed in (Ganikov,2018).

[1]. Gasnikov, Alexander, et al. "The global rate of convergence for optimal tensor methods in smooth convex optimization." arXiv preprint arXiv:1809.00382 (2018).

[2]. Nesterov, Yu. "Implementable tensor methods in unconstrained convex optmization.—2018.—CORE Discussion Papers 2018005."

We develop the particular code for two classes of problems, namely: (1) The class of convex functions described in (Nesterov,2018) which are hard for tensor methods, we will refer to this functions as "hard tensor functions", (2) logistic regression problems.

