%%
% This function computes the Hessian of the Hard Tensor functions at a
% particular point

function hess = hess_bad_nesterov(A_k,x,p)

hess = 3*A_k'*diag((A_k*x).^(p-1))*A_k;