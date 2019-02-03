%%
% This function computes the gradient at a point for the bad tesor
% functions

function grad = grad_bad_nesterov(A_k,x,p)

grad = A_k'*(A_k*x).^p;
grad(1) = grad(1)-1;