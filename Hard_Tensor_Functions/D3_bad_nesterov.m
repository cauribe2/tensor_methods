%%
% This function computes the directional third order derivative of the Hard
% Tensor functions at a particular point x, for a particular direction h

function D3 = D3_bad_nesterov(A_k,x,p,h)

D3 = 6*sum(A_k'*((A_k*x).*(A_k*h).^(p-1)),2);