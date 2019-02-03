%% 
% This function computes the function value for the bad tensor functions

function val = bad_nesterov(x,A_k,p)

 val = 1/(p+1)*sum(abs(A_k*x).^(p+1)) - x(1);