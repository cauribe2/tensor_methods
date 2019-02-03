%%
% This function computes an approximate soluttion to the auxiliary
% optimization problem for the accelerated tensor method proposed by
% Nesterov

% Input parameters:
%  -- y:  Current approximate solution
%  -- n:  Dimension of the problem
%  -- tau:  Method parameter
%  -- M:  Method parameter
%  -- L3:  Third Order Lipschitz Constant
%  -- gamma:  Subproblem parameter
%  -- options:  Auxiliary scalar problem Matlab toolbox opt. parameters
%  -- p:  Order of smoothness of the objective function (p=3 only)
%  -- A_k:  Hard Tensor Function paramer
%  -- N:  Maximum number of iterations of the inner problem loop
% 
% Output:
%  -- outer_inner:   Next iterate approximate solution


function [out_inner] = TensorNesterovInner(y,n,tau,M,L3,gamma,options,N,A_k,p)

h= zeros(n,1);

eval_grad_x = grad_bad_nesterov(A_k,y,p);
eval_hessian_x =  hess_bad_nesterov(A_k,y,p);

gradvv = 1;

for i=1:N
    

eval_D3 = D3_bad_nesterov(A_k,y,p,h);

A = (tau+1)/(tau)*eval_hessian_x;
c = eval_grad_x + 1/2*eval_D3 +  (eval_hessian_x  ...
            - 1/2*(tau+1)/(tau)*(eval_hessian_x+eval_hessian_x'))*h  ...
            - (tau*(tau+1)/2*L3-M/2)*h.^3;

g_aux = @(x)((1/2)*x^2 + 1/2*((sqrt(2*gamma)*x*eye(n)+A)\c)'*c);
tau_opt = fminbnd(g_aux,1e-12,10,options);  % Computes solution of scalara auxiliary problem
h = - (A + sqrt(2*gamma)*tau_opt*eye(n))\c;

gradnn = norm(eval_grad_x +  eval_hessian_x*h + 1/2*eval_D3 + M/2*h.^3  );

    if abs(gradnn-gradvv)<1e-8   % Early stopping condition if gradient hasn't changed much
        break
    end
    gradvv = gradnn;

end
out_inner = h;