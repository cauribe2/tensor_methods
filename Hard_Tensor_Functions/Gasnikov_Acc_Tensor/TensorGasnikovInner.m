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
%  -- R_k:  Current Value of the Line Search
%  -- N:  Maximum number of iterations of the inner problem loop
% 
% Output:
%  -- outer_inner:   Next iterate approximate solution

function [out_inner] = TensorGasnikovInner(y,n,tau,M,N,L3,R_k,gamma,options,A_k,p)

h= zeros(n,1);

eval_grad_x = grad_bad_nesterov(A_k,y,p);
eval_hessian_x =  hess_bad_nesterov(A_k,y,p);

gradvv = 1;

for i=1:N
%         if isnan(h)
%             1
%         end
% eval_D3 = D3_f_k_fun([y;h]);
 eval_D3 = D3_bad_nesterov(A_k,y,p,h);
 
%                  if isnan(eval_D3)
%             1
%         end

A = (tau+1)/(tau)*(eval_hessian_x + (tau/(tau+1))*R_k*eye(n));
c = eval_grad_x + 1/2*eval_D3 +...
            ( eval_hessian_x  + R_k*eye(n) - 1/2*(tau+1)/(tau)*(eval_hessian_x+2*(tau/(tau+1))*R_k*eye(n)+eval_hessian_x'))*h ...
            - (tau*(tau+1)/2*L3-  M/2)*h.^3;
        
%                 if isnan(c)
%             1
%         end

% vec_aux = @(x)(c'*x + 1/2*(A*x)'*x + gamma/4*norm(x,2)^4);
% h = fmincon(vec_aux,h,[],[],[],[],[],[],[],options);
g_aux = @(x)((1/2)*x^2 + 1/2*((sqrt(2*gamma)*x*eye(n)+A)\c)'*c);
tau_opt = fminbnd(g_aux,1e-12,10,options);
h = - (A + sqrt(2*gamma)*tau_opt*eye(n))\c;

%         if isnan(h)
%             1
%         end

gradnn = norm(eval_grad_x +  eval_hessian_x*h + 1/2*eval_D3 + M/2*h.^3  + R_k*h);
    if abs(gradnn-gradvv)<1e-8    % Early Stopping Condition
        break
    end
    gradvv = gradnn;

end

out_inner = h;