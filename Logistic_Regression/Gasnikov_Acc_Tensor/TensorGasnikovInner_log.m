%%
% Computes the solution for the inner problem

function [out_inner] = TensorGasnikovInner_log(y,n,tau,M,N,L3,R_k,gamma,options,XXdata,d)

h= zeros(n,1);

eval_grad_x = grad_log_reg(XXdata,y,0,d);
eval_hessian_x =  hessian_log_reg(XXdata,y,d);

gradvv = 1;

for i=1:N
    
eval_D3 = D3h2_log_reg(XXdata,y,d,h);

A = (tau+1)/(tau)*(eval_hessian_x + (tau/(tau+1))*R_k*eye(n));
c = eval_grad_x +  eval_hessian_x*h + 1/2*eval_D3 + M/2*h.^3  + R_k*h ...
            - 1/2*(tau+1)/(tau)*(eval_hessian_x+2*(tau/(tau+1))*R_k*eye(n)+eval_hessian_x')*h - tau*(tau+1)/2*L3*h.^3;

g_aux = @(x)((1/2)*x^2 + 1/2*((sqrt(2*gamma)*x*eye(n)+A)\c)'*c);
tau_opt = fminbnd(g_aux,1e-12,10,options);
h = - (A + sqrt(2*gamma)*tau_opt*eye(n))\c;


gradnn = norm(eval_grad_x +  eval_hessian_x*h + 1/2*eval_D3 + M/2*h.^3  + R_k*h);
    if abs(gradnn-gradvv)<1e-8    % Early termination condition
        break
    end
    gradvv = gradnn;

end
out_inner = h;