%%
% This function solves the auxiliary problem for a particular value of R_k

function [val,y_out,Ap,a] = logistic_sol_gasnikov(R_k,A,y_in,p,u,n,tau,M,N,L3,gamma,options,XXdata,d)


        a = (1/R_k + sqrt(1/R_k^2 + 4*A/R_k))/2;
        Ap = A + a;

        x = A/Ap*y_in + a/Ap*u;

        yt = TensorGasnikovInner_log(x,n,tau,M,N,L3,R_k,gamma,options,XXdata,d);
        y_out = yt + x;
        
        val = 2*(p+1)*L3/(factorial(p)*R_k)*norm(yt,2)^(p-1);