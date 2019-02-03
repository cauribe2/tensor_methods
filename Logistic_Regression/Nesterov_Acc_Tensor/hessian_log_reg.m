%
%% Evaluate the Hessian of the Logistic Regression function

function [hess] = hessian_log_reg(Data_label,inputs,d)

   t2 = 1./(exp(-Data_label*inputs)+1);
   hess = 1/d*Data_label'*diag(t2.*(1-t2))*Data_label;