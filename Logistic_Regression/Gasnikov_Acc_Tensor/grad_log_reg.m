%%
% Evaluate the gradient of the logistic regression function

function [grad] = grad_log_reg(Data_label,inputs,reg_param,d)


    t2 = 1./(exp(Data_label*inputs)+1);
    grad = -(1/(d))*(t2'*Data_label)' + reg_param*inputs;