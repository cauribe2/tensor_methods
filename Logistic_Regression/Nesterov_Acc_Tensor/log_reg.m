%%
% Evaluate the logistic regression function

function [val] = log_reg(Data_label,inputs,reg_param,d)

val = (1/(d))*sum(log(1+exp(-Data_label*inputs))) + (1/2)*(reg_param)*norm(inputs,2)^2;