%%
% Evaluate the Third order directional derivative of the logistic
% regression function

function d3 = D3h2_log_reg(Datax,inputs,d,h)

t1 = D3_uni_log(-Datax*inputs);
t2 = (Datax*h).^2;

d3 = - 1/d*sum(Datax'*(t1.*t2),2);