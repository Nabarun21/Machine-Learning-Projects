function ret = cd1(rbm_w, visible_data)
    visible_data = sample_bernoulli(visible_data);
    
    hidden_zero_probs=logistic(rbm_w*visible_data);
    hidden_zero=sample_bernoulli(hidden_zero_probs);
    
    visible_probs_one=logistic(rbm_w'*hidden_zero);
    visible_one=sample_bernoulli(visible_probs_one);
    
    hidden_probs_one=logistic(rbm_w*visible_one);
%    hidden_one=sample_bernoulli(hidden_probs_one);
    hidden_one=hidden_probs_one;
    
    grad_0=configuration_goodness_gradient(visible_data,hidden_zero);
    grad_1=configuration_goodness_gradient(visible_one,hidden_one);
    
    ret=grad_0-grad_1;

% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
  %  error('not yet implemented');
end
