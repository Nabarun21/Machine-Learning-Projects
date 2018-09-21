function G = configuration_goodness(rbm_w, visible_state, hidden_state)
%i hate using for loops so I am going to do something complicated, I am not
%going to be able to undesrtand it next time I look at this code but I
%don't care

copies_of_visible=repmat(visible_state,1,1,size(rbm_w,1));
%size(copies_of_visible)
permuted_copies_of_visible=permute(copies_of_visible,[3 1 2]);
%size(permuted_copies_of_visible)
copies_of_rbm_w=repmat(rbm_w,1,1,size(visible_state,2));
%size(copies_of_rbm_w)
mult_visible_w=permuted_copies_of_visible.*copies_of_rbm_w;
copies_of_hidden=repmat(hidden_state,1,1,size(rbm_w,2));
%size(copies_of_hidden)
permuted_copies_of_hidden=permute(copies_of_hidden,[1 3 2]);
%size(permuted_copies_of_hidden)
mult_visible_w_hidden=mult_visible_w.*permuted_copies_of_hidden;
G=sum(sum(sum(mult_visible_w_hidden)))/size(visible_state,2);



% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
% <hidden_state> is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
% This returns a scalar: the mean over cases of the goodness (negative energy) of the described configurations.
 %   error('not yet implemented');
end
