function partition_function =part_func(rbm_w)
    % <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
    exp_rbm_w=exp(rbm_w);
    to_add=ones(size(rbm_w));
    exp_rbw_modified=exp_rbm_w+to_add;
    vec=prod(exp_rbw_modified,2);
    add2=sum(vec)*2^(size(rbm_w,1)-1);
    partition_function=log(2^size(rbm_w,2)+add2);
    
end