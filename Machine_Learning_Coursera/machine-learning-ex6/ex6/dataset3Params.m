function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
a=[0.01];
b=[0.01];
for i=2:8
    a(i)=a(i-1)*3;
    b(i)=b(i-1)*3;
end    
C=a(1);
sigma=b(1);

error=100000;
for i=1:8
    for j=1:8
        model= svmTrain(X, y, a(i), @(x1, x2) gaussianKernel(x1, x2, b(j))); 
        predictions = svmPredict(model, Xval);
        error_loop= mean(double(predictions ~= yval));
        if error_loop<error
            C=a(i);
            sigma=b(j);
            error=error_loop;
        end
    end
end







% =========================================================================


end
