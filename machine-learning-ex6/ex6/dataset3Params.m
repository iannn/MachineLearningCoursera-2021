function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
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

%I'm literally just cycling through looking for the lowest error here
%function calls are found in ex6.m and their respective filesep

##Generic values to try
##Needs to be a row vector for for i = array to cycle through array properly
pickC = [0.1 0.2 0.3 0.5 0.7 1 1.5 2 3 5 7 10];
pickS = [0.1 0.2 0.3 0.5 0.7 1 1.5 2 3 5 7 10];

min_error = inf; #the highest number so all other guesses will be <=
##loop C, sigma
##train model for given C and sigma
##predict and compute error
##check current error with min error
for iC = pickC
  for jS = pickS
    model= svmTrain(X, y, iC, @(x1, x2) gaussianKernel(x1, x2, jS));
    predictions = svmPredict(model, Xval);#Xval is is an n x 2
    error = mean(double(predictions ~= yval));
    
    if error < min_error
      min_error = error;
      C = iC;
      sigma = jS;
    endif
    
  endfor
endfor





% =========================================================================

end
