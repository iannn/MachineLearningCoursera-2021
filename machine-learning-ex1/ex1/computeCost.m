function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%X is in the format of [1 input] with m rows
%y is in the format of [output] with m rows
%theta are the predictive scalars
%this course uses y = b + mx, NOT y=mx+b

temp = X*theta;
temp = temp - y;
J = temp.'*temp / (2*m);

% =========================================================================

end
