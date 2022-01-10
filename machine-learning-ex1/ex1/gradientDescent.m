function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    %X is in the format of [1 data; ...] ?x2 set is row vector
    %y is in the format of [data; ...] ?x1 column vector
    %theta is in the format of [th1; th2]
    temp = X*theta; %mx2 * 2x1 = mx1 row vector
    temp = temp - y; %mx1 still __ this is the vector of errors h-y 
    temp = X.'*temp; %2xm * mx1 = 2x1 __ this is each error multiplied by its respective x and summed
    temp = alpha*temp/m;
    theta = theta - temp;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
