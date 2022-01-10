function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%My Solution
%Part 1, as stated above: compute the output
%I have redone properly because z is the output, and a is the proper layer with 1s for the bias
a1 = [ones(m,1), X];
z2 = a1*Theta1';
a2 = [ones(m,1) sigmoid(z2)];
z3 = a2*Theta2';
hx = sigmoid(z3);  %no bias. I don't want to call a3 because it's the output and it doesn't have bias

%Upon rereading the Ex4 instructions again, the output recoding is explained once
%It is right below the cost function but as with many things in this course, it is NOT how I would have explained anything
%This course seems to have a lot of notes everywhere. matrix dimensions are given in yet another thread related to exercise tutorials, and then through to week 5
%I pieced most of this together through dimensional analysis because the variables and math aren't broken down
%For the sake of note consistency, if the input "y" is 8, the 8th position in the recoded expected is 1
Y_r = zeros(m,num_labels);
for i=1:m
  y_r(i, y(i))=1; %The sea of threads on the forum about this all talk about the "identity" matrix but i think that's stupid. you're assigning a "1" to a specific location - that's all
end;
%upon reading even more threads and mentor threads, and mentor replies, they use the identity matrix because rows are in the correct form
%The mentor lines are: y_matrix = eye(num_labels)(y,:) and "eye_matrix = eye(num_labels); y_matrix = eye_matrix(y,:)"


%prediction and expected are now in the correct format. Double sum the equation
%Now with regularization. Recall that the bias term isn't used. Ex4 says regularization is second but here says 3. Submission agrees with Ex4

reg = lambda*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)))/2/m;
J = sum(sum((-y_r).*log(hx) - (1-y_r).*log(1-hx), 2))/m + reg;

%Gradient checking backpropagation section
%As per Ex4 instructions, impliment the 2 equations
%delta3 = hx - y
%delta2 = Theta2-T * delta3 * sigmoidGradient z2

delta3 = hx .- y_r;
%I don't transpose according to dimensional analysis and I decrimented z so z2 is actually 5000x25. Needs a column of 1s
delta2 = (delta3*Theta2).*sigmoidGradient([ones(m, 1) z2]);
delta2 = delta2(:, 2:end); %instructions say to remove bias layer
##Dimensions check out so move on to capital delta computation

%Recall that Theta1 is 25x401 and Theta2 is 10x26, so the big m examples matrices need to be transposed
%D_L = D_L + d_L+1 * Transpose(a_L)
%D isn't calculated anywhere else so the bias/initial term is 0 here
%I had a dimension mismatch so I transposed delta instead
DDelta2 = delta3'*a2;
DDelta1 = delta2'*a1;

%Gradient time
Theta2_grad = DDelta2./m + (lambda/m)*[zeros(size(Theta2,1), 1) Theta2(:, 2:end)];
Theta1_grad = DDelta1./m + (lambda/m)*[zeros(size(Theta1,1), 1) Theta1(:, 2:end)];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
