function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
diff = zeros(m, 1);
num_features = size(X,2);
fprintf("^^^^^^^^^^^^^^^^^^^^^^^^^^^");
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    prediction = X*theta;
    diff = prediction - y;
    %i = 1;
    %for i =  1:num_features
    theta(1) = theta(1) - sum (diff);
    theta(2) = theta(2) - sum (diff.* X(:,2));
    theta(3) = theta(3) - sum (diff.* X(:,3));
    %fprintf("theta (%d) : %f", i, theta(i));
    %end
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end
fprintf("theta : %f %f %f", theta(1), theta(2), theta(3));
end
