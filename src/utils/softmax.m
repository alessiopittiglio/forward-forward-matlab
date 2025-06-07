function P = softmax(A)
%softmax - Computes the softmax function for a matrix of logits
%   This function returns a matrix P of the same size as A, where each row 
%   of P is a probability distribution (i.e., its elements are non-negative 
%   and sum to 1).
%
%   Syntax
%     P = softmax(A)
%
%   Input Arguments
%     A - Input logits
%         matrix
%
%   Output Arguments
%     P - Output probabilities
%         matrix

    % Subtract the max for numerical stability (prevents overflow in exp)
    A_max = max(A, [], 2);
    exp_A = exp(A - A_max);

    % Sum the exponentials along the rows
    sum_exp_A = sum(exp_A, 2);

    % Divide to get the probabilities
    P = exp_A ./ sum_exp_A;
end
