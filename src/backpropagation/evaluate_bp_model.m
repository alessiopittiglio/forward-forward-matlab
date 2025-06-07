function sumerrors = evaluate_bp_model(weights, biases, batchdata, batchtargets)
%evaluate_bp_model - Evaluate a model trained with backpropagation
%   This function runs a forward pass on the provided data and computes the 
%   total number of classification errors.
%
%   Syntax
%     sumerrors = evaluate_bp_model(weights, biases, batchdata, ...
%                     batchtargets)
%
%   Input Arguments
%     weights - Weight matrices
%       cell array
%     biases - Bias vectors
%       cell array
%     batchdata - Input data
%       3D matrix (numcases x numvis x numbatches)
%     batchtargets - One-hot encoded labels
%       3D matrix (numcases x numvis x numbatches)
%
%   Output Arguments
%     sumerrors - Total number of misclassified samples
%       integer scalar
%
%   See also backpropagation_train

% =========================================================================
%   1. INITIALIZATION
% =========================================================================
numlayers = length(weights);
numbatches = size(batchdata, 3);
sumerrors = 0;
normstates = cell(1, numlayers - 1);

% =========================================================================
%   2. EVALUATION LOOP (over all batches)
% =========================================================================
for batch = 1:numbatches
    % Extract the current minibatch
    data = batchdata(:, :, batch);
    targets = batchtargets(:, :, batch);

    % ---------------------------------
    %   FORWARD PASS
    % ---------------------------------
    normstates{1} = ffnormrows(data);
    for l = 2:numlayers-1
        totin = normstates{l-1} * weights{l} + biases{l};
        states = max(0, totin); % ReLU activation
        normstates{l} = ffnormrows(states);
    end
    % Final output layer (logits)
    labin = normstates{numlayers-1} * weights{numlayers} + biases{numlayers};

    % ---------------------------------
    %   PREDICTION
    % ---------------------------------
    [~, guesses] = max(labin, [], 2);
    [~, targetindices] = max(targets, [], 2);

    % ---------------------------------
    %   ERROR CALCULATION
    % ---------------------------------
    errors = sum(guesses ~= targetindices);
    sumerrors = sumerrors + errors;
end % end of minibatches loop

end % end of function
