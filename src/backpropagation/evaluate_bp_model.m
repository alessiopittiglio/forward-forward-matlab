function sumerrors = evaluate_bp_model(weights, biases, batchdata, batchtargets, use_normalization)
%evaluate_bp_model - Evaluate a model trained with backpropagation
%   This function runs a forward pass on the provided data and computes the 
%   total number of classification errors.
%
%   Syntax
%     sumerrors = evaluate_bp_model(weights, biases, 
%                                   batchdata, batchtargets, ...
%                                   use_normalization)
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
%     use_normalization - Flag to normalize data (true) or not (false)
%       boolean
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
states = cell(1, numlayers);
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
    if use_normalization
        normstates{1} = ffnormrows(data);
    else
        states{1} = data;
    end
    for l = 2:numlayers-1
        if use_normalization
            totin = normstates{l-1} * weights{l} + biases{l};
            states = max(0, totin); % ReLU activation
            normstates{l} = ffnormrows(states);
        else
            totin{l} = states{l-1} * weights{l} + biases{l};
            states{l} = max(0, totin{l});
        end
    end
    % Final output layer (logits)
    if use_normalization
        labin = normstates{numlayers-1} * weights{numlayers} + biases{numlayers};
    else
        labin = states{numlayers-1} * weights{numlayers} + biases{numlayers};
    end

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
