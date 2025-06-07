function [final_weights, final_biases, loss_history] = backpropagation_train( ...
    batchdata, batchtargets, ...
    validbatchdata, validbatchtargets, ...
    finaltestbatchdata, finaltestbatchtargets, ...
    maxepoch, restart)
%backpropagation_train - Train a feed-forward neural network using backpropagation
%   This function is designed as a "dual" to Hinton's ffnew.m, using the 
%   same architecture, data loading, and hyperparameters to ensure a fair 
%   comparison. It trains a feed-forward neural network with ReLU 
%   activations and row-wise normalization on the MNIST dataset.
%
%   Syntax
%     [final_weights, final_biases, loss_history] = backpropagation_train()
%
%   Output Arguments
%     final_weights - Cell array of weight matrices for each layer
%     final_biases - Cell array of bias vectors for each layer
%     loss_history - Vector containing training loss per epoch
%
%   Notes
%     - This function relies on several variables being present in the 
%       calling workspace (e.g., maxepoch, restart, and the data and 
%       targets
%       variables). This design choice mirrors the original ffnew.m script.
%     - It also calls external helper functions for testing 
%       (evaluate_bp_model) and normalization (ffnormrows)

% =========================================================================
%   1. HYPERPARAMETERS & CONFIGURATION
% =========================================================================
finaltest = 0;
myrandomseed = 17;
wc = 0.001;         % Weight cost (L2 regularization).
epsilon = 0.4;      % Learning rate for weight updates.
epsgain = 1;        % Multiplier on all weight changes, can decay over time.
delay = 0.9;        % Momentum term for smoothing gradients (1 - 0.1).

[numcases, numvis, numbatches] = size(batchdata);
numlab = 10;
% NOTE: The architecture is intentionally kept the same as in ffnew.m
layernums = {numvis 1000 1000 1000 numlab};
numlayers = size(layernums, 2);
tiny = exp(-50);    % Small constant to prevent log(0).

freq = 1;           % Base frequency for logging.
printfreq = freq; 
testfreq = 5 * freq;

% =========================================================================
%   2. MODEL INITIALIZATION (if restart is set)
% =========================================================================

if restart == 1
    rng(myrandomseed); % Set the seed for reproducible weight initialization.

    % The learning rate decays linearly after half of the epochs.
    epsgain = 1;
    restart = 0; % Allows for interruption and continuation.
    epoch = 1;

    % Initialize network parameters and gradient holders
    states = cell(1,numlayers);
    totin = cell(1,numlayers);
    normstates = cell(1,numlayers);
    weights = cell(1,numlayers);
    biases = cell(1,numlayers);
    dCbydin = cell(1,numlayers);
    dCbydweights = cell(1, numlayers);
    dCbydbiases = cell(1,numlayers);
    weightsgrad = cell(1,numlayers);
    biasesgrad = cell(1,numlayers);
    
    for l = 2:numlayers
        % Initialize weights with scaled random values (Xavier init).
        weights{l} = (1 / sqrt(layernums{l-1})) * randn(layernums{l-1}, layernums{l});
        biases{l} = 0*ones(1, layernums{l});
        % Pre-allocate gradient matrices
        weightsgrad{l} = zeros(layernums{l-1}, layernums{l});
        biasesgrad{l} = zeros(1, layernums{l});
    end

    % Convert data to single precision to save memory.
    batchdata = single(batchdata);
    batchtargets = single(batchtargets);
    validbatchdata = single(validbatchdata);
    validbatchtargets = single(validbatchtargets);
    finaltestbatchdata = single(finaltestbatchdata);
    finaltestbatchtargets = single(finaltestbatchtargets);
end

% =========================================================================
%   3. TRAINING LOOP
% =========================================================================
loss_history = zeros(1, maxepoch);

for epoch = epoch:maxepoch
    trainlogcost = 0;    

    % Learning rate decay schedule
    if epoch <= maxepoch / 2
        epsgain = 1;
    else
        epsgain = (1 + 2 * (maxepoch - epoch)) / maxepoch;
    end

    for batch = 1:numbatches
        data = batchdata(:, :, batch);
        targets = batchtargets(:, :, batch);

        % ---------------------------------
        %   FORWARD PASS
        % ---------------------------------
        normstates{1} = ffnormrows(data);
        for l = 2:numlayers-1
            totin{l} = normstates{l-1} * weights{l} + biases{l};
            states{l} = max(0, totin{l}); % ReLU activation
            normstates{l} = ffnormrows(states{l});
        end

        % Output layer (pre-softmax logits)
        totin{numlayers} = normstates{numlayers-1} * weights{numlayers} + biases{numlayers};

        % Softmax and Cross-Entropy Loss
        labin = totin{numlayers};
        labin = labin - repmat(max(labin, [], 2), 1, numlab);
        unnormlabprobs = exp(labin);
        trainpredictions = unnormlabprobs ./ repmat(sum(unnormlabprobs, 2), 1, numlab);
        correctprobs = sum(trainpredictions .* targets, 2);

        thistrainlogcosts = - log(tiny + correctprobs);
        trainlogcost = trainlogcost + sum(thistrainlogcosts) / numbatches;

        % ---------------------------------
        %   BACKWARD PASS
        % ---------------------------------
        % Initial gradient at the output (dLoss/dLogits)
        dCbydin{numlayers} = targets - trainpredictions;

        % Propagate gradients backwards through hidden layers
        for l = numlayers-1:-1:2
            dCbydin{l} = (dCbydin{l+1} * weights{l+1}') .* (totin{l} > 0);
        end

        % ---------------------------------
        %   WEIGHTS UPDATE
        % ---------------------------------
        for l = numlayers:-1:2
            dCbydweights{l} = normstates{l-1}' * dCbydin{l};
            dCbydbiases{l} = sum(dCbydin{l});
            
            % Apply momentum
            weightsgrad{l} = delay * weightsgrad{l} + (1 - delay) * dCbydweights{l} / numcases;
            biasesgrad{l} = delay * biasesgrad{l} + (1 - delay) * dCbydbiases{l} / numcases;

            % Update weights and biases with learning rate and weight decay
            biases{l} = biases{l} + epsgain * epsilon * biasesgrad{l};
            weights{l} = weights{l} + epsgain * epsilon * (weightsgrad{l} - wc * weights{l});
        end
    end % end of minibatches loop

    loss_history(epoch) = trainlogcost;
    if rem(epoch, printfreq) == 0 
        fprintf(1, 'ep %3i gain %1.3f trainlogcost %3.4f ', ...
                   epoch,    epsgain, trainlogcost) 
        fprintf(1, '\n');
    end

    % Perform validation test at specified frequency
    if rem(epoch, testfreq) == 0
       evaluate_bp_model;
    end
end % end of epochs loop

% =========================================================================
%   4. FINAL TEST & OUTPUT
% =========================================================================
finaltest = 1;
evaluate_bp_model;

% Assign final parameters to output variables
final_weights = weights;
final_biases = biases;

end % end of function
