function [final_weights, final_biases, loss_history] = train_bp_model( ...
    batchdata, batchtargets, ...
    validbatchdata, validbatchtargets, ...
    maxepoch, restart, ...
    use_normalization)
%train_bp_model - Train a neural network using backpropagation
%   This function is designed as a "dual" to Hinton's ffnew.m, using the 
%   same architecture, data loading, and hyperparameters to ensure a fair 
%   comparison. It trains a feed-forward neural network with ReLU 
%   activations and row-wise normalization on the MNIST dataset.
%
%   Syntax
%     [final_weights, final_biases, loss_history] = train_bp_model( ...
%         batchdata, batchtargets, ...
%         validbatchdata, validbatchtargets, ...
%         maxepoch, restart, ...
%         use_normalization)
%
%   Input Arguments
%     batchdata - Training inputs
%       3D matrix (numcases x numvis x numbatches)
%     batchtargets - Training labels
%       3D matrix (numcases x numlab x numbatches)
%     validbatchdata - Validation inputs
%       3D matrix
%     validbatchtargets - Validation labels
%       3D matrix
%     maxepoch - Maximum number of training epochs
%       positive integer scalar
%     restart - Flag to re-initialize training (1) or continue (0)
%       1 | 0
%     use_normalization - Flag to normalize data (true) or not (false)
%       true | false
%
%   Output Arguments
%     final_weights - Final weight matrices
%       cell array
%     final_biases - Final bias vectors
%       cell array
%     loss_history - Training loss per epoch
%       vector

% =========================================================================
%   1. HYPERPARAMETERS & CONFIGURATION
% =========================================================================
finaltest = 0;
myrandomseed = 17;
wc = 0.001;         % Weight cost (L2 regularization).
epsilon = 0.25;     % Learning rate for weight updates.
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
        if use_normalization
            normstates{1} = ffnormrows(data);
        else
            states{1} = data;
        end
        for l = 2:numlayers-1
            if use_normalization
                totin{l} = normstates{l-1} * weights{l} + biases{l};
                states{l} = max(0, totin{l}); % ReLU activation
                normstates{l} = ffnormrows(states{l});
            else
                totin{l} = states{l-1} * weights{l} + biases{l};
                states{l} = max(0, totin{l});
            end
        end

        % Output layer (pre-softmax logits)
        if use_normalization
            totin{numlayers} = normstates{numlayers-1} * weights{numlayers} + biases{numlayers};
        else
            totin{numlayers} = states{numlayers-1} * weights{numlayers} + biases{numlayers};
        end

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
            if use_normalization
                dCbydweights{l} = normstates{l-1}' * dCbydin{l};
            else
                dCbydweights{l} = states{l-1}' * dCbydin{l};
            end     
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
       valsumerrors = evaluate_bp_model(weights, biases, ...
        validbatchdata, validbatchtargets, ...
        use_normalization);
       fprintf(1, 'Softmax valid errs %4i  \n', valsumerrors);
    end
end % end of epochs loop

% =========================================================================
%   4. OUTPUT
% =========================================================================
final_weights = weights;
final_biases = biases;

end % end of function
