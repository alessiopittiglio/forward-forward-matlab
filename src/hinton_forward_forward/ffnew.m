% You need to set restart=1 externally to this script.
% It then sets restart=0 which allows you to interrupt and then carry on from where you were.

% You need to set maxepoch=<number of epochs>  externally to this script.

labelstrength = 1; % scaling up the activity of the label pixel doesn't seem to help much.
minlevelsup = 3; % used in training softmax predictor. Does not use hidden layers lower than this.
minlevelenergy = 3; % used in computing goodness at test time. Does not hidden layers lower than this.
finaltest = 0;

myrandomseed = 17;
wc =  0.001; %% weightcost on forward weights.
supwc =  0.003; %% weightcost on label prediction weights.

epsilon = 0.01; % learning rate for forward weights.
epsilonsup = 0.1;  % learning rate for linear softmax weights.
                    
% I do most hyperparameter searches in multiples of about three: .01, .03, .1, .3, 1, 3, 10, etc. 

epsgain = 1; %%  multiplier on all weight changes. Can be reduced during learning.
delay = 0.9; %%. used for smoothing the gradient over minibatches. 0.9 = 1 - 0.1

lambdamean = 0.03;
% Peer normalization: we regress the mean activity of each neuron towards the average mean for its layer. 
% This prevents dead or hysterical units. We pretend there is a gradient even when hidden units are off.
% Choose strength of regression (lambdamean) so that average activities are similar but not too similar.

temp = 1;  %rescales the logits used for deciding fake vs real 

[numcases, numvis, numbatches]=size(batchdata);
numlab=10;
numdims = numvis;  % for backward compatibilty with my graphics scripts.

layernums = {numvis 1000 1000 1000 numlab};  % must enter numvis for first layer and numlab for last.
numlayers = size(layernums, 2) ;

tiny = exp(-50); % for preventing divisions by zero. 

freq = 1;  %% freq determines the frequency of printing various things.
printfreq = freq; 
rmsfreq = 5*freq;
testfreq = 5*freq;
histfreq = freq;

if restart==1
    rng(myrandomseed);
    % Reset the seed so we always initialize the weights the same way.
    % This reduces variance between runs when searching for hyper-paramteters.
    
    epsgain = 1; %% I make the learning rate decay linearly after maxepoch/2;
    restart = 0;
    epoch = 1;
    
    states = cell(1,numlayers); % pre-normalized states
    totin = cell(1,numlayers); % total input to each neuron
    normstates = cell(1,numlayers);
    weights = cell(1,numlayers); %the forward weights. weights{2} is incoming weights to layer 2.
    biases = cell(1,numlayers);
    
    posprobs = cell(1,numlayers); % column vector of probs that positive cases are positive.
    negprobs = cell(1,numlayers); % column vector of probs that negative cases are POSITIVE.
    
    dCbydin = cell(1,numlayers); % gradients of goodness w.r.t. total input to a hidden unit.
    posdCbydweights = cell(1, numlayers); %gradients of probability of correct real/fake decision w.r.t. weights.
    negdCbydweights = cell(1, numlayers); % gradients on negative cases.
    posdCbydbiases = cell(1,numlayers);
    negdCbydbiases = cell(1,numlayers);
    weightsgrad = cell(1,numlayers); % The gradients for the weights smoothed over minibatches.
    biasesgrad = cell(1,numlayers);
    pairsumerrs = cell(1,numlayers); 
    % number of times an image with an incorrect label has higher goodness than the image with the correct label.
    
    for l = 2:numlayers-1
        meanstates{l} = 0.5*ones(1, layernums{l}); %initialize the running average of the mean activity of a hidden unit.
    end;
    
    for l = 2:numlayers
        %weights{l} = initialweightsize*randn(layernums{l-1}, layernums{l});
        weights{l} = (1/sqrt(layernums{l-1}))*randn(layernums{l-1}, layernums{l}); %scales initial weights by sqrt(fanin).
        biases{l} = 0*ones(1, layernums{l});
        posdCbydweights{l} = zeros(layernums{l-1}, layernums{l});
        negdCbydweights{l} = zeros(layernums{l-1}, layernums{l});
        posdCbydbiases{l} = zeros(1, layernums{l});
        negdCbydbiases{l} =  zeros(1, layernums{l});
        weightsgrad{l} = zeros(layernums{l-1}, layernums{l});
        biasesgrad{l} = zeros(1, layernums{l});
    end;
    
    
    
    supweightsfrom = cell(1,numlayers-1); % the weights used for predicting the label from the higher hidden layer activities.
    supweightsfromgrad = cell(1,numlayers-1); % the smoothed gradients.
    for l = 2:numlayers-1
        supweightsfrom{l} = zeros(layernums{l}, numlab);  
        supweightsfromgrad{l} = zeros(layernums{l}, numlab); 
    end;
    
    batchdata=single(batchdata);
    batchtargets=single(batchtargets);
    validbatchdata=single(validbatchdata);
    validbatchtargets=single(validbatchtargets);
    finaltestbatchdata=single(finaltestbatchdata);
    finaltestbatchtargets=single(finaltestbatchtargets);
    
end;

fprintf(1, 'nums per layer: ');
for l = 1:numlayers
    fprintf(1,' %4i ', layernums{l});
end;
fprintf(1, '\n');

fprintf(1, ' seed %4i  lambdamean %1.4f  minlevelsup %2i minlevelenergy %2i \n ', ... 
             myrandomseed, lambdamean, minlevelsup, minlevelenergy);
fprintf(1, '  labelstrength %2.2f temp %2.2f weightcost %1.5f supweightcost %1.5f \n ', ...
             labelstrength, temp,  wc, supwc);
fprintf(1, 'eps %1.5f epssup %1.5f delay  %1.4f \n ', ...
            epsilon,  epsilonsup, delay);
fprintf(1, 'maxepoch %5i numbatches %3i\n ', ...
            maxepoch, numbatches);

for epoch=epoch:maxepoch
    for l = 2:numlayers-1
        poserrs{l} = 0;
        negerrs{l} = 0; 
        pairsumerrs{l} = 0;
    end;
    trainlogcost = 0;    
 
    if epoch<=maxepoch/2
        epsgain = 1;
    else
        epsgain = (1 + 2*(maxepoch-epoch))/maxepoch;
    end;
    % The learning rate remains constant for the first half of the allotted epochs then declines linearly to zero.
    
    for batch = 1:numbatches
        data = batchdata(:,:,batch);
        targets = batchtargets(:,:,batch);
        data(:,1:10) = labelstrength*targets;
        normstates{1} = ffnormrows(data); 

        for l = 2:numlayers-1
            totin{l} = normstates{l-1}*weights{l} + biases{l};
            states{l} = max(0, totin{l});
            normstates{l} = ffnormrows(states{l});
            posprobs{l} = logistic( (sum(states{l}.^2,2) - layernums{l})/temp);
            dCbydin{l} = repmat(1 - posprobs{l}, 1, layernums{l}).*(states{l});
            %% wrong sign: rate at which it gets BETTER not worse. Think of C as goodness.
            meanstates{l} = 0.9*meanstates{l} + 0.1*mean(states{l});
            dCbydin{l} = dCbydin{l} + lambdamean*(mean(meanstates{l}) - meanstates{l});
            %% This is a regularizer that encourages the average activity of a unit to match that for all the units
            %%% in the layer. Notice that we do not gate by (states>0) for this extra term.
            %% This allows the extra term to revive units that are always off.  May not be needed.
            posdCbydweights{l} = normstates{l-1}'*dCbydin{l};
            posdCbydbiases{l} = sum(dCbydin{l});
        end;
        
        %%%% NOW WE GET THE HIDDEN STATES WHEN THE LABEL IS NEUTRAL AND USE THE NORMALIZED HIDDEN STATES AS INPUTS TO A
        %%%% SOFTMAX.  THIS SOFTMAX IS USED TO PICK HARD NEGATIVE LABELS
        
        data(:,1:10) = labelstrength*ones(numcases, numlab)/numlab;  % neutral label
        normstates{1} = ffnormrows(data); 
        for l = 2:numlayers-1
            totin{l} = normstates{l-1}*weights{l} + biases{l};
            states{l} = max(0, totin{l});
            normstates{l} = ffnormrows(max(0, totin{l}));
        end;
 
        labin = repmat(biases{numlayers}, numcases, 1);
        for l = minlevelsup:numlayers-1
            labin = labin + normstates{l}*supweightsfrom{l};
            % normstates seems to work better than states for predicting the label
        end;
        labin = labin - repmat(max(labin,[],2), 1, numlab);
        unnormlabprobs = exp(labin);
        trainpredictions = unnormlabprobs./repmat(sum(unnormlabprobs,2), 1, numlab);
        correctprobs = sum(trainpredictions.*targets,2); %a column vector 
        thistrainlogcosts = - log(tiny+correctprobs);
        trainlogcost = trainlogcost + sum(thistrainlogcosts)/numbatches;
        %% we print the log cost per batch (not per case).
        [score trainguesses] = max(trainpredictions,[],2);
        [tscore targetindices]=max(targets,[],2);
        trainerrors = sum(trainguesses ~= targetindices);
            
        dCbydin{numlayers} = targets - trainpredictions;
        dCbydbiases{numlayers} = sum(dCbydin{numlayers}, 1);
        for l = minlevelsup:numlayers-1
            dCbydsupweightsfrom{l} = normstates{l}'*dCbydin{numlayers};
            supweightsfromgrad{l} = delay*supweightsfromgrad{l} + (1-delay)*dCbydsupweightsfrom{l}/numcases;
            supweightsfrom{l} = supweightsfrom{l} + epsgain*epsilonsup*(supweightsfromgrad{l} - supwc*supweightsfrom{l});
        end;
        %% HACK: it works better without predicting the label from the first hidden layer.

        %% NOW WE MAKE NEGDATA
        negdata =  data; 
        labinothers = labin - 1000*targets; % add big negative logits for the targets so we do not choose them
        negdata(:, 1:10) = labelstrength*choosefrom(softmax(labinothers)); %% picks a bad label from the predicted distribution
        normstates{1} = ffnormrows(negdata); 
        
        for l = 2:numlayers-1
            totin{l} = normstates{l-1}*weights{l} + biases{l};
            states{l} = max(0, totin{l});
            normstates{l} = ffnormrows(states{l});
            negprobs{l} = logistic( (sum(states{l}.^2,2) - layernums{l})/temp);
            %% probability of saying a negative case is POSITIVE.                
            dCbydin{l} = repmat(- negprobs{l}, 1, layernums{l}).*(states{l});
            negdCbydweights{l} = normstates{l-1}'*dCbydin{l};
            negdCbydbiases{l} = sum(dCbydin{l});
            pairsumerrs{l} = pairsumerrs{l} + sum(negprobs{l}>posprobs{l});
        end;
        for l = 2:numlayers-1
            %% weightsgrad is the smoothed gradient for the weights
            weightsgrad{l} = delay*weightsgrad{l} + (1-delay)*(posdCbydweights{l} + negdCbydweights{l})/numcases;
            biasesgrad{l} = delay*biasesgrad{l} + (1-delay)*(posdCbydbiases{l} + negdCbydbiases{l})/numcases;
            biases{l} = biases{l} + epsgain*epsilon*biasesgrad{l};
            weights{l} = weights{l} + epsgain*epsilon*(weightsgrad{l} - wc*weights{l});
            % weights{l} = equicols(weights{l}); 
            % equicols makes the incoming weight vectors have the same L2 norm for all units in a layer.
        end;
    end; %%end of the for loop over batches

    if rem(epoch, printfreq)==0 
        fprintf(1, 'ep %3i gain %1.3f trainlogcost %3.4f PairwiseErrs: ', ...
                   epoch,    epsgain, trainlogcost) 
        for l = 2:numlayers-1
            fprintf(1,' %4i', pairsumerrs{l});
        end;
        fprintf(1, '\n');
    end;
   
   
   if rem(epoch, histfreq)==0
       figure(9); clf;
       for l=2:numlayers-1
           subplot(2, numlayers-2, l-1);
           hist(meanstates{l}, 100);
           title(l);
       end;
       for l=2:numlayers-1
           subplot(2, numlayers-2, numlayers-2 +l-1);
           hist(states{l}(:), 30);
           title(l);
       end;
       drawnow;
   end
     
   if rem(epoch, testfreq)==0
       % tests must come AFTER displaying the activity histograms for the last TRAINING batch.
       ffenergytest;
       %%fprintf(1,'\n');
       ffsoftmaxtest;
   end;

   if rem(epoch, rmsfreq)==0 
       fprintf(1, 'rms: ');
       for l = 2:numlayers-1
           fprintf(1,' %1.4f ', rms(weights{l}));
       end;
       fprintf(1, '\n');
       
       fprintf(1, 'suprms: ');
       for l = 2:numlayers-1
           fprintf(1,' %1.4f ', rms(supweightsfrom{l}));
       end;       
       fprintf(1, '\n');
       % the magnitudes of the sup weights show how much each hidden layer contributes to the softmax.
   end;
end; 

finaltest=1;
ffenergytest;
ffsoftmaxtest;

























