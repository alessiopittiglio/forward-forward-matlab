% minlevelenergy now set externally to this script
% the finaltest flag is set externally.

numtrainbatches=size(validbatchdata,3); 
% use same number of batches as validation data. Assumes validation data is not bigger than training data.
% Also assumes valid data is same size as testdata.

numcases = size(batchdata,1);
trainsumerrors=0;
logcost = 0;

for batch = 1:numtrainbatches
     %% we do not use all the training data to estimate the training error
     data =  batchdata(:,:,batch);
     targets= batchtargets(:,:,batch);
     actsumsq = zeros(numcases, numlab);
     for lab = 1:10
         data(:,1:10) = zeros(numcases, numlab);
         data(:,lab) = labelstrength*ones(numcases,1);
         normstates{1} = ffnormrows(data); 
         for l = 2:numlayers-1
             states{l} = max(0, normstates{l-1}*weights{l} + biases{l});
             if l>=minlevelenergy
                 actsumsq(:,lab) = actsumsq(:,lab) + sum(states{l}.^2,2);
             end;
             normstates{l} = ffnormrows(states{l});
         end;
     end;

     [score guesses] = max(actsumsq,[],2);
     [tscore targetindices]=max(targets,[],2);

     errors = sum(guesses ~= targetindices);
     trainsumerrors =   trainsumerrors+errors;
end; %%end of the for loop over batches

testsumerrors=0;
testlogcost = 0;

if finaltest==1
    testbatchdata =  finaltestbatchdata;
    testbatchtargets= finaltestbatchtargets;
else
    testbatchdata =  validbatchdata;
    testbatchtargets= validbatchtargets;
end;        

numtestbatches=size(testbatchdata,3);

for batch = 1:numtestbatches
    testdata =  testbatchdata(:,:,batch);
    testtargets= testbatchtargets(:,:,batch);
    actsumsq = zeros(numcases, numlab);
    for lab = 1:10
        testdata(:,1:10) = zeros(numcases, numlab);
        testdata(:,lab) = labelstrength*ones(numcases,1);
        normstates{1} = ffnormrows(testdata);
        for l = 2:numlayers-1
            states{l} = max(0, normstates{l-1}*weights{l} + biases{l});
            if l>=minlevelenergy
                actsumsq(:,lab) = actsumsq(:,lab) + sum(states{l}.^2,2);
            end;
            normstates{l} = ffnormrows(states{l});
        end;
    end;
    
    [score testguesses] = max(actsumsq,[],2);
    [tscore targetindices]=max(testtargets,[],2);
    
    testerrors = sum(testguesses ~= targetindices);
    testsumerrors =   testsumerrors+testerrors;
end; % end of the for loop over test batches
    
if finaltest==1
    fprintf(1,  'Energy-based errs: Train %4i Test %4i  ', ... 
               trainsumerrors, testsumerrors);
else
    fprintf(1,  'Energy-based errs: Train %4i Valid %4i  ', ... 
               trainsumerrors, testsumerrors);
end;

       
       









