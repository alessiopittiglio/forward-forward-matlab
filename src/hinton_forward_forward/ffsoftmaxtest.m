
if finaltest==1
    testbatchdata =  finaltestbatchdata;
    testbatchtargets= finaltestbatchtargets;
else
    testbatchdata =  validbatchdata;
    testbatchtargets= validbatchtargets;
end;        

numtestbatches=size(testbatchdata,3);
numtestcases = size(testbatchdata,1);

testsumerrors=0;
testlogcost = 0;

for batch = 1:numtestbatches
    testdata =  testbatchdata(:,:,batch);
    testdata(:, 1:10) = labelstrength*ones(numtestcases, 10)/10;
    normstates{1} = ffnormrows(testdata);
    targets= testbatchtargets(:,:,batch);
    for l = 2:numlayers-1
        states{l} = max(0, normstates{l-1}*weights{l} + biases{l});
        normstates{l} = ffnormrows(states{l});
    end;
    labin = repmat(biases{numlayers}, numcases, 1);
    for l = minlevelsup:numlayers-1
        labin = labin + normstates{l}*supweightsfrom{l};
        %labin = labin + states{l}*supweightsfrom{l};
    end;
    labin = labin - repmat(max(labin,[],2), 1, numlab);
    unnormlabprobs = exp(labin);
    
    testpredictions = unnormlabprobs./repmat(sum(unnormlabprobs,2), 1, numlab);
    %% correctprobs=sum(testpredictions.*targets,2);
    thistestlogcosts =  - sum(sum(targets.*(log(tiny+testpredictions))));
    testlogcost = testlogcost + sum(thistestlogcosts)/numtestbatches;
    % we report the log cost per batch (not per case).
    [score testguesses] = max(testpredictions,[],2);
    [tscore targetindices]=max(targets,[],2);
    
    testerrors = sum(testguesses ~= targetindices);
    testsumerrors=   testsumerrors+testerrors;
end; %%end of the for loop over batches
    
if finaltest==1
    fprintf(1, 'Softmax test errs %4i  \n', testsumerrors);
else
    fprintf(1, 'Softmax valid errs %4i  \n', testsumerrors);
end












