function [postchoiceprobs] = choosefrom(probs);

[numcases, numlab] = size(probs);

postchoiceprobs=zeros(size(probs));
for n=1:numcases
  r=rand(1,1);
  used=0;
  sumsofar=0;
  for lab=1:numlab
    sumsofar= sumsofar+probs(n,lab);
    if r<sumsofar & used==0
      used=1;
      postchoiceprobs(n,lab)=1;
      break;
    end;
  end;
end;
