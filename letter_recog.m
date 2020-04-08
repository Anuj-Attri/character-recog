function [trainedClassifier percentAccurate]=trainedClassifier(data)
%training data set%
data=readtable("letter-recognition.csv");


predictorNames = {'xbox', 'ybox', 'width', 'height', 'onpix', 'xbar', 'ybar', 'x2bar', 'y2bar', 'xybar', 'x2ybar', 'xy2bar', 'xedge', 'xedgey', 'yedge', 'yedgex'};
predictors=data(:,predictorNames);
responsevar=data.letter;

knnmdl=fitcknn(predictors,responsevar,"Standardize",true);

predictorExtractionFcn =@(t) t(:, predictorNames);
knnPredictFcn = @(x) predict(knnmdl, x);
trainedClassifier.predictFcn = @(x) knnPredictFcn(predictorExtractionFcn(x));

trainedClassifier.RequiredVariables = {'height', 'onpix', 'width', 'x2bar', 'x2ybar', 'xbar', 'xbox', 'xedge', 'xedgey', 'xy2bar', 'xybar', 'y2bar', 'ybar', 'ybox', 'yedge', 'yedgex'};
trainedClassifier.knnmdl = knnmdl;

%data set to be tested%
data=readtable("letter-recognition.csv");


predictorNames = {'xbox', 'ybox', 'width', 'height', 'onpix', 'xbar', 'ybar', 'x2bar', 'y2bar', 'xybar', 'x2ybar', 'xy2bar', 'xedge', 'xedgey', 'yedge', 'yedgex'};
predictors=data(:,predictorNames);
responsevar=data.letter;

partmdl=crossval(trainedClassifier.knnmdl,'Kfold',5);
[validPred score]=kfoldPredict(partmdl);
 
misclasserror=(categorical(responsevar)~=categorical(validPred));
percentAccurate=1-[sum(misclasserror)/numel(misclasserror)];
percentAccurate=percentAccurate*100
end