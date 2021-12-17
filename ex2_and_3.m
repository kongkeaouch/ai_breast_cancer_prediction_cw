clc
clear
close all

load cancer_dataset

[x,t] = cancer_dataset;

epoch = 8;
node = 8;

trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.
tind = vec2ind(t);

ensemble = 3:2:25;
ensemble_yind_iteration = zeros(3,699,30);
ensemble_iteration_test_errors = zeros(3,30);
ensemble_iteration_train_errors = zeros(3,30);

count = 0;

for e_i = 1:3
for i_i = 1:30
  
% Create a Pattern Recognition Network
hiddenLayerSize = epoch;
net = patternnet(hiddenLayerSize, trainFcn);
net.trainParam.epochs = node;
net.trainParam.min_grad = 10e-12;
% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivision
[trainInd,valInd,testInd] = dividerand(699,50/100,0/100,50/100);
net.divideFcn = 'divideind';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainInd = trainInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = testInd;

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'crossentropy';  % Cross-Entropy
%net.performFcn = 'classiferror';  % Classification-error

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotconfusion', 'plotroc'};

ensemble_yind = zeros(ensemble(e_i),699);

for c = 1:ensemble(e_i)

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
% e = gsubtract(t,y);
% performance = perform(net,t,y)
yind = vec2ind(y);
ensemble_yind(c,:) = yind;
% percentErrors = sum(tind ~= yind)/numel(tind);

% tr_testInd = tr.testInd;
% tr_trainInd = tr.trainInd;
% 
% epoch_node_iteration_test_error(e_i,e_i,i_i) = sum(tind(tr_testInd) ~= yind(tr_testInd))/numel(tind(tr_testInd));
% epoch_node_iteration_train_error(e_i,e_i,i_i) = sum(tind(tr_trainInd) ~= yind(tr_trainInd))/numel(tind(tr_trainInd));

% Recalculate Training, Validation and Test Performance
% trainTargets = t .* tr.trainMask{1};
% valTargets = t .* tr.valMask{1};
% testTargets = t .* tr.testMask{1};
% trainPerformance = perform(net,trainTargets,y)
% valPerformance = perform(net,valTargets,y)
% testPerformance = perform(net,testTargets,y)

% View the Network
% view(net)

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotconfusion(t,y)
%figure, plotroc(t,y)

% Deployment
% Change the (false) values to (true) to enable the following code blocks.
% See the help for each generation function for more information.
% if (false)
%     % Generate MATLAB function for neural network for application
%     % deployment in MATLAB scripts or with MATLAB Compiler and Builder
%     % tools, or simply to examine the calculations your trained neural
%     % network performs.
%     genFunction(net,'myNeuralNetworkFunction');
%     y = myNeuralNetworkFunction(x);
% end
% if (false)
%     % Generate a matrix-only MATLAB function for neural network code
%     % generation with MATLAB Coder tools.
%     genFunction(net,'myNeuralNetworkFunction','MatrixOnly','yes');
%     y = myNeuralNetworkFunction(x);
% end
% if (false)
%     % Generate a Simulink diagram for simulation or deployment with.
%     % Simulink Coder tools.
%     gensim(net);
% end

net = init(net);


count = count + 1;

disp([num2str(round(count*100/1230,2)) '%'])

end

for y_i = 1:699  
    if (sum(ensemble_yind(:,y_i) == 1) > sum(ensemble_yind(:,y_i) == 2))
        ensemble_yind_iteration(e_i,y_i,i_i) = 1;
    else
        ensemble_yind_iteration(e_i,y_i,i_i) = 2;
    end
end



ensemble_iteration_train_errors(e_i,i_i) = sum(tind(trainInd) ~= ensemble_yind_iteration(e_i,trainInd,i_i))/numel(tind(trainInd));
ensemble_iteration_test_errors(e_i,i_i) = sum(tind(testInd) ~= ensemble_yind_iteration(e_i,testInd,i_i))/numel(tind(testInd));

end
end

ensemble_train_errors = transpose(mean(ensemble_iteration_train_errors,2));
ensemble_test_errors = transpose(mean(ensemble_iteration_test_errors,2));

[M,I] = min(ensemble_test_errors);

disp('best ensemble classifier with')
disp(['node: ' num2str(node)])
disp(['epoch: ' num2str(epoch)])
disp(['trainFcn: ' trainFcn])
disp(['uses ' num2str(ensemble(I)) ' classifiers'])
disp(['give test error: ' num2str(M)])



