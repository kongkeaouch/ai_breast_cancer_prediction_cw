clear
close all
clc

load cancer_dataset

[x,t] = cancer_dataset;

epochs = [1 2 4 8 16 32 64];
nodes = [2 8 32];
classifiers = 3:2:25;
itrs = 30;
trainFcns = {'trainscg','trainlm','trainrp'};
performFcns = {'crossentropy','mse','crossentropy'};
c = 0;
tfInd = 3;

tind = vec2ind(t);
len_tind = length(t);
len_epochs = length(epochs);
len_nodes = length(nodes);
len_classifiers = length(classifiers);
trainFcn = trainFcns{tfInd};  
performFcn = performFcns{tfInd};
total_itrs = sum(classifiers)*itrs*len_nodes*len_epochs;

ind_train_errs = zeros(len_nodes,len_epochs,len_classifiers);
ind_test_errs = zeros(len_nodes,len_epochs,len_classifiers);

ens_train_errs = zeros(len_nodes,len_epochs,len_classifiers);
ens_test_errs = zeros(len_nodes,len_epochs,len_classifiers);

for nInd = 1:len_nodes
    
    net = patternnet(nodes(nInd), trainFcn);
    net.performFcn = performFcn;
    net.trainParam.min_grad = 0;
    net.input.processFcns = {'removeconstantrows','mapminmax'};
    net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
        'plotconfusion', 'plotroc'};
    
    for eInd = 1:len_epochs
        
        net.trainParam.epochs = epochs(eInd);
        
        for cInd = 1:len_classifiers
            itr_ind_train_errs = zeros(classifiers(cInd),itrs);
            itr_ind_test_errs = zeros(classifiers(cInd),itrs);
            
            itr_ens_train_errs = zeros(1,itrs);
            itr_ens_test_errs = zeros(1,itrs);
            
            for iInd = 1:itrs
                [trainInd,valInd,testInd] = dividerand(len_tind,0.5,0,0.5);
                net.divideFcn = 'divideind';  
                net.divideMode = 'sample';  
                net.divideParam.trainInd = trainInd;
                net.divideParam.valInd = valInd;
                net.divideParam.testInd = testInd;
                
                yinds = zeros(1,len_tind,classifiers(cInd));
                
                for i = 1:classifiers(cInd)
                    % Train the Network
                    [net,tr] = train(net,x,t);
                    % Test the Network
                    y = net(x);
                    
                    yind = vec2ind(y);
                    yinds(:,:,i) = yind;
                    
                    testPercenterrs = sum(tind(tr.testInd) ~= yind(tr.testInd))/numel(tind(tr.testInd));
                    trainPercenterrs = sum(tind(tr.trainInd) ~= yind(tr.trainInd))/numel(tind(tr.trainInd));
                    
                    itr_ind_train_errs(i,iInd) = trainPercenterrs;
                    itr_ind_test_errs(i,iInd) = testPercenterrs;
                    
                    net = init(net);
                    
                    c = c + 1;
                    progress = c/total_itrs*100;
                    disp([trainFcns{tfInd} ' e' num2str(epochs(eInd)) ' n' num2str(nodes(nInd)) ' i' num2str(iInd) ' ens' num2str(classifiers(cInd)) '_' num2str(i)])
                    disp([num2str(progress) ' ' num2str(c)])
                    
                end
                
                ens_yind = zeros(1,len_tind);
                limit = floor(classifiers(cInd)/2);
                
                for i = 1:len_tind
                    isTrue = sum(yinds(:,i,:) == 1) > limit;
                    
                    if (isTrue)
                        ens_yind(i) = 1;
                    else
                        ens_yind(i) = 2;
                    end
                end
                
                itr_ens_train_errs(iInd) = sum(tind(tr.trainInd) ~= ens_yind(tr.trainInd))/numel(tind(tr.trainInd));
                itr_ens_test_errs(iInd) = sum(tind(tr.testInd) ~= ens_yind(tr.testInd))/numel(tind(tr.testInd));
                
                
            end
            
            ens_train_errs(nInd,eInd,cInd) = mean(itr_ens_train_errs);
            ens_test_errs(nInd,eInd,cInd) = mean(itr_ens_test_errs);
            
            ind_train_errs(nInd,eInd,cInd) = mean(itr_ind_train_errs,[1 2]);
            ind_test_errs(nInd,eInd,cInd) = mean(itr_ind_test_errs,[1 2]);
            
            
        end
        
    end
    
end



%
% legend_labels = cellstr(append(string(classifiers), ' classifiers'));
%
% for tfInd = 1:len_trainFcns
%     for cInd = 1:len_classifiers
%
%         plot(epochs,ens_test_errs(:,:,cInd))
%         title(['Test err rates - ' num2str(epochs(eInd)) ' classifiers - ' trainFcns(tfInd)])
%         legend(legend_labels)
%         saveas(gcf,['Test_err_rates_' num2str(classifiers(cInd)) '_classifiers_' trainFcns(tfInd) '.png'])
%
%         plot(epochs,ens_train_errs(:,:,cInd))
%         title(['Train err rates - ' num2str(classifiers(cInd)) ' classifiers - ' trainFcns(tfInd)])
%         legend(legend_labels)
%         saveas(gcf,['Train_err_rates_' num2str(classifiers(cInd)) '_classifiers_' trainFcns(tfInd) '.png'])
%
%     end
% end


