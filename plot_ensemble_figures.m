epoch = [1 2 4 8 16 32 64];
node = [2 8 32];

ensemble = 3:2:25;

trainFcn = 'trainrp';



% figure(1)
% 
% y = reshape(ens_test_errors(2,4,:),[1 12]);
% 
% bar([0.039 y])
% b.FaceColor = 'flat';
% b.CData(1,:) = 222;
% set(gca, 'XTickLabel',{'Base','3','5','7','9','11','13','15','17','19','21','23','25'})
% 
% title('Ensemble test error rates of 8/8 combination - trainscg')
% saveas(gcf,'ensemble_test_error_8_8.png')

best_loss = zeros(3,12);


for eInd = 1:length(ensemble)
    [M,I] = min(ens_test_errors(:,:,eInd),[],'all','linear');
    
    best_loss(1,eInd) = M;
    
    [a,b] = find(ens_test_errors(:,:,eInd) == M)
    best_loss(2,eInd) = node(a);
    best_loss(3,eInd) = epoch(b);
    

% disp('best ensemble classifier with')
% disp(['node: ' num2str(node)])
% disp(['epoch: ' num2str(epoch)])
% disp(['trainFcn: ' trainFcn])
% disp(['uses ' num2str(ensemble(I)) ' classifiers'])
% disp(['give test error: ' num2str(M)])
%     figure(1)
%     
%     plot(epochs, ens_test_errors(:,:,eInd))
%     title(['Ensemble of ' num2str(ensemble(eInd)) ' test error rates - ' trainFcn])
%     legend('2 nodes','8 nodes','32 nodes')
%     saveas(gcf,[trainFcn '_ensemble_test_' num2str(ensemble(eInd)) '.png'])
%     
%     figure(2)
%     
%     plot(epochs, ens_train_errors(:,:,eInd))
%     title(['Ensemble of ' num2str(ensemble(eInd)) ' train error rates - ' trainFcn])
%     legend('2 nodes','8 nodes','32 nodes')
%     saveas(gcf,[trainFcn '_ensemble_train_' num2str(ensemble(eInd)) '.png'])
end

