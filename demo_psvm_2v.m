clear
load ('ionosphere.mat');

X1=mapminmax(X1',0,1);
X2=mapminmax(X2',0,1);

data=X1';
data2=X2';
[M,N]=size(data);
g=2;c=1;d=1;gamma=0.1;
tic
indices=crossvalind('Kfold',data(1:M,N),5);
for k=1:5
    test = (indices == k);
    train = ~test;
    train_data=data(train,:);
    train_data2=data2(train,:);
    train_target=y(train,:);
    test_data=data(test,:);
    test_data2=data2(test,:);
    test_target=y(test,:);
    
    model=train_psvm_2v(train_data,train_data2,train_target,'rbf',c,c,d,g,gamma);
    [accuracy(k),accuracy1(k),accuracy2(k)]=predict_psvm_2v(model,test_data,test_data2,test_target);
    clear model;
end
accuracy21=mean(accuracy);
fprintf('psvm_2v %.4f\n',mean(accuracy21));
toc

