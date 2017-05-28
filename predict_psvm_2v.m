function [accuracy,accuracy1,accuracy2]=predict_psvm_2v(model,x,x2,y)
[n,~] = size(x);
x=[x,ones(n,1)];
x2=[x2,ones(n,1)];
label=kernel(x,model.x,model.kerType,model.gamma)*model.Wa+kernel(x2,model.x2,model.kerType,model.gamma)*model.Wb;
label=label/2;

label1=kernel(x,model.x,model.kerType,model.gamma)*model.Wa;
label2=kernel(x2,model.x2,model.kerType,model.gamma)*model.Wb;

for i=1:n
    if label(i)>=0
        label(i)=1;
    else
        label(i)=-1;
    end
end

accuracy_2v=length(find(y-label==0))/n;

for i=1:n
    if label1(i)>=0
        label1(i)=1;
    else
        label1(i)=-1;
    end
end

accuracy1=length(find(y-label1==0))/n;

for i=1:n
    
    if label2(i)>=0
        label2(i)=1;
    else
        label2(i)=-1;
    end
end

accuracy2=length(find(y-label2==0))/n;

accuracy=max([accuracy1,accuracy2,accuracy_2v]);
end