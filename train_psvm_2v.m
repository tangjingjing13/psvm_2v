function model=train_psvm_2v(x,x2,y,kerType,Ca,Cb,D,gamma,gamma4psvm)
    epsilon=0.001;
    model.kerType=kerType;
    model.gamma=gamma;
    [~,~] = size(x);
    [n,~] =size(x2);
    x=[x,ones(n,1)];
    x2=[x2,ones(n,1)];
    options = optimset;
    options.LargeScale = 'off';
    options.Display = 'off';
    temp1=kernel(x,x,kerType,gamma);
    temp2=(1/gamma4psvm)*kernel(x2,x2,kerType,gamma);
    
    tempy=ones(n,1)*y';
    A1=(y*y').*temp1;
    A2=tempy.*temp1;
    A3=(y*y').*temp2;
    A4=tempy.*temp2;    
    A5=temp1+temp2;
    A6=temp2;
    A7=temp1;
    
    H=[A1,zeros(n,n),-A2',A2',zeros(n,n),-A1;
        zeros(n,n),A3,A4',-A4',-A3,zeros(n,n);
        -A2,A4,A5,-A5',-A4,A2;
        A2,-A4,-A5,A5,A4,-A2;
        zeros(n,n),-A3,-A4',A4',A3,zeros(n,n);
        -A1,zeros(n,n),A2',-A2',zeros(n,n),A1];
    
    f=[-ones(2*n,1);epsilon*ones(2*n,1);zeros(2*n,1)];
    A = [eye(n),zeros(n,3*n),eye(n),zeros(n,n);
         zeros(n,n),eye(n),zeros(n,3*n),eye(n);
         zeros(n,2*n),eye(n),eye(n),zeros(n,2*n)];
    b = [Ca*ones(n,1);Cb*ones(n,1);D*ones(n,1)];
    
    lb = zeros(6*n,1);
    ub = [Ca*ones(n,1);Cb*ones(n,1);D*ones(2*n,1);Ca*ones(n,1);Cb*ones(n,1)];
    a0 = zeros(6*n,1);
    options.Algorithm='interior-point-convex';
    [a]  = quadprog(H,f,A,b,[],[],lb,[],a0,options);
    alpha_a=a(1:n);
    alpha_b=a(n+1:2*n);
    beta_a=a(2*n+1:3*n);
    beta_b=a(3*n+1:4*n);
    lamda_a=a(4*n+1:5*n);
    lamda_b=a(5*n+1:6*n);
    model.Wa=beta_b-beta_a+alpha_a.*y-lamda_b.*y;
    model.Wb=(1/gamma4psvm)*(beta_a-beta_b+alpha_b.*y-lamda_a.*y);
    model.x=x;
    model.x2=x2;
    model.y=y;
    model.gamma4psvm=gamma4psvm;
end