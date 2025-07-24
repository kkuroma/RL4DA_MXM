clc;clear;
randn('state',0);

% time variable conventions
% ...        ...        ...
%  +          +          +    . . . 
% t_0        t_1         t_2  
%
% + : analysis times
% . : observation times
%
% dtda = t_1 - t_0 : time between analyses
% odt = t_1 - t_0 : time between observations

ensemble_size =20;
dtda   = 1.0;
odt    = 0.01;
nassim = 100;
H      = eye(3);
R      = eye(3); % the distribution of obs' is given

% control parameter settings for L63
par = [10 28 8/3];

% allocate array
Xt = zeros(3,nassim*(dtda/odt)+1);
Yo = zeros(3,nassim);

% ********************** initialize *************************
x = [10 20 30]; % set an arbitrary initial condition
[tims, states] = ode45(@derivsL63,[0 100],x,[],par);
xt = states(end,:)';
for l = 1:ensemble_size
    xa(:,l) = xt + 1.0*randn(3,1); %introduce perturbing to create Xb
end

% ********************* assimilation ************************
for k = 1:nassim
    [~, states] = ode45(@derivsL63,[0 dtda],xt,[],par);
    xt = states(end,:)';
    
    % create obs; assume that yo is unbiased E(yo)=xt
    yo = H*xt + sqrtm(R)*randn(3,1);
    ypert = sqrtm(R)*randn(3,20);
    ypert_mean = repmat(mean(ypert,2),1,20);
    ypert = ypert - ypert_mean; % xpert is ubiased
    for l = 1:ensemble_size 
       %Y(:,l) = yo + ypert(:,l); % resampling Y is unbiased
       Y(:,l) = yo; % no resampling
    end

    for l = 1:ensemble_size 
        [~, bstates] = ode45(@derivsL63,[0 dtda],xa(:,l),[],par);
        xb(:,l) = bstates(end,:)';
    end
    
    xb_mean = mean(xb,1); 
    Hx_mean = mean(H*xb,1);
    nsample = size(xb,2);
    
    PHt = 0;
    for l = 1:ensemble_size 
        PHt = PHt + (xb(:,l)-xb_mean)*(H*xb(:,l)-Hx_mean)';
    end
    PHt = PHt/(nsample-1);
    
    HpfHt = 0;
    for l = 1:ensemble_size 
        HpfHt = HpfHt + (H*xb(:,l)-Hx_mean)*(H*xb(:,l)-Hx_mean)';
    end
    HpfHt = HpfHt/(nsample-1);
        
    K = PHt * (HpfHt + R)^(-1);
    
    xa = xb + K*(Y-H*xb);
   
    %error statistics
    xbe(:,k) = mean(xb,2) - xt;
    xae(:,k) = mean(xa,2) - xt;
    %xae(:,k) = xa(:,2) - xt;
        
    xspread(:,k) = std(xa,0,2);

end    

xbe_var = var(xbe,0,2);
xae_var = var(xae,0,2);
xbe_mean_rmse = mean(sqrt(mean(xbe.^2,1)));
xae_mean_rmse = mean(sqrt(mean(xae.^2,1)));

bias = abs(xae);

datime = 1:100;
%draw
figure(1)
plot(datime,abs(xbe(1,:)),datime,abs(xae(1,:)),'LineWidth',2.5);
%semilogy(abs(xbe(1,:)),'b-'); hold on; semilogy(abs(xae(1,:)),'r-')
legend('background','analysis');
xlabel('Assimilation Step','fontweight','bold','fontsize',12);
ylabel('Absolute Error','fontweight','bold','fontsize',12); 
title('enkf Absolute Error','fontweight','bold','fontsize',12);
%set(gca,'linewidth',2,'fontweight','bold','fontsize',10)

figure(2)
plot(datime,bias(1,:),datime,xspread(1,:),'LineWidth',2.5);
%semilogy(abs(xbe(1,:)),'b-'); hold on; semilogy(abs(xae(1,:)),'r-')
legend('bias','spread');
xlabel('Assimilation Step','fontweight','bold','fontsize',12);
ylabel('Absolute Error','fontweight','bold','fontsize',12); 
title('enkf Bias & Spread','fontweight','bold','fontsize',12);
%set(gca,'linewidth',2,'fontweight','bold','fontsize',10)
