close all
clear
clc

mt_1=[];
mt_2=[];
mt_21=[];
mt_22=[];
mt_23=[];
mt_3=[];
mt_4=[];
mt_41=[];
mt_5=[];
mt_6=[];
mt_7=[];

mRMSE_1 = [];
mRMSE_2 = [];
mRMSE_21 = [];
mRMSE_22 = [];
mRMSE_23 = [];
mRMSE_3 = [];
mRMSE_4 = [];
mRMSE_41 = [];
mRMSE_5 = [];
mRMSE_6 = [];
mRMSE_7 = [];


for iii = 1:1:4
t_1=[];
t_2=[];
t_21=[];
t_22=[];
t_23=[];
t_3=[];
t_4=[];
t_41=[];
t_5=[];
t_6=[];
t_7=[];
t_8=[];
t_9=[];
t_10=[];
t_20=[];
RMSE = [];
RMSE_1 = [];
RMSE_2 = [];
RMSE_21 = [];
RMSE_22 = [];
RMSE_23 = [];
RMSE_3 = [];
RMSE_4 = [];
RMSE_41 = [];
RMSE_5 = [];
RMSE_6 = [];
RMSE_7 = [];
RMSE_8 = [];
RMSE_9 = [];
RMSE_10 = [];
RMSE_20 = [];
RMSE_11 = [];
RMSE_12 = [];

maxiter = 100;
% iii = 1;
m = 300*iii;
n = 200*iii;
rak = 5*iii;
    
 for kk = 1:10
%% Synthetic_data

M = randn(m,rak)*randn(rak,n);
SNR = 10;
% M_1 = awgn(M,20,'measured');
noise = Gaussian_noise(M,'GM',SNR);
D = (M+noise);

% add missing entries
per = 0.5;
array_Omega = binornd( 1, per, [ m, n ] );
M_noise = D.* array_Omega;

%% RMC-HOW
ip = 2;
ieta = ip*sqrt(2);
tic
[X_1, MSE_1, loss_HOW] = RMC_HOW(M,M_noise,array_Omega,rak,maxiter,ip,ieta);
toc
t_1 = [t_1 toc];
error2=norm((M-X_1),'fro')/sqrt(m*n);
RMSE_1 = [RMSE_1 error2];


%% RMC-HOC
tic
[X_2, MSE_2, loss_HOC] = RMC_HOC(M,M_noise,array_Omega,rak,maxiter,ip);
toc
t_2 = [t_2 toc];
error2=norm((M-X_2),'fro')/sqrt(m*n);
RMSE_2 = [RMSE_2 error2];

%% RMC-HOP
p = 1;
tic
[X_21, MSE_21, loss_HOP1] = RMC_HOP(M,M_noise,array_Omega,rak,maxiter,ip,p);
toc
t_21 = [t_21 toc];
error2=norm((M-X_21),'fro')/sqrt(m*n);
RMSE_21 = [RMSE_21 error2];

p = 0.6;
tic
[X_22, MSE_22, loss_HOP2] = RMC_HOP(M,M_noise,array_Omega,rak,maxiter,ip,p);
toc
t_22 = [t_22 toc];
error2=norm((M-X_22),'fro')/sqrt(m*n);
RMSE_22 = [RMSE_22 error2];

p = 0.3;
tic
[X_23, MSE_23, loss_HOP3] = RMC_HOP(M,M_noise,array_Omega,rak,maxiter,ip,p);
toc
t_23 = [t_23 toc];
error2=norm((M-X_23),'fro')/sqrt(m*n);
RMSE_23 = [RMSE_23 error2];

 end

end


