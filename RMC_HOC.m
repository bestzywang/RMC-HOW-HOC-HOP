function [Out_X, RMSE,loss_HOW, rel_1,S,RMSE_Omega,U,V] = RMC_HOC(M,M_Omega,Omega_array,rak,maxiter,ip)
% This matlab code implements the RMC-HOW
% method for Robust Matrix Completion.
%
% M_Omega - m x n observed matrix 
% Omega_array is the subset
% rak is the rank of the object matrix
% maxiter - maximum number of iterations

f_hoc=@(x,c,gamma) x.^2/2.*(abs(x)<=c)+((gamma^2+c^2)/2*log(1+(x/gamma).^2)+c^2/2-(gamma^2+c^2)/2*log(1+(c/gamma).^2)).*(abs(x)>c);
% Pro = @(T,c,gamma) T.*(1-(gamma^2+c^2)./(gamma^2+T.^2));
Pro = @(x,LAMB,sigm) 0.*(abs(x)<=LAMB)+((abs(x)-(sigm^2+LAMB^2)*abs(x)./(sigm^2+x.^2)).*sign(x)).*(abs(x)>LAMB);
loss_HOW = [];
[m,n] = size(M_Omega);
RMSE = [];
RMSE_Omega = [];
%sum_Omega = sum(Omega_array(:));
S = 0;
scale =100;
rel_1 = [];
L_f_1 = 0;
th = 10^(-4);
loc = 1000;

% Initializing U and V
U =randn(m,rak);
V =randn(rak,n);
X_1 = U*V;


    
for k = 1:100
     D = (M_Omega - S);   
%     D = (M_Omega - S).*Omega_array;
    dU = -(D-Omega_array.*(U*V))*V';
    du=-dU*((V*V')^(-1));
    tu=-trace(dU'*du)/(norm(Omega_array.*(du*V),'fro'))^2;
    U=U+tu*du;
        
    dV=-U'*((D-Omega_array.*(U*V)));
    dv=(U'*U)^(-1)*dV;
    tv=-trace(dV'*dv)/(norm(Omega_array.*(U*dv),'fro'))^2;
    V=V+tv*dv;
    X=U*V;
   
    X_err = norm(X-X_1,'fro')^2/norm(X_1,'fro')^2;
    X_1 = X;
%     RMSE= [RMSE norm(M-X,'fro')/sqrt(m*n)];
%     RMSE_Omega= [RMSE_Omega norm((M-X).*Omega_array,'fro')/(sum_Omega)];
    loss_HOW = [loss_HOW sum(sum(norm(M_Omega - X.*Omega_array,'fro')^2))];
    if X_err<10^(-3) 
        break
    end
end
    
for iter = 1 : maxiter        
        T = M_Omega - X.*Omega_array; 
        t_m_n = T(find(T));
        loc_1 = 0.7413*(quantile(t_m_n,0.75)-quantile(t_m_n,0.25));
%         loc_1 = 1.4826*median(abs(t_m_n - median(t_m_n)));
        loc = min(loc,loc_1);
        c = ip*loc;
        gamma = c;
        S = Pro(T,c,gamma);
        RMSE= [RMSE norm(M-X,'fro')/sqrt(m*n)];
        
        
     D = (M_Omega - S);   
    dU = -(D-Omega_array.*(U*V))*V';
    du=-dU*((V*V')^(-1));
    tu=-trace(dU'*du)/(norm(Omega_array.*(du*V),'fro'))^2;
    U=U+tu*du;
        
    dV=-U'*((D-Omega_array.*(U*V)));
    dv=(U'*U)^(-1)*dV;
    tv=-trace(dV'*dv)/(norm(Omega_array.*(U*dv),'fro'))^2;
    V=V+tv*dv;
    X=U*V;
        
loss_HOW = [loss_HOW sum(sum(f_hoc(M_Omega - X.*Omega_array,c,gamma)))];
if abs(loss_HOW(end)-loss_HOW(end-1))/loss_HOW(end-1)<0.0001
    break
end
end
    Out_X = X;
end
