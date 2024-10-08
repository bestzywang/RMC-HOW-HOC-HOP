function [noise] = Gaussian_noise(Signal,model,SNR)
% SNR (dB)
Signal_size = size(Signal);
num = 1;
for i = 1 : length(Signal_size)
    num = num * Signal_size(i);
end
Signal_power = sum(abs(Signal(:)).^2) / num;
s_n = 10^(SNR/10);


switch model
    case 'GW'
        Noise_power = Signal_power/s_n;
        noise = sqrt(Noise_power) * randn( Signal_size);
        
    case 'GM'
        c_1 = 0.9;
        c_2 = 0.1;
        h = 10; % the power of outliers over that of Gaussian noise is h.
        sigma_v_2 = Signal_power/ s_n ; % sigma_v_2 = c_1 * sigma_1^2 + c_2 * sigma_2^2 = 0.9*sigma_1^2 + 0.1*sigma_2^2; sigma_2 = 10 * sigma_1
        sigma_1 = sqrt(sigma_v_2 / (c_1 + h^2*c_2));
        p = c_1/1;        % the percentage of outliers c_1/c_2
        sigma = [ h*sigma_1 sigma_1 ];
        flag = binornd( 1, p, Signal_size );
        noise = sigma(1) * randn( Signal_size).*(ones(Signal_size) - flag) + sigma(2)*randn( Signal_size).*flag;
end
end