function capacity = fitness_SIM_MIMO(theta)

%% THAM SO HE THONG
Nt = 4; Nr = 4;       % so anten
M  = length(theta);   % so phan tu metasurface
SNR = 10;             % SNR (dB)

%% MA TRAN KENH NGAU NHIEN
H_t = (randn(M,Nt)+1j*randn(M,Nt))/sqrt(2);
H_r = (randn(Nr,M)+1j*randn(Nr,M))/sqrt(2);

%% MA TRAN PHA METASURFACE
Phi = diag(exp(1j*theta));

%% KENH HIEU DUNG
H_eff = H_r * Phi * H_t;

%% TINH CAPACITY
snr_lin = 10^(SNR/10);
capacity = real(log2(det( eye(Nr) + (snr_lin/Nt)*(H_eff*H_eff') )));

end
