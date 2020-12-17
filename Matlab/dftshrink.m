function i_out = dftshrink(img_n,nv,N)

[rowso, colso] = size(img_n);
img_n = padarray(img_n, ceil(size(img_n)/N)*N-size(img_n),'post','symmetric');

i_out = zeros(size(img_n));
[rows, cols] = size(img_n);
ol = zeros(size(img_n));
stp = N/2;

for n = 1:stp:cols-N+1
    for m = 1:stp:rows-N+1
        
        fftblk_als = fft2(img_n(m:m+N-1,n:n+N-1));
        pwr_fft_als =(fftblk_als.*conj(fftblk_als))/(N*N);
                       
        prb_err = exp(-nv./pwr_fft_als);
        
        
        synthF = fftblk_als.*prb_err;
        sigFilt = real(ifft2(synthF));
        i_out(m:m+N-1,n:n+N-1) = i_out(m:m+N-1,n:n+N-1) + sigFilt;
        ol(m:m+N-1,n:n+N-1) = ol(m:m+N-1,n:n+N-1) + 1; 
    end
end

i_out = i_out./ol;
i_out = i_out(1:rowso,1:colso);