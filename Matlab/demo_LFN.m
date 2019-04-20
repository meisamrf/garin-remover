clc
close all
clear

disp("BMCNN can be replaced by any AWGN denoiser")

model_weights = bmcnn_open('../models/bmcnn_16.bin');
if model_weights(1)==0
	disp('could not find the model');
    return
end
    
I = single(imread('../dataset/I04_noisy.png'));
sigma_a = 0.6;
l = 1;
I_o = double(imread('../dataset/I04.png'));
h = fspecial('gaussian', 3, sigma_a);
N = 8;

%------------  noise variance calculation ----------------------
noise = I_o-I;
noise_f = imfilter(noise,h,'symmetric');
sigma_w = std(noise_f(:)); %Variance of noise at l=1
sigma_n = std(noise(:)); %Variance of noise at l=0
%----------------Proposed Framework------------------
I = padarray(I, ceil(size(I)/N)*N-size(I),'post','symmetric');

tic;
d = 2^l;
I_h = imfilter(I,h,'symmetric');
A1 = I_h(1:d:end,1:d:end);

%[~, A1_f] = BM3D(1, double(A1), double(sigma_w));A1_f = A1_f*255;
A1_f = bmcnn_denoiser(single(A1), model_weights, single(sigma_w));
%*********** Proposed LF denoiser ****************
I_F = LFdenoiser(A1_f, I_h, I, sigma_w);
el = toc;
disp(['elapsed time of proposed is ' num2str(el,'%2.3f') ' seconds'])
%---------------denoiser all pass-------------------------
tic;
%[~, y_bm3d] = BM3D(1, double(I), double(sigma_n));
y_allpass = bmcnn_denoiser(single(I), model_weights, single(sigma_n));
el = toc;
disp(['elapsed time of all-pass is ' num2str(el,'%2.3f') ' seconds'])

%----------------------------------------

figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,2,1);
imshow(I/255);title('Noisy (spatially correlated)')
subplot(2,2,2);
imshow(y_allpass/255);title('Denoised by BMCNN (all-pass)')
subplot(2,2,[3 4]);
imshow(I_F/255);title('Denoised by Proposed, BMCNN (high-pass)')

function imgo = bmcnn_denoiser(imgn, weights, sigma_n)
    base_sigma = 15;    
    imgn = single(imgn*base_sigma/sigma_n/255);
    imgo = bmcnn_predict(imgn, weights);
    imgo = imgo/(base_sigma/sigma_n/255);
end

