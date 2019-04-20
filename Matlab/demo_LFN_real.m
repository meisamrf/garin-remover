clc
close all
clear

disp("BMCNN can be replaced by any AWGN denoiser")

model_weights = bmcnn_open('../models/bmcnn_16.bin');
if model_weights(1)==0
	disp('could not find the model');
    return
end

img_list = {'../dataset/JFK.png','../dataset/fight_club.jpg'};

for img_num = 1:2
    disp(['------ Image num ' num2str(img_num) '------'])
    Irgb = imread(char(img_list(img_num)));
    Iycbcr = rgb2ycbcr(Irgb);
    I = single(Iycbcr(:,:,1));
    sigma_a = 0.6;
    l = 1;
    h = fspecial('gaussian', 3, sigma_a);
    N = 8;
    
    %------------  noise variance calculation ----------------------
    sigma_w = 4; %Variance of noise at l=1
    sigma_n = 8; %Variance of noise at l=0
    %----------------Proposed Framework------------------
    tic;
    d = 2^l;
    [rowso, colso] = size(I);
    I = padarray(I, ceil(size(I)/N)*N-size(I),'post','symmetric');
    I_h = imfilter(I,h,'symmetric');
    A1 = I_h(1:d:end,1:d:end);
    
    %[~, A1_f] = BM3D(1, double(A1), double(sigma_w));A1_f = A1_f*255;
    A1_f = bmcnn_denoiser(single(A1), model_weights, single(sigma_w));

    %*********** Proposed LF denoiser ****************
    I_F = LFdenoiser(A1_f, I_h, I, single(sigma_w));
    I_F = I_F(1:rowso,1:colso);I = I(1:rowso,1:colso);
    clear I_h A1;
    el = toc;
    disp(['elapsed time of proposed is ' num2str(el,'%2.3f') ' seconds'])

    %---------------BM3D all pass-------------------------
    tic;
    %[~, y_bm3d] = BM3D(1, double(I), double(sigma_n));
    y_allpass = bmcnn_denoiser(single(I), model_weights, single(sigma_n));
    el = toc;
    disp(['elapsed time of all-pass is ' num2str(el,'%2.3f') ' seconds'])

    %----------------------------------------
    
    figure('units','normalized','outerposition',[0 0 1 1])
    subplot(2,2,1);
    
    Iycbcr_f = Iycbcr;
    Iycbcr_f(:,:,1) =  uint8(I);Irgb_f = ycbcr2rgb(Iycbcr_f);
    imshow(Irgb_f);title('Noisy (spatially correlated)')
    subplot(2,2,2);
    Iycbcr_f(:,:,1) =  uint8(y_allpass);Irgb_f = ycbcr2rgb(Iycbcr_f);
    imshow(Irgb_f);title('Denoised by BM3D (all-pass)')
    subplot(2,2,[3 4]);
    Iycbcr_f(:,:,1) =  uint8(I_F);Irgb_f = ycbcr2rgb(Iycbcr_f);
    imshow(Irgb_f);title('Denoised by Proposed, BM3D (high-pass)')
end

function imgo = bmcnn_denoiser(imgn, weights, sigma_n)
    base_sigma = 15;    
    imgn = single(imgn*base_sigma/sigma_n/255);
    imgo = bmcnn_predict(imgn, weights);
    imgo = imgo/(base_sigma/sigma_n/255);
end
