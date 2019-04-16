function I_F = LFdenoiser(A1_f, I_h , I, sigma_w)

z = I_h - bilineartwo(single(A1_f));
z_hat = fwddecouple(z);
z_hat_s = dftshrink(z_hat,single(sigma_w*sigma_w*2));
p = max(abs(z_hat_s)/(sigma_w*2/3)-1,0);
p_2 = p.*p;
z_tilde_s = (p_2./(1+p_2)).*z_hat_s;
I_F = invdecouple(z_tilde_s)-z+I;

function imgo = invdecouple(img)

imgo = zeros(size(img));
imgo(1:2:end,1:2:end) = img(1:end/2,1:end/2);
imgo(1:2:end,2:2:end) = fliplr(img(1:end/2,end/2+1:end));
imgo(2:2:end,1:2:end) = flipud(img(end/2+1:end,1:end/2));
imgo(2:2:end,2:2:end) = rot90(img(end/2+1:end,end/2+1:end),2);

function imgo = fwddecouple(img)

imgo = [img(1:2:end,1:2:end) fliplr(img(1:2:end,2:2:end));...
    flipud([img(2:2:end,1:2:end) fliplr(img(2:2:end,2:2:end))])];
