import lfnfilter
import numpy as np
import skimage.filters


def downsample(I, sigma_a, d):
    I_h = skimage.filters.gaussian(I/255.0, sigma_a, mode='reflect')
    A1 = I_h[0::d,0::d]
    return np.float32(I_h*255.0), np.float32(A1*255.0)

def fwddecouple(img):

    t = np.concatenate((img[0::2,0::2], np.fliplr(img[0::2,1::2])), axis = 1)
    b = np.concatenate((img[1::2,0::2], np.fliplr(img[1::2,1::2])), axis = 1)
    imgo = np.concatenate((t,np.flipud(b)), axis = 0)
    return imgo

def invdecouple(img):
    imgo = np.zeros_like(img)
    c,r =  img.shape[1]//2, img.shape[0]//2
    imgo[0::2,0::2] = img[0:r,0:c]
    imgo[0::2,1::2] = np.fliplr(img[0:r,c:])
    imgo[1::2,0::2] = np.flipud(img[r:,0:c])
    imgo[1::2,1::2] = np.rot90(img[r:,c:], 2)
    return imgo

def LFdenoiser(I, A1_f, I_h, sigma_w):
    '''
    =====================================================================
    Low-Frequency Image Noise Removal Using White Noise Filter
    Written by Meisam Rakhshnfar, Vidpro Lab
    (https://users.encs.concordia.ca/~amer/LFNFilter/)
    Revised: April 2019
    ===================================================================== 
    LFdenoiser(I,  I_h, A1_f, sigma_w)

    INPUTS:
    I: 2D grayscale noisy image 
    A1_f: I down-sampled and denoised by an AWGN filter.
    I_h: filtered by a lowpass filter.
    sigma_w: standard deviation of grain noise 

    OUTPUTS:
    I_F: 2D grayscale denoised output
    '''

    if (I.ndim!=2):
        # only gray image
        sys.stderr.write('Input image should be grayscale')
        sys.exit(1)

    if I.dtype!=np.dtype('float32'):
        I = np.float32(I)

    # check the size is divisible by 8
    pad_r, pad_c = 0,0
    if (I.shape[0] % 8 or I.shape[1] % 8):
        pad_r = np.int32(8*np.ceil(I.shape[0]/8.0)-I.shape[0])
        pad_c = np.int32(8*np.ceil(I.shape[1]/8.0)-I.shape[1])
        I = np.pad(I,((0,pad_r),(0,pad_c)),'symmetric')

    # low-freq grain denoiser core
    A1_f2 = np.zeros((A1_f.shape[0]*2, A1_f.shape[1]*2), np.float32)
    lfnfilter.bilinear2(A1_f, A1_f2)
    z = I_h - A1_f2
    z_hat = fwddecouple(z)

    z_hat_s = np.zeros((z_hat.shape[0], z_hat.shape[1]), np.float32)
    lfnfilter.dftshrink(z_hat, z_hat_s, sigma_w*np.sqrt(2))
    p = np.maximum(np.abs(z_hat_s)/(sigma_w*2/3)-1,0)
    p_2 = p*p
    z_tilde_s = (p_2/(1+p_2))*z_hat_s
    I_F = invdecouple(z_tilde_s)-z+I

    if (pad_r or pad_c):
        I_F = I_F[0:y3.shape[0]-pad_r,0:y3.shape[1]-pad_c]

    #clip
    np.clip(I_F, 0, 255, I_F)

    return np.uint8(I_F)
