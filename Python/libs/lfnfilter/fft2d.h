#pragma once

class fft2d
{
public:
	fft2d();
	~fft2d();
	static void fft16HorzHalf(const float *blk_in, float *save_fft_r, float *save_fft_i);
	static void fft16HorzHalfComplex(const float *blk_inx, const float *blk_iny, float *save_fft_r, float *save_fft_i);
	static void fft8HorzHalf(float *blk_inx, float *save_fft_r, float *save_fft_i);
	static void fft16Full(float *horz_fft_r, float *horz_fft_i, float *fft_r, float *fft_i);
	static void ifftx16_2DNS(float *in_r, float *in_i);
	static void ifftx16Complex(float *in_r, float *in_i);
	static void fftx8_2D(float *in_r, float *in_i);
	static void ifftx8_2D(float *in_r, float *in_i);
	static void ifftx8_2DV(float *in_r, float *in_i);
	static void fft16Complex(float *horz_fft_r, float *horz_fft_i, float *fft_r, float *fft_i);
	static void fft8_2d_v(float *horz_fft_r, float *horz_fft_i, float *fft_r, float *fft_i); 	
};

