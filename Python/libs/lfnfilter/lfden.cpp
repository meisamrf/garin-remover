#include "fft2d.h"
#include "math.h"
#include <string.h>
#include <algorithm>
#include <thread>

#define max_threads 64


static void GetHalfBlock8(const float *img2D, float *block, int off_tmp)
{
	for (int k = 0; k<8; k++)
	{
		for (int m = 0; m<4; m++)
		{
			block[m * 8 + k] = *img2D++;
		}
		img2D += off_tmp;
	}
}

static void Add2Dblk8(float *img2D, const float *block, int off_tmp)
{
	for (int k = 0; k<8; k++)
	{
		for (int m = 0; m<8; m++)
		{			
			(*img2D++) += (*block++);
		}
		img2D += off_tmp;
	}
}

static void RescaleOverlap8(float *imgin, int img_col, int img_row)
{
	int l, k, imsize, strp, endp;
	imsize = img_col*img_row;
	strp = 4 * img_row;
	endp = imsize - strp;


	for (k = 0; k<strp; k++)
	{
		imgin[k] *= 2;
	}
	for (k = endp; k<imsize; k++)
	{
		imgin[k] *= 2;
	}

	strp = 0; endp = img_row - 1;
	for (k = 0; k<img_col; k++)
	{
		for (l = 0; l<4; l++)
		{
			imgin[strp + l] *= 2;
			imgin[endp - l] *= 2;
		}

		strp += img_row;
		endp += img_row;
	}
}

static void ShrinkCoef(float *fft_r, float *fft_i, float shrinkval, int n)
{
	// n for ifft and 4 for overlapping
	float scale = (float)n * 4;

	for (int k = 0; k<n; k++)
	{
		float ar = fft_r[k];
		float ai = fft_i[k];
		float shval = expf(-shrinkval / (ar*ar + ai*ai + 0.0001f)) / scale;
		fft_r[k] *= shval;
		fft_i[k] *= shval;
	}

}


void dft8shrinkrow(float *dst, const float *src, int img_row, int img_col, float nv, int col)
{

	fft2d m_fft2d;
	const int bsize = 8;
	const int bsize2 = bsize * bsize;
	const int HALF_BLK_SIZE = bsize / 2;
	const int hsize = bsize / 2;
	const int harea = (bsize2 / 2);
	int off_tmp, bound_col, bound_row, off_tmp_O;
	float block[bsize2], sfft_r[bsize2]
		, sfft_i[bsize2];
	float ffto_r_x[bsize2], ffto_i_x[bsize2];

	nv *= bsize2;

	off_tmp = img_row - hsize;
	bound_col = img_col - bsize;
	bound_row = img_row - bsize;
	off_tmp_O = img_row - bsize;

	if (col > bound_col)
		return;


	const float *ginrblk = src + col * img_row;
	float *goutrblk = dst + col * img_row;
	GetHalfBlock8(ginrblk, block, off_tmp);

	m_fft2d.fft8HorzHalf(block, sfft_r, sfft_i);
	ginrblk += hsize;
	for (int j = 0; j <= bound_row; j += hsize)
	{
		GetHalfBlock8(ginrblk, block + harea, off_tmp);
		ginrblk += hsize;
		m_fft2d.fft8HorzHalf(block + harea, sfft_r + harea, sfft_i + harea);
		m_fft2d.fft8_2d_v(sfft_r, sfft_i, ffto_r_x, ffto_i_x);

		ShrinkCoef(ffto_r_x, ffto_i_x, nv, bsize2);

		m_fft2d.ifftx8_2D(ffto_r_x, ffto_i_x);

		Add2Dblk8(goutrblk, ffto_r_x, off_tmp_O);

		goutrblk += hsize;
		memcpy(sfft_r, sfft_r + harea, harea * sizeof(float));
		memcpy(sfft_i, sfft_i + harea, harea * sizeof(float));
		memcpy(block, block + (bsize2 / 2), (bsize2 / 2) * sizeof(float));
	}

}

static void dft8shrinkjump(
	float *dst,
	const float *src,
	float nv,
	int img_row, int img_col, int offest, int jump) {

	const int bsize = 8;
	int img_col_lim = img_col - bsize;

	for (int x = offest; x <= img_col_lim; x += jump) {
		dft8shrinkrow(dst, src, img_row, img_col, nv, x);
	}
}

void stftshrink(float *dst, const float *src, int img_row, int img_col, float nv, int th_num)
{
	const int bsize = 8;
	std::thread t[max_threads];

	memset(dst, 0, img_col*img_row * sizeof(float));

	for (int i = 0; i < th_num; ++i) {
		t[i] = std::thread(dft8shrinkjump, dst, src, nv, img_row, img_col, i*bsize, th_num*bsize);
	}
	for (int i = 0; i < th_num; ++i) {
		t[i].join();
	}

	for (int i = 0; i < th_num; ++i) {
		t[i] = std::thread(dft8shrinkjump, dst, src, nv, img_row, img_col, i*bsize + bsize / 2, th_num*bsize);
	}
	for (int i = 0; i < th_num; ++i) {
		t[i].join();
	}

	RescaleOverlap8(dst, img_col, img_row);
}



void stft8shrink(const float *ginr, float *goutr, int img_row, int img_col, float nv)
{

	const float c_f = 4;
	fft2d m_fft2d;
	const int bsize = 8;
	const int bsize2 = bsize * bsize;
	const int HALF_BLK_SIZE = bsize / 2;
	const int hsize = bsize / 2;
	const int harea = (bsize2 / 2);
	int off_tmp, bound_col, bound_row, off_tmp_O;
	float block[bsize2], sfft_r[bsize2]
		, sfft_i[bsize2];
	float ffto_r_x[bsize2], ffto_i_x[bsize2];
	
	nv *= bsize2;
	memset(goutr, 0, img_col*img_row*sizeof(float));

	off_tmp = img_row - hsize;
	bound_col = img_col - bsize;
	bound_row = img_row - bsize;
	off_tmp_O = img_row - bsize;
	const float *ginrblk;
	float *goutrblk;

	
	for (int i = 0; i <= bound_col; i += hsize)
		{
			ginrblk = ginr + i*img_row;
			goutrblk = goutr + i*img_row;
			GetHalfBlock8(ginrblk, block, off_tmp);

			m_fft2d.fft8HorzHalf(block, sfft_r, sfft_i);
			ginrblk += hsize;
			for (int j = 0; j <= bound_row; j += hsize)
			{
				GetHalfBlock8(ginrblk, block + harea, off_tmp);
				ginrblk += hsize;
				m_fft2d.fft8HorzHalf(block + harea, sfft_r + harea, sfft_i + harea);
				m_fft2d.fft8_2d_v(sfft_r, sfft_i, ffto_r_x, ffto_i_x);

				ShrinkCoef(ffto_r_x, ffto_i_x, nv, bsize2);

				m_fft2d.ifftx8_2D(ffto_r_x, ffto_i_x);

				Add2Dblk8(goutrblk, ffto_r_x, off_tmp_O);
				
				goutrblk += hsize;
				memcpy(sfft_r, sfft_r + harea, harea * sizeof(float));
				memcpy(sfft_i, sfft_i + harea, harea * sizeof(float));
				memcpy(block, block + (bsize2 / 2), (bsize2 / 2)*sizeof(float));			
			}			
		}
	
	RescaleOverlap8(goutr, img_col, img_row);
}


void bilinear2Row(float *dst, const float *src, int img_row, int img_col, int x)
{
	int imc = img_col / 2;
	int imr = img_row / 2;

	int imcl = imc - 1;
	int imrl = imr - 1;


	int ifx = x / 2;
	float dx = (x % 2) / 2.f;
	int x_ceil = std::min(ifx + (x % 2), imcl)*imr;
	int x_floor = ifx * imr;

	dst = dst + x * img_row;

	for (int y = 0; y < img_row; ++y)
	{
		int ify = y / 2;
		float dy = (y % 2) / 2.f;
		int y_ceil = std::min(ify + (y % 2), imrl);
		int y_floor = ify;

		float Q11 = src[x_floor + y_floor];
		float Q21 = src[x_ceil + y_floor];
		float Q12 = src[x_floor + y_ceil];
		float Q22 = src[x_ceil + y_ceil];

		float R1 = (1 - dx)*Q11 + dx * Q21;
		float R2 = (1 - dx)*Q12 + dx * Q22;

		float v = (1 - dy)*R1 + dy * R2;

		*dst++ = v;
	}

}

static void bilinear2jump(
	float *dst,
	const float *src,
	int img_row, int img_col, int offest, int jump) {

	for (int x = offest; x < img_col; x += jump) {

		bilinear2Row(
			dst,
			src,
			img_row, img_col, x);
	}
}

static void bilinear2_mth_call(float *dst,
	const float *src,
	int img_row, int img_col, int th_num)
{

	std::thread t[max_threads];

	for (int i = 0; i < th_num; ++i) {
		t[i] = std::thread(bilinear2jump, dst, src, img_row, img_col, i, th_num);
	}
	for (int i = 0; i < th_num; ++i) {
		t[i].join();
	}
}

void bilinear2(const float *A, float *B, int img_row, int img_col)
{
	int imc = img_col / 2;
	int imr = img_row / 2;

	int imcl = imc - 1;
	int imrl = imr - 1;

	for (int x = 0; x < img_col; ++x)
	{
		int ifx = x/2;
		float dx = (x % 2) / 2.f;
		int x_ceil = std::min(ifx + (x % 2), imcl)*imr;
		int x_floor = ifx*imr;

		for (int y = 0; y <img_row; ++y)
		{
			int ify = y / 2;
			float dy = (y % 2) / 2.f;
			int y_ceil = std::min(ify + (y % 2), imrl);
			int y_floor = ify;

			float Q11 = A[x_floor + y_floor];
			float Q21 = A[x_ceil + y_floor];
			float Q12 = A[x_floor + y_ceil];
			float Q22 = A[x_ceil + y_ceil];

			float R1 = (1 - dx)*Q11 + dx*Q21;
			float R2 = (1 - dx)*Q12 + dx*Q22;

			float v = (1 - dy)*R1 + dy*R2;
			
			B[x*img_row+y] = v;
		}
	}
}

class LFNFilter
{
public:
	//static void bilinear2(const float * input, float * dataout, int img_row, int img_col);
	static void bilinear2(const float * input, float * dataout, int img_row, int img_col);
	//static void dftshrink(const float * input, float * dataout, int img_row, int img_col, float nv);
	static void dftshrink(const float *src, float *dst, int img_row, int img_col, float nv);
};

/*void LFNFilter::bilinear2(const float * input, float * dataout, int img_row, int img_col) {

	// change column and row major
	lfden::bilinear2(input, dataout, img_col * 2, img_row * 2);
}*/

void LFNFilter::bilinear2(const float * input, float * dataout, int img_row, int img_col) {

	int th_num = std::thread::hardware_concurrency();
	th_num = std::min(th_num, max_threads);
	// change column and row major
	bilinear2_mth_call(dataout,  input, img_col * 2, img_row * 2, th_num);
}

/*void LFNFilter::dftshrink(const float * input, float * dataout, int img_row, int img_col, float nv) {

	// change column and row major
	lfden::stft8shrink(input, dataout, img_col, img_row, nv);
}*/

void LFNFilter::dftshrink(const float *src, float *dst, int img_row, int img_col, float nv) {

	int th_num = std::thread::hardware_concurrency();
	th_num = std::min(th_num, max_threads);
	// change column and row major
	stftshrink(dst, src, img_col, img_row, nv, th_num);
}
