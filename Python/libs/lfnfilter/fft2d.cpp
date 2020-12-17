#include "fft2d.h"
#include "math.h"
#include <string.h>

fft2d::fft2d()
{
}


fft2d::~fft2d()
{
}


static void fftx16_4_s1(const float *in_r, float *o_r, float *o_i)
{
	float x1, x2;

	x1 = (in_r[0] + in_r[8]);
	x2 = (in_r[4] + in_r[12]);

	o_r[0] = x1 + x2;
	o_i[0] = 0;

	o_r[1] = (in_r[0] - in_r[8]);
	o_i[1] = (in_r[12] - in_r[4]);

	o_r[2] = x1 - x2;
	o_i[2] = 0;

	o_r[3] = o_r[1];
	o_i[3] = -o_i[1];
}

static void fftx16_4_s3(const float *in_r, float *o_r, float *o_i)
{
	float x1, x2;

	x1 = (in_r[2] + in_r[10]);
	x2 = (in_r[6] + in_r[14]);

	o_r[0] = x1 + x2;
	o_i[0] = 0;

	o_r[1] = (in_r[2] - in_r[10]);
	o_i[1] = (in_r[14] - in_r[6]);

	o_r[2] = x1 - x2;
	o_i[2] = 0;

	o_r[3] = o_r[1];
	o_i[3] = -o_i[1];
}


static void fftx16_4_s2(const float *in_r, float *o_r, float *o_i)
{
	float x1, x2;

	x1 = (in_r[1] + in_r[9]);
	x2 = (in_r[5] + in_r[13]);

	o_r[0] = x1 + x2;
	o_i[0] = 0;

	o_r[1] = (in_r[1] - in_r[9]);
	o_i[1] = (in_r[13] - in_r[5]);

	o_r[2] = x1 - x2;
	o_i[2] = 0;

	o_r[3] = o_r[1];
	o_i[3] = -o_i[1];
}

static void fftx16_4_s4(const float *in_r, float *o_r, float *o_i)
{
	float x1, x2;

	x1 = (in_r[3] + in_r[11]);
	x2 = (in_r[7] + in_r[15]);

	o_r[0] = x1 + x2;
	o_i[0] = 0;

	o_r[1] = (in_r[3] - in_r[11]);
	o_i[1] = (in_r[15] - in_r[7]);

	o_r[2] = x1 - x2;
	o_i[2] = 0;

	o_r[3] = o_r[1];
	o_i[3] = -o_i[1];
}


static void fftx16_8_o(const float *in_r, float *o_r, float *o_i)
{

	float X_r_e[4], X_i_e[4], X_r_o[4], X_i_o[4];

	fftx16_4_s2(in_r, X_r_e, X_i_e);
	fftx16_4_s4(in_r, X_r_o, X_i_o);

	o_r[0] = X_r_e[0] + X_r_o[0];
	o_i[0] = 0;

	o_r[1] = X_r_e[1] + 0.707106781186548F*(X_r_o[1] + X_i_o[1]);
	o_i[1] = X_i_e[1] + 0.707106781186548F*(X_i_o[1] - X_r_o[1]);

	o_r[2] = X_r_e[2] + X_i_o[2];
	o_i[2] = X_i_e[2] - X_r_o[2];

	o_r[3] = X_r_e[3] + 0.707106781186548F*(X_i_o[3] - X_r_o[3]);
	o_i[3] = X_i_e[3] - 0.707106781186548F*(X_i_o[3] + X_r_o[3]);

	o_r[4] = X_r_e[0] - X_r_o[0];
	o_i[4] = 0;

	o_r[5] = o_r[3];
	o_i[5] = -o_i[3];

	o_r[6] = o_r[2];
	o_i[6] = -o_i[2];

	o_r[7] = o_r[1];
	o_i[7] = -o_i[1];
}


static void fftx16_8_e(const float *in_r, float *o_r, float *o_i)
{
	float X_r_e[4], X_i_e[4], X_r_o[4], X_i_o[4];

	fftx16_4_s1(in_r, X_r_e, X_i_e);
	fftx16_4_s3(in_r, X_r_o, X_i_o);

	o_r[0] = X_r_e[0] + X_r_o[0];
	o_i[0] = 0;

	o_r[1] = X_r_e[1] + 0.707106781186548F*(X_r_o[1] + X_i_o[1]);
	o_i[1] = X_i_e[1] + 0.707106781186548F*(X_i_o[1] - X_r_o[1]);

	o_r[2] = X_r_e[2] + X_i_o[2];
	o_i[2] = X_i_e[2] - X_r_o[2];

	o_r[3] = X_r_e[3] + 0.707106781186548F*(X_i_o[3] - X_r_o[3]);
	o_i[3] = X_i_e[3] - 0.707106781186548F*(X_i_o[3] + X_r_o[3]);

	o_r[4] = X_r_e[0] - X_r_o[0];
	o_i[4] = 0;

	o_r[5] = o_r[3];
	o_i[5] = -o_i[3];

	o_r[6] = o_r[2];
	o_i[6] = -o_i[2];

	o_r[7] = o_r[1];
	o_i[7] = -o_i[1];
}

static void fftx16_1D_real_horz(const float *in_r, float *o_r, float *o_i)
{
	float X_r_e[8], X_i_e[8], X_r_o[8], X_i_o[8];
	float TS0_r, TS0_i, TS1_r, TS1_i, TS2_r, TS2_i, TS3_r, TS3_i, TS4_r, TS4_i, TS5_r, TS5_i, TS6_r, TS6_i;

	fftx16_8_e(in_r, X_r_e, X_i_e);
	fftx16_8_o(in_r, X_r_o, X_i_o);

	*o_r++ = X_r_e[0] + X_r_o[0];
	*o_i++ = 0;

	TS0_r = X_r_e[1] + 0.923879532511287F*X_r_o[1] + 0.382683432365090F*X_i_o[1];
	TS0_i = X_i_e[1] - 0.382683432365090F*X_r_o[1] + 0.923879532511287F*X_i_o[1];
	*o_r++ = TS0_r;
	*o_i++ = TS0_i;

	TS1_r = X_r_e[2] + 0.707106781186548F*(X_r_o[2] + X_i_o[2]);
	TS1_i = X_i_e[2] + 0.707106781186548F*(X_i_o[2] - X_r_o[2]);
	*o_r++ = TS1_r;
	*o_i++ = TS1_i;

	TS2_r = X_r_e[3] + 0.382683432365090F*X_r_o[3] + 0.923879532511287F*X_i_o[3];
	TS2_i = X_i_e[3] - 0.923879532511287F*X_r_o[3] + 0.382683432365090F*X_i_o[3];
	*o_r++ = TS2_r;
	*o_i++ = TS2_i;

	TS3_r = X_r_e[4] + X_i_o[4];
	TS3_i = X_i_e[4] - X_r_o[4];
	*o_r++ = TS3_r;
	*o_i++ = TS3_i;

	TS4_r = X_r_e[5] - 0.382683432365090F*X_r_o[5] + 0.923879532511287F*X_i_o[5];
	TS4_i = X_i_e[5] - 0.923879532511287F*X_r_o[5] - 0.382683432365090F*X_i_o[5];
	*o_r++ = TS4_r;
	*o_i++ = TS4_i;

	TS5_r = X_r_e[6] + 0.707106781186548F*(X_i_o[6] - X_r_o[6]);
	TS5_i = X_i_e[6] - 0.707106781186548F*(X_i_o[6] + X_r_o[6]);
	*o_r++ = TS5_r;
	*o_i++ = TS5_i;

	TS6_r = X_r_e[7] - 0.923879532511287F*X_r_o[7] + 0.382683432365090F*X_i_o[7];
	TS6_i = X_i_e[7] - 0.382683432365090F*X_r_o[7] - 0.923879532511287F*X_i_o[7];
	*o_r++ = TS6_r;
	*o_i++ = TS6_i;

	*o_r++ = X_r_e[0] - X_r_o[0];
	*o_i++ = 0;

	*o_r++ = TS6_r;
	*o_i++ = -TS6_i;

	*o_r++ = TS5_r;
	*o_i++ = -TS5_i;

	*o_r++ = TS4_r;
	*o_i++ = -TS4_i;

	*o_r++ = TS3_r;
	*o_i++ = -TS3_i;

	*o_r++ = TS2_r;
	*o_i++ = -TS2_i;

	*o_r++ = TS1_r;
	*o_i++ = -TS1_i;

	*o_r++ = TS0_r;
	*o_i++ = -TS0_i;
}

void fft2d::fft16HorzHalf(const float *blk_in, float *save_fft_r, float *save_fft_i)
{
	int k;
	float *p_out_r, *p_out_i;


	const float* p_in_r = blk_in;
	p_out_r = save_fft_r;
	p_out_i = save_fft_i;

	for (k = 0; k<8; k++)
	{

		fftx16_1D_real_horz(p_in_r, p_out_r, p_out_i);

		p_in_r += 16;
		p_out_r += 16;
		p_out_i += 16;
	}
}

void fft2d::fft16HorzHalfComplex(const float *blk_inx, const float *blk_iny, float *save_fft_r, float *save_fft_i)
{

	float dummmy_r[16];
	float dummmy_i[16];

	const float* p_in_r_x = blk_inx;
	const float* p_in_r_y = blk_iny;

	float *p_out_r = save_fft_r;
	float *p_out_i = save_fft_i;

	for (int k = 0; k<8; k++)
	{

		fftx16_1D_real_horz(p_in_r_x, p_out_r, p_out_i);
		fftx16_1D_real_horz(p_in_r_y, dummmy_r, dummmy_i);

		for (int j = 0; j<16; j++)
		{
			p_out_r[j] -= dummmy_i[j];
			p_out_i[j] += dummmy_r[j];
		}

		p_in_r_x += 16;
		p_in_r_y += 16;
		p_out_r += 16;
		p_out_i += 16;
	}
}

static void fftx8_1D_realT(float *in_r, float *o_r, float *o_i)
{

	float X_r_e[4], X_i_e[4], X_r_o[4], X_i_o[4];
	float x1, x2;

	// *********************
	x1 = (in_r[0] + in_r[4]);
	x2 = (in_r[2] + in_r[6]);

	X_r_e[0] = x1 + x2;
	X_i_e[0] = 0;

	X_r_e[1] = (in_r[0] - in_r[4]);
	X_i_e[1] = (in_r[6] - in_r[2]);

	X_r_e[2] = x1 - x2;
	X_i_e[2] = 0;

	X_r_e[3] = X_r_e[1];
	X_i_e[3] = -X_i_e[1];

	// **************************

	x1 = (in_r[1] + in_r[5]);
	x2 = (in_r[3] + in_r[7]);

	X_r_o[0] = x1 + x2;
	X_i_o[0] = 0;

	X_r_o[1] = (in_r[1] - in_r[5]);
	X_i_o[1] = (in_r[7] - in_r[3]);

	X_r_o[2] = x1 - x2;
	X_i_o[2] = 0;

	X_r_o[3] = X_r_o[1];
	X_i_o[3] = -X_i_o[1];

	// **************************
	float TXR1, TXR2, TXR3, TXI1, TXI2, TXI3;
	*o_r++ = X_r_e[0] + X_r_o[0];
	*o_i++ = 0;

	TXR1 = X_r_e[1] + (0.707106781186548f)*(X_r_o[1] + X_i_o[1]);
	*o_r++ = TXR1;
	TXI1 = X_i_e[1] + (0.707106781186548f)*(X_i_o[1] - X_r_o[1]);
	*o_i++ = TXI1;

	TXR2 = X_r_e[2] + X_i_o[2];
	*o_r++ = TXR2;
	TXI2 = X_i_e[2] - X_r_o[2];
	*o_i++ = TXI2;

	TXR3 = X_r_e[3] + (0.707106781186548f)*(X_i_o[3] - X_r_o[3]);
	*o_r++ = TXR3;
	TXI3 = X_i_e[3] - (0.707106781186548f)*(X_i_o[3] + X_r_o[3]);
	*o_i++ = TXI3;

	*o_r++ = X_r_e[0] - X_r_o[0];
	*o_i++ = 0;

	*o_r++ = TXR3;
	*o_i++ = -TXI3;

	*o_r++ = TXR2;
	*o_i++ = -TXI2;

	*o_r++ = TXR1;
	*o_i++ = -TXI1;

}

void fft2d::fft8HorzHalf(float *blk_in, float *save_fft_r, float *save_fft_i)
{


	float* p_in_r = blk_in;

	float *p_out_r = save_fft_r;
	float *p_out_i = save_fft_i;


	for (int k = 0; k<4; k++)
	{
		fftx8_1D_realT(p_in_r, p_out_r, p_out_i);
	
		p_in_r += 8;
		p_out_r += 8;
		p_out_i += 8;
	}
}

static void fftx16_4_s1_complex(const float *in_r, const float *in_i, float *o_r, float *o_i)
{

	float x1, x2, x1_i, x2_i, CC1_r, CC1_i, CC2_r, CC2_i;

	x1 = (in_r[0] + in_r[8]);
	x2 = (in_r[4] + in_r[12]);

	x1_i = (in_i[0] + in_i[8]);
	x2_i = (in_i[4] + in_i[12]);

	o_r[0] = x1 + x2;
	o_i[0] = x1_i + x2_i;

	CC1_r = (in_r[0] - in_r[8]);
	CC1_i = (in_i[0] - in_i[8]);

	CC2_r = (in_r[12] - in_r[4]);
	CC2_i = (in_i[12] - in_i[4]);

	o_r[1] = CC1_r - CC2_i;
	o_i[1] = CC2_r + CC1_i;

	o_r[2] = x1 - x2;
	o_i[2] = x1_i - x2_i;

	o_r[3] = CC1_r + CC2_i;
	o_i[3] = -CC2_r + CC1_i;
}

static void fftx16_4_s3_complex(const float *in_r, const float *in_i, float *o_r, float *o_i)
{

	float x1, x2, x1_i, x2_i, CC1_r, CC1_i, CC2_r, CC2_i;

	x1 = (in_r[2] + in_r[10]);
	x2 = (in_r[6] + in_r[14]);

	x1_i = (in_i[2] + in_i[10]);
	x2_i = (in_i[6] + in_i[14]);

	o_r[0] = x1 + x2;
	o_i[0] = x1_i + x2_i;

	CC1_r = (in_r[2] - in_r[10]);
	CC1_i = (in_i[2] - in_i[10]);

	CC2_r = (in_r[14] - in_r[6]);
	CC2_i = (in_i[14] - in_i[6]);

	o_r[1] = CC1_r - CC2_i;
	o_i[1] = CC2_r + CC1_i;

	o_r[2] = x1 - x2;
	o_i[2] = x1_i - x2_i;

	o_r[3] = CC1_r + CC2_i;
	o_i[3] = -CC2_r + CC1_i;
}

static void fftx16_8_1D_complex_e(const float *in_r, const float *in_i, float *o_r, float *o_i)
{

	float X_r_e[4], X_i_e[4], X_r_o[4], X_i_o[4], tx1, tx2, tx3, tx4;


	fftx16_4_s1_complex(in_r, in_i, X_r_e, X_i_e);
	fftx16_4_s3_complex(in_r, in_i, X_r_o, X_i_o);

	o_r[0] = X_r_e[0] + X_r_o[0];
	o_i[0] = X_i_e[0] + X_i_o[0];

	tx1 = 0.707106781186548F*(X_r_o[1] + X_i_o[1]);
	tx2 = 0.707106781186548F*(X_i_o[1] - X_r_o[1]);


	o_r[1] = X_r_e[1] + tx1;
	o_i[1] = X_i_e[1] + tx2;

	o_r[2] = X_r_e[2] + X_i_o[2];
	o_i[2] = X_i_e[2] - X_r_o[2];

	tx3 = 0.707106781186548F*(X_i_o[3] - X_r_o[3]);
	tx4 = 0.707106781186548F*(X_i_o[3] + X_r_o[3]);

	o_r[3] = X_r_e[3] + tx3;
	o_i[3] = X_i_e[3] - tx4;

	o_r[4] = X_r_e[0] - X_r_o[0];
	o_i[4] = X_i_e[0] - X_i_o[0];

	o_r[5] = X_r_e[1] - tx1;
	o_i[5] = X_i_e[1] - tx2;

	o_r[6] = X_r_e[2] - X_i_o[2];
	o_i[6] = X_i_e[2] + X_r_o[2];

	o_r[7] = X_r_e[3] - tx3;
	o_i[7] = X_i_e[3] + tx4;
}

static void fftx16_4_s2_complex(const float *in_r, const float *in_i, float *o_r, float *o_i)
{

	float x1, x2, x1_i, x2_i, CC1_r, CC1_i, CC2_r, CC2_i;

	x1 = (in_r[1] + in_r[9]);
	x2 = (in_r[5] + in_r[13]);

	x1_i = (in_i[1] + in_i[9]);
	x2_i = (in_i[5] + in_i[13]);

	o_r[0] = x1 + x2;
	o_i[0] = x1_i + x2_i;

	CC1_r = (in_r[1] - in_r[9]);
	CC1_i = (in_i[1] - in_i[9]);

	CC2_r = (in_r[13] - in_r[5]);
	CC2_i = (in_i[13] - in_i[5]);

	o_r[1] = CC1_r - CC2_i;
	o_i[1] = CC2_r + CC1_i;

	o_r[2] = x1 - x2;
	o_i[2] = x1_i - x2_i;

	o_r[3] = CC1_r + CC2_i;
	o_i[3] = -CC2_r + CC1_i;
}

static void fftx16_4_s4_complex(const float *in_r, const float *in_i, float *o_r, float *o_i)
{

	float x1, x2, x1_i, x2_i, CC1_r, CC1_i, CC2_r, CC2_i;

	x1 = (in_r[3] + in_r[11]);
	x2 = (in_r[7] + in_r[15]);

	x1_i = (in_i[3] + in_i[11]);
	x2_i = (in_i[7] + in_i[15]);

	o_r[0] = x1 + x2;
	o_i[0] = x1_i + x2_i;

	CC1_r = (in_r[3] - in_r[11]);
	CC1_i = (in_i[3] - in_i[11]);

	CC2_r = (in_r[15] - in_r[7]);
	CC2_i = (in_i[15] - in_i[7]);

	o_r[1] = CC1_r - CC2_i;
	o_i[1] = CC2_r + CC1_i;

	o_r[2] = x1 - x2;
	o_i[2] = x1_i - x2_i;

	o_r[3] = CC1_r + CC2_i;
	o_i[3] = -CC2_r + CC1_i;

}

static void fftx16_8_1D_complex_o(const float *in_r, const float *in_i, float *o_r, float *o_i)
{

	float X_r_e[4], X_i_e[4], X_r_o[4], X_i_o[4], tx1, tx2, tx3, tx4;

	fftx16_4_s2_complex(in_r, in_i, X_r_e, X_i_e);
	fftx16_4_s4_complex(in_r, in_i, X_r_o, X_i_o);

	o_r[0] = X_r_e[0] + X_r_o[0];
	o_i[0] = X_i_e[0] + X_i_o[0];

	tx1 = 0.707106781186548F*(X_r_o[1] + X_i_o[1]);
	tx2 = 0.707106781186548F*(X_i_o[1] - X_r_o[1]);


	o_r[1] = X_r_e[1] + tx1;
	o_i[1] = X_i_e[1] + tx2;

	o_r[2] = X_r_e[2] + X_i_o[2];
	o_i[2] = X_i_e[2] - X_r_o[2];

	tx3 = 0.707106781186548F*(X_i_o[3] - X_r_o[3]);
	tx4 = 0.707106781186548F*(X_i_o[3] + X_r_o[3]);

	o_r[3] = X_r_e[3] + tx3;
	o_i[3] = X_i_e[3] - tx4;

	o_r[4] = X_r_e[0] - X_r_o[0];
	o_i[4] = X_i_e[0] - X_i_o[0];

	o_r[5] = X_r_e[1] - tx1;
	o_i[5] = X_i_e[1] - tx2;

	o_r[6] = X_r_e[2] - X_i_o[2];
	o_i[6] = X_i_e[2] + X_r_o[2];

	o_r[7] = X_r_e[3] - tx3;
	o_i[7] = X_i_e[3] + tx4;
}

static void fftx16_1D_complex_horz(const float *in_r, const float *in_i, float *o_r, float *o_i)
{
	float X_r_e[8], X_i_e[8], X_r_o[8], X_i_o[8];
	float tx1, tx2, tx3, tx4, tx5, tx6, tx7, tx8, tx9, tx10, tx11, tx12;

	fftx16_8_1D_complex_e(in_r, in_i, X_r_e, X_i_e);
	fftx16_8_1D_complex_o(in_r, in_i, X_r_o, X_i_o);

	*o_r++ = X_r_e[0] + X_r_o[0];
	*o_i++ = X_i_e[0] + X_i_o[0];

	tx1 = 0.923879532511287F*X_r_o[1] + 0.382683432365090F*X_i_o[1];
	tx2 = 0.382683432365090F*X_r_o[1] - 0.923879532511287F*X_i_o[1];

	*o_r++ = X_r_e[1] + tx1;
	*o_i++ = X_i_e[1] - tx2;

	tx3 = 0.707106781186548F*(X_r_o[2] + X_i_o[2]);
	tx4 = 0.707106781186548F*(X_i_o[2] - X_r_o[2]);

	*o_r++ = X_r_e[2] + tx3;
	*o_i++ = X_i_e[2] + tx4;

	tx5 = 0.382683432365090F*X_r_o[3] + 0.923879532511287F*X_i_o[3];
	tx6 = 0.923879532511287F*X_r_o[3] - 0.382683432365090F*X_i_o[3];

	*o_r++ = X_r_e[3] + tx5;
	*o_i++ = X_i_e[3] - tx6;

	*o_r++ = X_r_e[4] + X_i_o[4];
	*o_i++ = X_i_e[4] - X_r_o[4];

	tx7 = 0.382683432365090F*X_r_o[5] - 0.923879532511287F*X_i_o[5];
	tx8 = 0.923879532511287F*X_r_o[5] + 0.382683432365090F*X_i_o[5];

	*o_r++ = X_r_e[5] - tx7;
	*o_i++ = X_i_e[5] - tx8;

	tx9 = 0.707106781186548F*(X_i_o[6] - X_r_o[6]);
	tx10 = 0.707106781186548F*(X_i_o[6] + X_r_o[6]);

	*o_r++ = X_r_e[6] + tx9;
	*o_i++ = X_i_e[6] - tx10;

	tx11 = 0.923879532511287F*X_r_o[7] - 0.382683432365090F*X_i_o[7];
	tx12 = 0.382683432365090F*X_r_o[7] + 0.923879532511287F*X_i_o[7];

	*o_r++ = X_r_e[7] - tx11;
	*o_i++ = X_i_e[7] - tx12;

	*o_r++ = X_r_e[0] - X_r_o[0];
	*o_i++ = X_i_e[0] - X_i_o[0];

	*o_r++ = X_r_e[1] - tx1;
	*o_i++ = X_i_e[1] + tx2;

	*o_r++ = X_r_e[2] - tx3;
	*o_i++ = X_i_e[2] - tx4;

	*o_r++ = X_r_e[3] - tx5;
	*o_i++ = X_i_e[3] + tx6;

	*o_r++ = X_r_e[4] - X_i_o[4];
	*o_i++ = X_i_e[4] + X_r_o[4];

	*o_r++ = X_r_e[5] + tx7;
	*o_i++ = X_i_e[5] + tx8;

	*o_r++ = X_r_e[6] - tx9;
	*o_i++ = X_i_e[6] + tx10;

	*o_r++ = X_r_e[7] + tx11;
	*o_i++ = X_i_e[7] + tx12;
}

static void ifftx16_4_s1_sym(float *in_r, float *in_i, float *o_r)
{
	float x1, x2, CC1_r, CC2_i;

	x1 = (in_r[0] + in_r[8]);
	x2 = (in_r[4] + in_r[12]);

	o_r[0] = x1 + x2;

	CC1_r = (in_r[0] - in_r[8]);
	CC2_i = (in_i[12] - in_i[4]);

	o_r[1] = CC1_r + CC2_i;

	o_r[2] = x1 - x2;

	o_r[3] = CC1_r - CC2_i;
}

static void ifftx16_4_s3_sym(float *in_r, float *in_i, float *o_r, float *o_i)
{
	float x1, x1_i, CC1_r, CC1_i;

	x1 = (in_r[2] + in_r[10]);
	x1_i = (in_i[2] + in_i[10]);

	o_r[0] = x1 + x1;
	o_i[0] = 0;

	CC1_r = (in_r[2] - in_r[10]);
	CC1_i = (in_i[2] - in_i[10]);

	o_r[1] = CC1_r - CC1_i;
	o_i[1] = -o_r[1];

	o_r[2] = 0;
	o_i[2] = x1_i + x1_i;

	o_r[3] = CC1_r + CC1_i;
	o_i[3] = o_r[3];
}

static void ifftx16_8_1D_sym_e(float *in_r, float *in_i, float *o_r)
{
	float X_r_e[4], X_r_o[4], X_i_o[4], tx1, tx3;

	ifftx16_4_s1_sym(in_r, in_i, X_r_e);

	ifftx16_4_s3_sym(in_r, in_i, X_r_o, X_i_o);

	o_r[0] = X_r_e[0] + X_r_o[0];

	tx1 = 0.707106781186548F*(X_r_o[1] - X_i_o[1]);

	o_r[1] = X_r_e[1] + tx1;

	o_r[2] = X_r_e[2] - X_i_o[2];

	tx3 = 0.707106781186548F*(X_i_o[3] + X_r_o[3]);

	o_r[3] = X_r_e[3] - tx3;

	o_r[4] = X_r_e[0] - X_r_o[0];

	o_r[5] = X_r_e[1] - tx1;

	o_r[6] = X_r_e[2] + X_i_o[2];

	o_r[7] = X_r_e[3] + tx3;
}

static void ifftx16_4_s2_sym(float *in_r, float *in_i, float *o_r, float *o_i)
{
	float x1, x2, x1_i, x2_i, CC1_r, CC1_i, CC2_r, CC2_i;

	x1 = (in_r[1] + in_r[9]);
	x2 = (in_r[5] + in_r[13]);

	x1_i = (in_i[1] + in_i[9]);
	x2_i = (in_i[5] + in_i[13]);

	o_r[0] = x1 + x2;
	o_i[0] = x1_i + x2_i;

	CC1_r = (in_r[1] - in_r[9]);
	CC1_i = (in_i[1] - in_i[9]);

	CC2_r = (in_r[13] - in_r[5]);
	CC2_i = (in_i[13] - in_i[5]);

	o_r[1] = CC1_r + CC2_i;
	o_i[1] = -CC2_r + CC1_i;

	o_r[2] = x1 - x2;
	o_i[2] = x1_i - x2_i;

	o_r[3] = CC1_r - CC2_i;
	o_i[3] = CC2_r + CC1_i;
}

static void ifftx16_8_1D_sym_o(float *in_r, float *in_i, float *o_r, float *o_i)
{
	float X_r_e[4], X_i_e[4], tx1, tx2, tx3, tx4;

	ifftx16_4_s2_sym(in_r, in_i, X_r_e, X_i_e);

	o_r[0] = X_r_e[0] + X_r_e[0];
	o_i[0] = 0;

	tx1 = 0.707106781186548F*(-X_i_e[1] + X_r_e[1]);
	tx2 = 0.707106781186548F*(-X_r_e[1] - X_i_e[1]);

	o_r[1] = X_r_e[1] + tx1;
	o_i[1] = X_i_e[1] + tx2;

	o_r[2] = X_r_e[2] - X_i_e[2];
	o_i[2] = X_i_e[2] - X_r_e[2];

	tx3 = 0.707106781186548F*(X_r_e[3] + X_i_e[3]);
	tx4 = 0.707106781186548F*(X_r_e[3] - X_i_e[3]);

	o_r[3] = X_r_e[3] - tx3;
	o_i[3] = X_i_e[3] - tx4;

	o_r[4] = 0;
	o_i[4] = X_i_e[0] + X_i_e[0];

	o_r[5] = X_r_e[1] - tx1;
	o_i[5] = X_i_e[1] - tx2;

	o_r[6] = X_r_e[2] + X_i_e[2];
	o_i[6] = X_i_e[2] + X_r_e[2];

	o_r[7] = X_r_e[3] + tx3;
	o_i[7] = X_i_e[3] + tx4;
}

static void ifftx16_1D_sym(float *in_r, float *in_i, int row, float *o_r, float *o_i)
{

	float X_r_e[8], X_r_o[8], X_i_o[8], tx1, tx3, tx5, tx7, tx9, tx11;

	ifftx16_8_1D_sym_e(in_r, in_i, X_r_e);

	ifftx16_8_1D_sym_o(in_r, in_i, X_r_o, X_i_o);

	o_r[row] = X_r_e[0] + X_r_o[0];
	o_i[row] = 0;

	tx1 = 0.923879532511287F*X_r_o[1] - 0.382683432365090F*X_i_o[1];

	o_r[16 + row] = X_r_e[1] + tx1;
	o_i[16 + row] = 0;

	tx3 = 0.707106781186548F*(X_r_o[2] - X_i_o[2]);

	o_r[32 + row] = X_r_e[2] + tx3;
	o_i[32 + row] = 0;

	tx5 = 0.382683432365090F*X_r_o[3] - 0.923879532511287F*X_i_o[3];

	o_r[48 + row] = X_r_e[3] + tx5;
	o_i[48 + row] = 0;

	o_r[64 + row] = X_r_e[4] - X_i_o[4];
	o_i[64 + row] = 0;

	tx7 = 0.382683432365090F*X_r_o[5] + 0.923879532511287F*X_i_o[5];

	o_r[80 + row] = X_r_e[5] - tx7;
	o_i[80 + row] = 0;

	tx9 = 0.707106781186548F*(X_i_o[6] + X_r_o[6]);

	o_r[96 + row] = X_r_e[6] - tx9;
	o_i[96 + row] = 0;

	tx11 = 0.923879532511287F*X_r_o[7] + 0.382683432365090F*X_i_o[7];

	o_r[112 + row] = X_r_e[7] - tx11;
	o_i[112 + row] = 0;

	o_r[128 + row] = X_r_e[0] - X_r_o[0];
	o_i[128 + row] = 0;

	o_r[144 + row] = X_r_e[1] - tx1;
	o_i[144 + row] = 0;

	o_r[160 + row] = X_r_e[2] - tx3;
	o_i[160 + row] = 0;

	o_r[176 + row] = X_r_e[3] - tx5;
	o_i[176 + row] = 0;

	o_r[192 + row] = X_r_e[4] + X_i_o[4];
	o_i[192 + row] = 0;

	o_r[208 + row] = X_r_e[5] + tx7;
	o_i[208 + row] = 0;

	o_r[224 + row] = X_r_e[6] + tx9;
	o_i[224 + row] = 0;

	o_r[240 + row] = X_r_e[7] + tx11;
	o_i[240 + row] = 0;
}

static void ifftx16_4_s1_asym(float *in_r, float *in_i, float *o_r, float *o_i)
{
	float x1, x2, x1_i, x2_i, CC1_r, CC1_i, CC2_r, CC2_i;

	x1 = (in_r[0] + in_r[8]);
	x2 = (in_r[4] + in_r[12]);

	x1_i = (in_i[0] + in_i[8]);
	x2_i = (in_i[4] + in_i[12]);

	o_r[0] = x1 + x2;
	o_i[0] = x1_i + x2_i;

	CC1_r = (in_r[0] - in_r[8]);
	CC1_i = (in_i[0] - in_i[8]);

	CC2_r = (in_r[12] - in_r[4]);
	CC2_i = (in_i[12] - in_i[4]);

	o_r[1] = CC1_r + CC2_i;
	o_i[1] = -CC2_r + CC1_i;

	o_r[2] = x1 - x2;
	o_i[2] = x1_i - x2_i;

	o_r[3] = CC1_r - CC2_i;
	o_i[3] = CC2_r + CC1_i;
}

static void ifftx16_4_s3_asym(float *in_r, float *in_i, float *o_r, float *o_i)
{
	float x1, x2, x1_i, x2_i, CC1_r, CC1_i, CC2_r, CC2_i;

	x1 = (in_r[2] + in_r[10]);
	x2 = (in_r[6] + in_r[14]);

	x1_i = (in_i[2] + in_i[10]);
	x2_i = (in_i[6] + in_i[14]);

	o_r[0] = x1 + x2;
	o_i[0] = x1_i + x2_i;

	CC1_r = (in_r[2] - in_r[10]);
	CC1_i = (in_i[2] - in_i[10]);

	CC2_r = (in_r[14] - in_r[6]);
	CC2_i = (in_i[14] - in_i[6]);

	o_r[1] = CC1_r + CC2_i;
	o_i[1] = -CC2_r + CC1_i;

	o_r[2] = x1 - x2;
	o_i[2] = x1_i - x2_i;

	o_r[3] = CC1_r - CC2_i;
	o_i[3] = CC2_r + CC1_i;
}

static void ifftx16_8_1D_asym_e(float *in_r, float *in_i, float *o_r, float *o_i)
{
	float X_r_e[4], X_i_e[4], X_r_o[4], X_i_o[4], tx1, tx2, tx3, tx4;


	ifftx16_4_s1_asym(in_r, in_i, X_r_e, X_i_e);
	ifftx16_4_s3_asym(in_r, in_i, X_r_o, X_i_o);

	o_r[0] = X_r_e[0] + X_r_o[0];
	o_i[0] = X_i_e[0] + X_i_o[0];

	tx1 = 0.707106781186548F*(X_r_o[1] - X_i_o[1]);
	tx2 = 0.707106781186548F*(X_i_o[1] + X_r_o[1]);


	o_r[1] = X_r_e[1] + tx1;
	o_i[1] = X_i_e[1] + tx2;

	o_r[2] = X_r_e[2] - X_i_o[2];
	o_i[2] = X_i_e[2] + X_r_o[2];

	tx3 = 0.707106781186548F*(X_i_o[3] + X_r_o[3]);
	tx4 = 0.707106781186548F*(X_i_o[3] - X_r_o[3]);

	o_r[3] = X_r_e[3] - tx3;
	o_i[3] = X_i_e[3] - tx4;

	o_r[4] = X_r_e[0] - X_r_o[0];
	o_i[4] = X_i_e[0] - X_i_o[0];

	o_r[5] = X_r_e[1] - tx1;
	o_i[5] = X_i_e[1] - tx2;

	o_r[6] = X_r_e[2] + X_i_o[2];
	o_i[6] = X_i_e[2] - X_r_o[2];

	o_r[7] = X_r_e[3] + tx3;
	o_i[7] = X_i_e[3] + tx4;
}

static void ifftx16_4_s2_asym(float *in_r, float *in_i, float *o_r, float *o_i)
{
	float x1, x2, x1_i, x2_i, CC1_r, CC1_i, CC2_r, CC2_i;

	x1 = (in_r[1] + in_r[9]);
	x2 = (in_r[5] + in_r[13]);

	x1_i = (in_i[1] + in_i[9]);
	x2_i = (in_i[5] + in_i[13]);

	o_r[0] = x1 + x2;
	o_i[0] = x1_i + x2_i;

	CC1_r = (in_r[1] - in_r[9]);
	CC1_i = (in_i[1] - in_i[9]);

	CC2_r = (in_r[13] - in_r[5]);
	CC2_i = (in_i[13] - in_i[5]);

	o_r[1] = CC1_r + CC2_i;
	o_i[1] = -CC2_r + CC1_i;

	o_r[2] = x1 - x2;
	o_i[2] = x1_i - x2_i;

	o_r[3] = CC1_r - CC2_i;
	o_i[3] = CC2_r + CC1_i;
}

static void ifftx16_4_s4_asym(float *in_r, float *in_i, float *o_r, float *o_i)
{
	float x1, x2, x1_i, x2_i, CC1_r, CC1_i, CC2_r, CC2_i;

	x1 = (in_r[3] + in_r[11]);
	x2 = (in_r[7] + in_r[15]);

	x1_i = (in_i[3] + in_i[11]);
	x2_i = (in_i[7] + in_i[15]);

	o_r[0] = x1 + x2;
	o_i[0] = x1_i + x2_i;

	CC1_r = (in_r[3] - in_r[11]);
	CC1_i = (in_i[3] - in_i[11]);

	CC2_r = (in_r[15] - in_r[7]);
	CC2_i = (in_i[15] - in_i[7]);

	o_r[1] = CC1_r + CC2_i;
	o_i[1] = -CC2_r + CC1_i;

	o_r[2] = x1 - x2;
	o_i[2] = x1_i - x2_i;

	o_r[3] = CC1_r - CC2_i;
	o_i[3] = CC2_r + CC1_i;
}



static void ifftx16_8_1D_asym_o(float *in_r, float *in_i, float *o_r, float *o_i)
{
	float X_r_e[4], X_i_e[4], X_r_o[4], X_i_o[4], tx1, tx2, tx3, tx4;

	ifftx16_4_s2_asym(in_r, in_i, X_r_e, X_i_e);
	ifftx16_4_s4_asym(in_r, in_i, X_r_o, X_i_o);

	o_r[0] = X_r_e[0] + X_r_o[0];
	o_i[0] = X_i_e[0] + X_i_o[0];

	tx1 = 0.707106781186548F*(X_r_o[1] - X_i_o[1]);
	tx2 = 0.707106781186548F*(X_i_o[1] + X_r_o[1]);

	o_r[1] = X_r_e[1] + tx1;
	o_i[1] = X_i_e[1] + tx2;

	o_r[2] = X_r_e[2] - X_i_o[2];
	o_i[2] = X_i_e[2] + X_r_o[2];

	tx3 = 0.707106781186548F*(X_i_o[3] + X_r_o[3]);
	tx4 = 0.707106781186548F*(X_i_o[3] - X_r_o[3]);

	o_r[3] = X_r_e[3] - tx3;
	o_i[3] = X_i_e[3] - tx4;

	o_r[4] = X_r_e[0] - X_r_o[0];
	o_i[4] = X_i_e[0] - X_i_o[0];

	o_r[5] = X_r_e[1] - tx1;
	o_i[5] = X_i_e[1] - tx2;

	o_r[6] = X_r_e[2] + X_i_o[2];
	o_i[6] = X_i_e[2] - X_r_o[2];

	o_r[7] = X_r_e[3] + tx3;
	o_i[7] = X_i_e[3] + tx4;
}



static void ifftx16_1D_asym(float *in_r, float *in_i, int row, float *o_r, float *o_i)
{

	float X_r_e[8], X_i_e[8], X_r_o[8], X_i_o[8], tx1, tx2, tx3, tx4, tx5, tx6, tx7, tx8, tx9, tx10, tx11, tx12;

	ifftx16_8_1D_asym_e(in_r, in_i, X_r_e, X_i_e);
	ifftx16_8_1D_asym_o(in_r, in_i, X_r_o, X_i_o);

	o_r[row] = X_r_e[0] + X_r_o[0];
	o_i[row] = X_i_e[0] + X_i_o[0];

	tx1 = 0.923879532511287F*X_r_o[1] - 0.382683432365090F*X_i_o[1];
	tx2 = 0.382683432365090F*X_r_o[1] + 0.923879532511287F*X_i_o[1];

	o_r[16 + row] = X_r_e[1] + tx1;
	o_i[16 + row] = X_i_e[1] + tx2;

	tx3 = 0.707106781186548F*(X_r_o[2] - X_i_o[2]);
	tx4 = 0.707106781186548F*(X_i_o[2] + X_r_o[2]);

	o_r[32 + row] = X_r_e[2] + tx3;
	o_i[32 + row] = X_i_e[2] + tx4;

	tx5 = 0.382683432365090F*X_r_o[3] - 0.923879532511287F*X_i_o[3];
	tx6 = 0.923879532511287F*X_r_o[3] + 0.382683432365090F*X_i_o[3];

	o_r[48 + row] = X_r_e[3] + tx5;
	o_i[48 + row] = X_i_e[3] + tx6;

	o_r[64 + row] = X_r_e[4] - X_i_o[4];
	o_i[64 + row] = X_i_e[4] + X_r_o[4];

	tx7 = 0.382683432365090F*X_r_o[5] + 0.923879532511287F*X_i_o[5];
	tx8 = 0.923879532511287F*X_r_o[5] - 0.382683432365090F*X_i_o[5];

	o_r[80 + row] = X_r_e[5] - tx7;
	o_i[80 + row] = X_i_e[5] + tx8;

	tx9 = 0.707106781186548F*(X_i_o[6] + X_r_o[6]);
	tx10 = 0.707106781186548F*(X_i_o[6] - X_r_o[6]);

	o_r[96 + row] = X_r_e[6] - tx9;
	o_i[96 + row] = X_i_e[6] - tx10;

	tx11 = 0.923879532511287F*X_r_o[7] + 0.382683432365090F*X_i_o[7];
	tx12 = 0.382683432365090F*X_r_o[7] - 0.923879532511287F*X_i_o[7];

	o_r[112 + row] = X_r_e[7] - tx11;
	o_i[112 + row] = X_i_e[7] + tx12;

	o_r[128 + row] = X_r_e[0] - X_r_o[0];
	o_i[128 + row] = X_i_e[0] - X_i_o[0];

	o_r[144 + row] = X_r_e[1] - tx1;
	o_i[144 + row] = X_i_e[1] - tx2;

	o_r[160 + row] = X_r_e[2] - tx3;
	o_i[160 + row] = X_i_e[2] - tx4;

	o_r[176 + row] = X_r_e[3] - tx5;
	o_i[176 + row] = X_i_e[3] - tx6;

	o_r[192 + row] = X_r_e[4] + X_i_o[4];
	o_i[192 + row] = X_i_e[4] - X_r_o[4];

	o_r[208 + row] = X_r_e[5] + tx7;
	o_i[208 + row] = X_i_e[5] - tx8;

	o_r[224 + row] = X_r_e[6] + tx9;
	o_i[224 + row] = X_i_e[6] + tx10;

	o_r[240 + row] = X_r_e[7] + tx11;
	o_i[240 + row] = X_i_e[7] - tx12;
}


static void ifftx16_1D_sym_final(float *in_r, float *in_i, int row, float *o_r)
{
	float X_r_e[8], X_r_o[8], X_i_o[8], tx1, tx3, tx5, tx7, tx9, tx11;


	ifftx16_8_1D_sym_e(in_r, in_i, X_r_e);
	ifftx16_8_1D_sym_o(in_r, in_i, X_r_o, X_i_o);

	o_r[row] = X_r_e[0] + X_r_o[0];

	tx1 = 0.923879532511287F*X_r_o[1] - 0.382683432365090F*X_i_o[1];

	o_r[16 + row] = X_r_e[1] + tx1;

	tx3 = 0.707106781186548F*(X_r_o[2] - X_i_o[2]);

	o_r[32 + row] = X_r_e[2] + tx3;

	tx5 = 0.382683432365090F*X_r_o[3] - 0.923879532511287F*X_i_o[3];

	o_r[48 + row] = X_r_e[3] + tx5;

	o_r[64 + row] = X_r_e[4] - X_i_o[4];

	tx7 = 0.382683432365090F*X_r_o[5] + 0.923879532511287F*X_i_o[5];

	o_r[80 + row] = X_r_e[5] - tx7;

	tx9 = 0.707106781186548F*(X_i_o[6] + X_r_o[6]);

	o_r[96 + row] = X_r_e[6] - tx9;

	tx11 = 0.923879532511287F*X_r_o[7] + 0.382683432365090F*X_i_o[7];

	o_r[112 + row] = X_r_e[7] - tx11;
	o_r[128 + row] = X_r_e[0] - X_r_o[0];
	o_r[144 + row] = X_r_e[1] - tx1;
	o_r[160 + row] = X_r_e[2] - tx3;
	o_r[176 + row] = X_r_e[3] - tx5;
	o_r[192 + row] = X_r_e[4] + X_i_o[4];
	o_r[208 + row] = X_r_e[5] + tx7;
	o_r[224 + row] = X_r_e[6] + tx9;
	o_r[240 + row] = X_r_e[7] + tx11;
}

void fft2d::ifftx16_2DNS(float *in_r, float *in_i) // ifft2d no scale
{
	float o_r_temp[256], o_i_temp[256];
	int k, j;

	ifftx16_1D_sym(&in_r[0], &in_i[0], 0, o_r_temp, o_i_temp);

	j = 16;
	for (k = 1; k<16; k++)
	{
		ifftx16_1D_asym(&in_r[j], &in_i[j], k, o_r_temp, o_i_temp);
		j += 16;
	}

	j = 0;
	for (k = 0; k<16; k++)
	{
		ifftx16_1D_sym_final(&o_r_temp[j], &o_i_temp[j], k, in_r);
		j += 16;
	}
}

void fft2d::ifftx16Complex(float *in_r, float *in_i)
{
	float o_r_temp[256], o_i_temp[256];
	int j;

	j = 0;
	for (int k = 0; k<16; k++)
	{
		ifftx16_1D_asym(&in_r[j], &in_i[j], k, o_r_temp, o_i_temp);
		j += 16;
	}

	j = 0;
	for (int k = 0; k<16; k++)
	{
		ifftx16_1D_asym(&o_r_temp[j], &o_i_temp[j], k, in_r, in_i);
		j += 16;
	}
}

void fft2d::fft16Complex(float *horz_fft_r, float *horz_fft_i, float *fft_r, float *fft_i)
{
	float vec_r[16], vec_i[16];

	for (int k = 0; k<16; k++)
	{

		for (int j = 0; j<16; j++)
		{
			int idx = j * 16 + k;
			vec_r[j] = horz_fft_r[idx];
			vec_i[j] = horz_fft_i[idx];
		}

		int col = k * 16;
		fftx16_1D_complex_horz(vec_r, vec_i, fft_r + col, fft_i + col);
	}

}

static void fftx8_1D_complex_horz(float *in_r, float *in_i, float *o_r, float *o_i)
{

	float X_r_e[4], X_i_e[4], X_r_o[4], X_i_o[4], tx1, tx2, tx3, tx4;
	float x1, x2, x1_i, x2_i, CC1_r, CC1_i, CC2_r, CC2_i;

	// **********************************************

	x1 = (in_r[0] + in_r[4]);
	x2 = (in_r[2] + in_r[6]);

	x1_i = (in_i[0] + in_i[4]);
	x2_i = (in_i[2] + in_i[6]);

	X_r_e[0] = x1 + x2;
	X_i_e[0] = x1_i + x2_i;

	CC1_r = (in_r[0] - in_r[4]);
	CC1_i = (in_i[0] - in_i[4]);

	CC2_r = (in_r[6] - in_r[2]);
	CC2_i = (in_i[6] - in_i[2]);

	X_r_e[1] = CC1_r - CC2_i;
	X_i_e[1] = CC2_r + CC1_i;

	X_r_e[2] = x1 - x2;
	X_i_e[2] = x1_i - x2_i;

	X_r_e[3] = CC1_r + CC2_i;
	X_i_e[3] = -CC2_r + CC1_i;

	// **********************************************

	x1 = (in_r[1] + in_r[5]);
	x2 = (in_r[3] + in_r[7]);

	x1_i = (in_i[1] + in_i[5]);
	x2_i = (in_i[3] + in_i[7]);

	X_r_o[0] = x1 + x2;
	X_i_o[0] = x1_i + x2_i;

	CC1_r = (in_r[1] - in_r[5]);
	CC1_i = (in_i[1] - in_i[5]);

	CC2_r = (in_r[7] - in_r[3]);
	CC2_i = (in_i[7] - in_i[3]);

	X_r_o[1] = CC1_r - CC2_i;
	X_i_o[1] = CC2_r + CC1_i;

	X_r_o[2] = x1 - x2;
	X_i_o[2] = x1_i - x2_i;

	X_r_o[3] = CC1_r + CC2_i;
	X_i_o[3] = -CC2_r + CC1_i;

	// **********************************************

	*o_r++ = X_r_e[0] + X_r_o[0];
	*o_i++ = X_i_e[0] + X_i_o[0];

	tx1 = (0.707106781186548f)*(X_r_o[1] + X_i_o[1]);
	tx2 = (0.707106781186548f)*(X_i_o[1] - X_r_o[1]);


	*o_r++ = X_r_e[1] + tx1;
	*o_i++ = X_i_e[1] + tx2;

	*o_r++ = X_r_e[2] + X_i_o[2];
	*o_i++ = X_i_e[2] - X_r_o[2];

	tx3 = (0.707106781186548f)*(X_i_o[3] - X_r_o[3]);
	tx4 = (0.707106781186548f)*(X_i_o[3] + X_r_o[3]);

	*o_r++ = X_r_e[3] + tx3;
	*o_i++ = X_i_e[3] - tx4;

	*o_r++ = X_r_e[0] - X_r_o[0];
	*o_i++ = X_i_e[0] - X_i_o[0];

	*o_r++ = X_r_e[1] - tx1;
	*o_i++ = X_i_e[1] - tx2;

	*o_r++ = X_r_e[2] - X_i_o[2];
	*o_i++ = X_i_e[2] + X_r_o[2];

	*o_r++ = X_r_e[3] - tx3;
	*o_i++ = X_i_e[3] + tx4;

}


void fft2d::fft8_2d_v(float *horz_fft_r, float *horz_fft_i, float *fft_r, float *fft_i)
{

	const int ridx[24] = { 24, 31, 30, 29, 28, 27, 26, 25, 16, 23, 22, 21, 20, 19, 18, 17, 8, 15, 14, 13, 12, 11, 10, 9};

	float vec_r[8], vec_i[8];

	for (int j = 0; j<8; j++)
	{
		int idx = j * 8;
		vec_r[j] = horz_fft_r[idx];
	}

	fftx8_1D_realT(vec_r, fft_r, fft_i);

	for (int k = 1; k < 4; k++)
	{
		for (int j = 0; j<8; j++)
		{
			int idx = j * 8 + k;
			vec_r[j] = horz_fft_r[idx];
			vec_i[j] = horz_fft_i[idx];
		}

		fftx8_1D_complex_horz(vec_r, vec_i, fft_r + k * 8, fft_i + k * 8);
	}

	for (int j = 0; j<8; j++)
	{
		int idx = j * 8 + 4;
		vec_r[j] = horz_fft_r[idx];
	}

	fftx8_1D_realT(vec_r, fft_r + 32, fft_i + 32);

	for (int k = 40; k<64; k++)
	{
		int r = ridx[k-40];
		fft_r[k] = fft_r[r];
		fft_i[k] = -fft_i[r];
	}
}

void fft2d::fft16Full(float *horz_fft_r, float *horz_fft_i, float *fft_r, float *fft_i)
{
	float vec_r[16], vec_i[16];

	const int ridx[112] = { 112, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 96, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97, 80, 95, 94, 93, 92, 91, 90
		, 89, 88, 87, 86, 85, 84, 83, 82, 81, 64, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 48, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51
		, 50, 49, 32, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 16, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17 };


	for (int j = 0; j<16; j++)
	{
		vec_r[j] = horz_fft_r[j * 16];
	}
	fftx16_1D_real_horz(vec_r, fft_r, fft_i);


	for (int k = 1; k<8; k++)
	{

		for (int j = 0; j<16; j++)
		{
			int idx = j * 16 + k;
			vec_r[j] = horz_fft_r[idx];
			vec_i[j] = horz_fft_i[idx];
		}

		int col = k * 16;
		fftx16_1D_complex_horz(vec_r, vec_i, fft_r + col, fft_i + col);
	}

	for (int j = 0; j<16; j++)
	{
		vec_r[j] = horz_fft_r[j * 16 + 8];
	}

	fftx16_1D_real_horz(vec_r, fft_r + 128, fft_i + 128);

	float *por = fft_r + 144;
	float *poi = fft_i + 144;
	for (int k = 0; k<112; k++)
	{
		int r = ridx[k];
		por[k] = fft_r[r];
		poi[k] = -fft_i[r];
	}

}

/*void wrFilterTools::STFT16(
wrBuffer* imgInOut, const wrBuffer* imgorg, const wrBuffer* grec8x8,
wrBuffer* dummy1a, wrBuffer* dummy1b, wrBuffer* dummy1c,
int rows, int cols, bool gpu)
{
if (!HaveOpenCL())
gpu = false;
if (gpu)
{
m_STFT16->STFT16(
imgInOut->clidRW(), imgorg->clidR(), grec8x8->clidR(),
dummy1a->clidW(), dummy1b->clidW(), dummy1c->clidW(), rows, cols);
}
else
{
stft16restdiff(
imgInOut->DataRW<float>(), imgorg->DataR<float>(), grec8x8->DataR<float>(),
dummy1a->DataW<float>(), rows, cols);
}
}*/


static void fftx8_1D_real(float *in_r, int row, float *o_r, float *o_i)
{

	float X_r_e[4], X_i_e[4], X_r_o[4], X_i_o[4];
	float x1, x2;

	// *********************
	x1 = (in_r[0] + in_r[4]);
	x2 = (in_r[2] + in_r[6]);

	X_r_e[0] = x1 + x2;
	X_i_e[0] = 0;

	X_r_e[1] = (in_r[0] - in_r[4]);
	X_i_e[1] = (in_r[6] - in_r[2]);

	X_r_e[2] = x1 - x2;
	X_i_e[2] = 0;

	X_r_e[3] = X_r_e[1];
	X_i_e[3] = -X_i_e[1];

	// **************************

	x1 = (in_r[1] + in_r[5]);
	x2 = (in_r[3] + in_r[7]);

	X_r_o[0] = x1 + x2;
	X_i_o[0] = 0;

	X_r_o[1] = (in_r[1] - in_r[5]);
	X_i_o[1] = (in_r[7] - in_r[3]);

	X_r_o[2] = x1 - x2;
	X_i_o[2] = 0;

	X_r_o[3] = X_r_o[1];
	X_i_o[3] = -X_i_o[1];

	// **************************

	o_r[row] = X_r_e[0] + X_r_o[0];
	o_i[row] = 0;

	o_r[8 + row] = X_r_e[1] + (0.707106781186548f)*(X_r_o[1] + X_i_o[1]);
	o_i[8 + row] = X_i_e[1] + (0.707106781186548f)*(X_i_o[1] - X_r_o[1]);

	o_r[16 + row] = X_r_e[2] + X_i_o[2];
	o_i[16 + row] = X_i_e[2] - X_r_o[2];

	o_r[24 + row] = X_r_e[3] + (0.707106781186548f)*(X_i_o[3] - X_r_o[3]);
	o_i[24 + row] = X_i_e[3] - (0.707106781186548f)*(X_i_o[3] + X_r_o[3]);

	o_r[32 + row] = X_r_e[0] - X_r_o[0];
	o_i[32 + row] = 0;

	o_r[40 + row] = o_r[24 + row];
	o_i[40 + row] = -o_i[24 + row];

	o_r[48 + row] = o_r[16 + row];
	o_i[48 + row] = -o_i[16 + row];

	o_r[56 + row] = o_r[8 + row];
	o_i[56 + row] = -o_i[8 + row];

}


static void fftx8_1D_complex(float *in_r, float *in_i, int row, float *o_r, float *o_i)
{

	float X_r_e[4], X_i_e[4], X_r_o[4], X_i_o[4], tx1, tx2, tx3, tx4;
	float x1, x2, x1_i, x2_i, CC1_r, CC1_i, CC2_r, CC2_i;

	// **********************************************

	x1 = (in_r[0] + in_r[4]);
	x2 = (in_r[2] + in_r[6]);

	x1_i = (in_i[0] + in_i[4]);
	x2_i = (in_i[2] + in_i[6]);

	X_r_e[0] = x1 + x2;
	X_i_e[0] = x1_i + x2_i;

	CC1_r = (in_r[0] - in_r[4]);
	CC1_i = (in_i[0] - in_i[4]);

	CC2_r = (in_r[6] - in_r[2]);
	CC2_i = (in_i[6] - in_i[2]);

	X_r_e[1] = CC1_r - CC2_i;
	X_i_e[1] = CC2_r + CC1_i;

	X_r_e[2] = x1 - x2;
	X_i_e[2] = x1_i - x2_i;

	X_r_e[3] = CC1_r + CC2_i;
	X_i_e[3] = -CC2_r + CC1_i;

	// **********************************************

	x1 = (in_r[1] + in_r[5]);
	x2 = (in_r[3] + in_r[7]);

	x1_i = (in_i[1] + in_i[5]);
	x2_i = (in_i[3] + in_i[7]);

	X_r_o[0] = x1 + x2;
	X_i_o[0] = x1_i + x2_i;

	CC1_r = (in_r[1] - in_r[5]);
	CC1_i = (in_i[1] - in_i[5]);

	CC2_r = (in_r[7] - in_r[3]);
	CC2_i = (in_i[7] - in_i[3]);

	X_r_o[1] = CC1_r - CC2_i;
	X_i_o[1] = CC2_r + CC1_i;

	X_r_o[2] = x1 - x2;
	X_i_o[2] = x1_i - x2_i;

	X_r_o[3] = CC1_r + CC2_i;
	X_i_o[3] = -CC2_r + CC1_i;

	// **********************************************

	o_r[row] = X_r_e[0] + X_r_o[0];
	o_i[row] = X_i_e[0] + X_i_o[0];

	tx1 = (0.707106781186548f)*(X_r_o[1] + X_i_o[1]);
	tx2 = (0.707106781186548f)*(X_i_o[1] - X_r_o[1]);


	o_r[8 + row] = X_r_e[1] + tx1;
	o_i[8 + row] = X_i_e[1] + tx2;

	o_r[16 + row] = X_r_e[2] + X_i_o[2];
	o_i[16 + row] = X_i_e[2] - X_r_o[2];

	tx3 = (0.707106781186548f)*(X_i_o[3] - X_r_o[3]);
	tx4 = (0.707106781186548f)*(X_i_o[3] + X_r_o[3]);

	o_r[24 + row] = X_r_e[3] + tx3;
	o_i[24 + row] = X_i_e[3] - tx4;

	o_r[32 + row] = X_r_e[0] - X_r_o[0];
	o_i[32 + row] = X_i_e[0] - X_i_o[0];

	o_r[40 + row] = X_r_e[1] - tx1;
	o_i[40 + row] = X_i_e[1] - tx2;

	o_r[48 + row] = X_r_e[2] - X_i_o[2];
	o_i[48 + row] = X_i_e[2] + X_r_o[2];

	o_r[56 + row] = X_r_e[3] - tx3;
	o_i[56 + row] = X_i_e[3] + tx4;

}

void fft2d::fftx8_2D(float *in_r, float *in_i)
{

	float o_r_temp[64], o_i_temp[64];

	const int tidx[24] = { 5, 6, 7, 13, 14, 15, 21, 22, 23, 29, 30, 31, 37, 38, 39, 45, 46, 47, 53, 54, 55, 61, 62, 63 };
	const int ridx[24] = { 3, 2, 1, 59, 58, 57, 51, 50, 49, 43, 42, 41, 35, 34, 33, 27, 26, 25, 19, 18, 17, 11, 10, 9 };

	for (int k = 0; k<8; k++)
		fftx8_1D_real(in_r + k * 8, k, o_r_temp, o_i_temp);


	fftx8_1D_real(&o_r_temp[0], 0, in_r, in_i);
	fftx8_1D_complex(&o_r_temp[8], &o_i_temp[8], 1, in_r, in_i);
	fftx8_1D_complex(&o_r_temp[16], &o_i_temp[16], 2, in_r, in_i);
	fftx8_1D_complex(&o_r_temp[24], &o_i_temp[24], 3, in_r, in_i);
	fftx8_1D_real(&o_r_temp[32], 4, in_r, in_i);

	for (int k = 0; k<24; k++)
	{
		int t = tidx[k];
		int r = ridx[k];
		in_r[t] = in_r[r];
		in_i[t] = -in_i[r];
	}
}


static void ifftx8_1D_symV(float *in_r, float *in_i, float *o_r, float *o_i)
{

	float X_r_e[4], X_i_e[4], X_r_o[4], X_i_o[4], tx1, tx3;
	float x1, x2, x3, x4;

	// *********************************************

	x1 = (in_r[0] + in_r[4]);
	x2 = (in_r[2] + in_r[6]);

	x3 = (in_r[0] - in_r[4]);
	x4 = (in_i[2] - in_i[6]);

	X_r_e[0] = x1 + x2;
	X_i_e[0] = 0;

	X_r_e[1] = x3 - x4;
	X_i_e[1] = 0;

	X_r_e[2] = x1 - x2;
	X_i_e[2] = 0;

	X_r_e[3] = x3 + x4;
	X_i_e[3] = 0;

	// *********************************************

	float x1_i, x2_i, CC1_r, CC2_i;

	x1 = (in_r[1] + in_r[5]);
	x2 = (in_r[3] + in_r[7]);

	x1_i = (in_i[1] + in_i[5]);
	x2_i = (in_i[3] + in_i[7]);

	X_r_o[0] = x1 + x2;
	X_i_o[0] = 0;

	CC1_r = (in_r[1] - in_r[5]);
	CC2_i = (in_i[7] - in_i[3]);

	X_r_o[1] = CC1_r + CC2_i;
	X_i_o[1] = -X_r_o[1];

	X_r_o[2] = 0;
	X_i_o[2] = x1_i - x2_i;

	X_r_o[3] = CC1_r - CC2_i;
	X_i_o[3] = X_r_o[3];


	// *********************************************
	*o_r++ = X_r_e[0] + X_r_o[0];
	*o_i++ = 0;

	tx1 = (0.707106781186548f)*(X_r_o[1] - X_i_o[1]);

	*o_r++ = X_r_e[1] + tx1;
	*o_i++ = 0;

	*o_r++ = X_r_e[2] - X_i_o[2];
	*o_i++ = 0;

	tx3 = (0.707106781186548f)*(X_i_o[3] + X_r_o[3]);

	*o_r++ = X_r_e[3] - tx3;
	*o_i++ = 0;

	*o_r++ = X_r_e[0] - X_r_o[0];
	*o_i++ = 0;

	*o_r++ = X_r_e[1] - tx1;
	*o_i++ = 0;

	*o_r++ = X_r_e[2] + X_i_o[2];
	*o_i++ = 0;

	*o_r++ = X_r_e[3] + tx3;
	*o_i++ = 0;
}


static void ifftx8_1D_sym(float *in_r, float *in_i, int row, float *o_r, float *o_i)
{

	float X_r_e[4], X_i_e[4], X_r_o[4], X_i_o[4], tx1, tx3;
	float x1, x2, x3, x4;

	// *********************************************

	x1 = (in_r[0] + in_r[4]);
	x2 = (in_r[2] + in_r[6]);

	x3 = (in_r[0] - in_r[4]);
	x4 = (in_i[2] - in_i[6]);

	X_r_e[0] = x1 + x2;
	X_i_e[0] = 0;

	X_r_e[1] = x3 - x4;
	X_i_e[1] = 0;

	X_r_e[2] = x1 - x2;
	X_i_e[2] = 0;

	X_r_e[3] = x3 + x4;
	X_i_e[3] = 0;

	// *********************************************

	float x1_i, x2_i, CC1_r, CC2_i;

	x1 = (in_r[1] + in_r[5]);
	x2 = (in_r[3] + in_r[7]);

	x1_i = (in_i[1] + in_i[5]);
	x2_i = (in_i[3] + in_i[7]);

	X_r_o[0] = x1 + x2;
	X_i_o[0] = 0;

	CC1_r = (in_r[1] - in_r[5]);
	CC2_i = (in_i[7] - in_i[3]);

	X_r_o[1] = CC1_r + CC2_i;
	X_i_o[1] = -X_r_o[1];

	X_r_o[2] = 0;
	X_i_o[2] = x1_i - x2_i;

	X_r_o[3] = CC1_r - CC2_i;
	X_i_o[3] = X_r_o[3];


	// *********************************************
	o_r[row] = X_r_e[0] + X_r_o[0];
	o_i[row] = 0;

	tx1 = (0.707106781186548f)*(X_r_o[1] - X_i_o[1]);

	o_r[8 + row] = X_r_e[1] + tx1;
	o_i[8 + row] = 0;

	o_r[16 + row] = X_r_e[2] - X_i_o[2];
	o_i[16 + row] = 0;

	tx3 = (0.707106781186548f)*(X_i_o[3] + X_r_o[3]);

	o_r[24 + row] = X_r_e[3] - tx3;
	o_i[24 + row] = 0;

	o_r[32 + row] = X_r_e[0] - X_r_o[0];
	o_i[32 + row] = 0;

	o_r[40 + row] = X_r_e[1] - tx1;
	o_i[40 + row] = 0;

	o_r[48 + row] = X_r_e[2] + X_i_o[2];
	o_i[48 + row] = 0;

	o_r[56 + row] = X_r_e[3] + tx3;
	o_i[56 + row] = 0;
}

static void ifftx8_1D_asym(float *in_r, float *in_i, int row, float *o_r, float *o_i)
{

	float X_r_e[4], X_i_e[4], X_r_o[4], X_i_o[4], tx1, tx2, tx3, tx4;

	// *********************************************
	float x1, x2, x1_i, x2_i, CC1_r, CC1_i, CC2_r, CC2_i;

	x1 = (in_r[0] + in_r[4]);
	x2 = (in_r[2] + in_r[6]);

	x1_i = (in_i[0] + in_i[4]);
	x2_i = (in_i[2] + in_i[6]);

	X_r_e[0] = x1 + x2;
	X_i_e[0] = x1_i + x2_i;

	CC1_r = (in_r[0] - in_r[4]);
	CC1_i = (in_i[0] - in_i[4]);

	CC2_r = (in_r[6] - in_r[2]);
	CC2_i = (in_i[6] - in_i[2]);

	X_r_e[1] = CC1_r + CC2_i;
	X_i_e[1] = -CC2_r + CC1_i;

	X_r_e[2] = x1 - x2;
	X_i_e[2] = x1_i - x2_i;

	X_r_e[3] = CC1_r - CC2_i;
	X_i_e[3] = CC2_r + CC1_i;
	// *********************************************

	x1 = (in_r[1] + in_r[5]);
	x2 = (in_r[3] + in_r[7]);

	x1_i = (in_i[1] + in_i[5]);
	x2_i = (in_i[3] + in_i[7]);

	X_r_o[0] = x1 + x2;
	X_i_o[0] = x1_i + x2_i;

	CC1_r = (in_r[1] - in_r[5]);
	CC1_i = (in_i[1] - in_i[5]);

	CC2_r = (in_r[7] - in_r[3]);
	CC2_i = (in_i[7] - in_i[3]);

	X_r_o[1] = CC1_r + CC2_i;
	X_i_o[1] = -CC2_r + CC1_i;

	X_r_o[2] = x1 - x2;
	X_i_o[2] = x1_i - x2_i;

	X_r_o[3] = CC1_r - CC2_i;
	X_i_o[3] = CC2_r + CC1_i;
	// *********************************************

	o_r[row] = X_r_e[0] + X_r_o[0];
	o_i[row] = X_i_e[0] + X_i_o[0];

	tx1 = (0.707106781186548f)*(X_r_o[1] - X_i_o[1]);
	tx2 = (0.707106781186548f)*(X_i_o[1] + X_r_o[1]);

	o_r[8 + row] = X_r_e[1] + tx1;
	o_i[8 + row] = X_i_e[1] + tx2;

	o_r[16 + row] = X_r_e[2] - X_i_o[2];
	o_i[16 + row] = X_i_e[2] + X_r_o[2];

	tx3 = (0.707106781186548f)*(X_i_o[3] + X_r_o[3]);
	tx4 = (0.707106781186548f)*(X_i_o[3] - X_r_o[3]);

	o_r[24 + row] = X_r_e[3] - tx3;
	o_i[24 + row] = X_i_e[3] - tx4;

	o_r[32 + row] = X_r_e[0] - X_r_o[0];
	o_i[32 + row] = X_i_e[0] - X_i_o[0];

	o_r[40 + row] = X_r_e[1] - tx1;
	o_i[40 + row] = X_i_e[1] - tx2;

	o_r[48 + row] = X_r_e[2] + X_i_o[2];
	o_i[48 + row] = X_i_e[2] - X_r_o[2];

	o_r[56 + row] = X_r_e[3] + tx3;
	o_i[56 + row] = X_i_e[3] + tx4;
}

void fft2d::ifftx8_2D(float *in_r, float *in_i)
{
	float o_r_temp[64], o_i_temp[64];

	ifftx8_1D_sym(&in_r[0], &in_i[0], 0, o_r_temp, o_i_temp);
	ifftx8_1D_asym(&in_r[8], &in_i[8], 1, o_r_temp, o_i_temp);
	ifftx8_1D_asym(&in_r[16], &in_i[16], 2, o_r_temp, o_i_temp);
	ifftx8_1D_asym(&in_r[24], &in_i[24], 3, o_r_temp, o_i_temp);
	ifftx8_1D_asym(&in_r[32], &in_i[32], 4, o_r_temp, o_i_temp);
	ifftx8_1D_asym(&in_r[40], &in_i[40], 5, o_r_temp, o_i_temp);
	ifftx8_1D_asym(&in_r[48], &in_i[48], 6, o_r_temp, o_i_temp);
	ifftx8_1D_asym(&in_r[56], &in_i[56], 7, o_r_temp, o_i_temp);



	ifftx8_1D_sym(&o_r_temp[0], &o_i_temp[0], 0, in_r, in_i);
	ifftx8_1D_sym(&o_r_temp[8], &o_i_temp[8], 1, in_r, in_i);
	ifftx8_1D_sym(&o_r_temp[16], &o_i_temp[16], 2, in_r, in_i);
	ifftx8_1D_sym(&o_r_temp[24], &o_i_temp[24], 3, in_r, in_i);
	ifftx8_1D_sym(&o_r_temp[32], &o_i_temp[32], 4, in_r, in_i);
	ifftx8_1D_sym(&o_r_temp[40], &o_i_temp[40], 5, in_r, in_i);
	ifftx8_1D_sym(&o_r_temp[48], &o_i_temp[48], 6, in_r, in_i);
	ifftx8_1D_sym(&o_r_temp[56], &o_i_temp[56], 7, in_r, in_i);

	/*for (k = 0; k<64; k++)
	{
		in_r[k] = in_r[k] / 64;
		in_i[k] = in_i[k] / 64;
	}*/

}

void fft2d::ifftx8_2DV(float *in_r, float *in_i)
{
	float o_r_temp[64], o_i_temp[64];
	int k;

	ifftx8_1D_sym(&in_r[0], &in_i[0], 0, o_r_temp, o_i_temp);
	ifftx8_1D_asym(&in_r[8], &in_i[8], 1, o_r_temp, o_i_temp);
	ifftx8_1D_asym(&in_r[16], &in_i[16], 2, o_r_temp, o_i_temp);
	ifftx8_1D_asym(&in_r[24], &in_i[24], 3, o_r_temp, o_i_temp);
	ifftx8_1D_asym(&in_r[32], &in_i[32], 4, o_r_temp, o_i_temp);
	ifftx8_1D_asym(&in_r[40], &in_i[40], 5, o_r_temp, o_i_temp);
	ifftx8_1D_asym(&in_r[48], &in_i[48], 6, o_r_temp, o_i_temp);
	ifftx8_1D_asym(&in_r[56], &in_i[56], 7, o_r_temp, o_i_temp);


	for (k = 0; k<64; k+=8)
		ifftx8_1D_symV(&o_r_temp[k], &o_i_temp[k], in_r + k, in_i + k);
	

	for (k = 0; k<64; k++)
	{
		in_r[k] = in_r[k] / 64;
		in_i[k] = in_i[k] / 64;
	}

}
