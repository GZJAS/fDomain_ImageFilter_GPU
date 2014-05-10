/*
 * 	dft2D_kernel.cu - GPU kernel definition
 *
 *  Created on: May 2, 2014
 *  Author: 	Evan Hosseini
 *  Class:		CS708
 */
#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_string.h>

#include "dft2D.h"

// Complex data type
typedef float2 Complex;


// Complex addition
static __device__ inline Complex devComplexAdd(Complex a, Complex b)
{
    Complex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

// Complex scale
static __device__ inline Complex devComplexScale(Complex a, float s)
{
    Complex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

// Complex multiplication
static __device__ inline Complex devComplexMul(Complex a, Complex b)
{
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

// Complex magnitude
static __device__ inline unsigned char devComplexMag(Complex *a)
{
    float c;
    c = a->x * a->x + a->y * a->y;
    c = sqrt( c );
    return (unsigned char) c;
}

/* ***************************************************************/
// This kernel calculates a one dimensional discrete Fourier transform
//! @param g_pInput			Input image pointer
//! @param g_pXfmCoeffs 	Fourier transform coefficients
//! @param g_pTempMatrix	1D DFT/IDFT transformation
//! @param g_p2D_xfm		Complex Output 2D Fourier transformation
//! @param isIDFT			DFT/IDFT option
//! @param isRowDFT			Row-wise/Column-wise DFT option
//! @param g_filteredImg	IDFT conversion to unsigned char

//  NOTE: Kernel assumes that the client calls w/ the row-wise
//        DFT first, followed by column-wise DFT
/* ***************************************************************/
__global__ void dft1DKernel( unsigned char *g_pInput,
			 	 	 	 	 Complex *g_pXfmCoeffs,
			 	 	 	 	 Complex *g_pTempMatrix,
			 	 	 	 	 Complex *g_p2D_xfm,
			 	 	 	 	 bool isIDFT,
			 	 	 	 	 bool isRowDFT,
			 	 	 	 	 unsigned char *g_pImgOut )
{
	// Calculate global index from thread and block indices
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	// Calculate thread 2D index pointer offset
	int ptrOffset = y * N + x;

	// Declare thread local buffers for the multiplication inputs
	float fIn[N];
	Complex zTwiddle[N];
	Complex zTemp[N];

	// Capture local iterators
	unsigned char *m_pInput = g_pInput;
	Complex *m_pXfmCoeffs = g_pXfmCoeffs;
	Complex *m_pTempMatrix = g_pTempMatrix;
	Complex *m_p2D_xfm = g_p2D_xfm;
	unsigned char *m_pImgOut = g_pImgOut;

	// Need to differentiate between the row-wise and column-wise 1D DFT
	// and DFT vs. IDFT
	if ( isRowDFT && !isIDFT )
	{
		// Generate appropriate starting ptr offsets
		m_pInput += y * N;
		m_pXfmCoeffs += x;

		// Assign appropriate values from global matrices to local buffers
		for ( int i = 0; i < N; ++i, ++m_pInput )
		{
			fIn[i] = (float)*m_pInput;
			zTwiddle[i] = *m_pXfmCoeffs;
			m_pXfmCoeffs += N;
		}

		// Multiply and store sub-result in temp matrix
		m_pTempMatrix += ptrOffset;
		m_pTempMatrix->x = 0; m_pTempMatrix->y = 0;
		for ( int i = 0; i < N; ++i )
			*m_pTempMatrix = devComplexAdd( *m_pTempMatrix, devComplexScale( zTwiddle[i], fIn[i] ) );
	}
	else if( !isRowDFT && !isIDFT )
	{
		// Generate appropriate starting ptr offsets
		m_pTempMatrix += x;
		m_pXfmCoeffs += y * N;

		// Assign appropriate values from global matrices to local buffers
		for ( int i = 0; i < N; ++i, ++m_pXfmCoeffs )
		{
			zTemp[i] = *m_pTempMatrix;
			zTwiddle[i] = *m_pXfmCoeffs;
			m_pTempMatrix += N;
		}

		// Multiply and store sub-result in xfm matrix
		m_p2D_xfm += ptrOffset;
		m_p2D_xfm->x = 0; m_p2D_xfm->y = 0;
		for ( int i = 0; i < N; ++i )
			*m_p2D_xfm = devComplexAdd( *m_p2D_xfm, devComplexMul( zTwiddle[i], zTemp[i] ) );
	}
	else if( isRowDFT && isIDFT )
	{
		// Generate appropriate starting ptr offsets
		m_pTempMatrix += y * N;
		m_pXfmCoeffs += x;

		// Assign appropriate values from global matrices to local buffers
		for ( int i = 0; i < N; ++i, ++m_pTempMatrix )
		{
			zTemp[i] = *m_pTempMatrix;
			zTwiddle[i] = *m_pXfmCoeffs;
			m_pXfmCoeffs += N;
		}

		// Multiply and store sub-result in xfm matrix
		m_p2D_xfm += ptrOffset;
		m_p2D_xfm->x = 0; m_p2D_xfm->y = 0;
		for ( int i = 0; i < N; ++i )
			*m_p2D_xfm = devComplexAdd( *m_p2D_xfm, devComplexMul( zTwiddle[i], zTemp[i] ) );
	}
	else if ( !isRowDFT && isIDFT )
	{
		// Generate appropriate starting ptr offsets
		m_p2D_xfm += x;
		m_pXfmCoeffs += y * N;

		// Assign appropriate values from global matrices to local buffers
		for ( int i = 0; i < N; ++i, ++m_pXfmCoeffs )
		{
			zTemp[i] = *m_p2D_xfm;
			zTwiddle[i] = *m_pXfmCoeffs;
			m_p2D_xfm += N;
		}

		// Multiply and store sub-result in temp matrix
		m_pTempMatrix += ptrOffset;
		m_pTempMatrix->x = 0; m_pTempMatrix->y = 0;
		for ( int i = 0; i < N; ++i )
			*m_pTempMatrix = devComplexAdd( *m_pTempMatrix , devComplexMul( zTwiddle[i], zTemp[i] ) );

		// Write complex magnitude to result buffer
		m_pTempMatrix = g_pTempMatrix;
		m_pTempMatrix += ptrOffset;
		m_pImgOut += ptrOffset;
		*m_pImgOut = devComplexMag( m_pTempMatrix );
	}
}

/* ***************************************************************/
// This kernel applies a frequency domain low/high pass filter to
// a two dimensional spectrum matrix
//! @param g_pIn 		2D Image spectral composition
//! @param g_pCoeffs	Filter coefficient matrix
//! @param g_pOut		Filtered 2D image spectral composition
//! @param isLowPass	lowpass/highpass filter option
/* ***************************************************************/
__global__ void filterKernel( Complex *g_pIn,
			 	 	 	 	  float *g_pCoeffs,
			 	 	 	 	  Complex *g_pOut )
{
	// Calculate global index from thread and block indices
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	// Calculate thread 2D index pointer offset
	int ptrOffset = y * N + x;

	// Capture local iterators
	Complex *m_pInput = g_pIn;
	float *m_pCoeffs = g_pCoeffs;
	Complex *m_pOut = g_pOut;

	// Generate appropriate starting ptr offsets
	m_pInput += ptrOffset;
	m_pCoeffs += ptrOffset;

	// Multiply and store sub-result in output matrix
	m_pOut += ptrOffset;
	m_pOut->x = 0; m_pOut->y = 0;
	for ( int i = 0; i < N; ++i )
		*m_pOut = devComplexScale( *m_pInput, *m_pCoeffs );
}


// Kernel Wrappers
void dispatchDFTkernel(unsigned char* g_pInput,
					   Complex *g_pXfmCoeffs,
					   Complex *g_pTempMatrix,
					   Complex *g_p2D_xfm,
					   bool isIDFT,
					   bool isRowDFT,
					   unsigned char *g_pImgOut,
					   dim3 nBlocks,
					   dim3 nThreads)
{
	dft1DKernel<<< nBlocks, nThreads >>>( g_pInput, g_pXfmCoeffs, g_pTempMatrix, g_p2D_xfm, isIDFT, isRowDFT, g_pImgOut );
}

void dispatchFilterKernel(Complex *g_pIn,
					   	  float *g_pCoeffs,
					   	  Complex *g_pOut,
					   	  dim3 nBlocks,
					   	  dim3 nThreads)
{
	filterKernel<<< nBlocks, nThreads >>>( g_pIn, g_pCoeffs, g_pOut );
}

