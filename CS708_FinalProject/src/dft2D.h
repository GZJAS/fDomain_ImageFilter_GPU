/*
 * 	dft2D.h - Helper functions and forward declarations
 *
 *  Created on: May 1, 2014
 *  Author: 	Evan Hosseini
 *  Class:  	CS708
 */

#ifndef DFT2D_H_
#define DFT2D_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <cuda_runtime.h>
#include <helper_functions.h> // includes for SDK helper functions
#include <helper_cuda.h>      // includes for cuda initialization and error checking
#include <helper_string.h>

#define blkSize_x 32
#define blkSize_y 8

#define N 512
#define W_C 0.3

// Pixel definition
typedef unsigned char Pixel;

// Complex definition
typedef float2 Complex;

// Forward declarations
static inline Complex ComplexAdd(Complex, Complex);
static inline Complex ComplexScale(Complex, float);
static inline Complex ComplexMul(Complex, Complex);
void initializeData(char *file);
void loadDefaultImage(char *loc_exec);
void twiddleMatrixGen(int w, int h, bool isIDFT, Complex *pTwiddleMatrix);
void calcGoldRef( unsigned char *pIn, double complex *pTwiddleDFT, double complex *pTwiddleIDFT,
				  float *pFilterCoeffs );
void dispatchDFTkernel(unsigned char* g_pInput,
					   Complex *g_pXfmCoeffs,
					   Complex *g_pTempMatrix,
					   Complex *g_p2D_xfm,
					   bool isIDFT,
					   bool isRowDFT,
					   unsigned char *g_pImgOut,
					   dim3 nBlocks,
					   dim3 nThreads);
void dispatchFilterKernel(Complex *g_pIn,
					   	  float *g_pCoeffs,
					   	  Complex *g_pOut,
					   	  dim3 nBlocks,
					   	  dim3 nThreads);

////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex addition
static inline Complex ComplexAdd(Complex a, Complex b)
{
    Complex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

// Complex scale
static inline Complex ComplexScale(Complex a, float s)
{
    Complex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

// Complex multiplication
static inline Complex ComplexMul(Complex a, Complex b)
{
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

// Complex exponentiation - only intended for host use
static inline double complex ComplexExp(double complex z)
{
	return cexp ( z );
}

#endif /* DFT2D_H_ */
