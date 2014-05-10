/*
 * 	dft2D.cpp - Main for Project
 *
 *  Created on: May 1, 2014
 *  Author: 	Evan Hosseini
 *  Class:  	CS708
 */

// Helper functions and declarations
#include "dft2D.h"

unsigned char *h_pImage = NULL;						// Host input image handle
static int imWidth = 0, imHeight = 0;				// Image Dimensions


/* Modified utility function taken from CUDA samples */
void initializeData(char *file)
{
    unsigned int w, h;
    size_t file_length= strlen(file);

    if (!strcmp(&file[file_length-3], "pgm"))
    {
        if (sdkLoadPGM<unsigned char>(file, &h_pImage, &w, &h) != true)
        {
            printf("Failed to load PGM image file: %s\n", file);
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        // cudaDeviceReset causes the driver to clean up all state. While
        // not mandatory in normal operation, it is good practice.  It is also
        // needed to ensure correct operation when the application is being
        // profiled. Calling cudaDeviceReset causes all profile data to be
        // flushed before the application exits
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }

    imWidth = (int)w;
    imHeight = (int)h;

    assert( imWidth == N );
    assert( imHeight == N );

}

/* Modified utility function taken from CUDA samples */
void loadDefaultImage(char *loc_exec)
{

    printf("Reading image... \n");
    const char *image_filename = "lena.pgm";
    char *image_path = sdkFindFilePath(image_filename, loc_exec);

    if (image_path == NULL)
    {
        printf("Failed to read image file: <%s>\n", image_filename);
        exit(EXIT_FAILURE);
    }

    initializeData(image_path);
    free(image_path);
}

// Function to generate the DFT/IDFT N x N Twiddle matrix to be passed to the GPU
//! @param isIDFT 				IDFT/DFT option
//! @param pTwiddleMatrix		Generated twiddle matrix handle
//! @param pTwiddleMatrix_z		Generated twiddle matrix complex type handle
void twiddleMatrixGen(bool isIDFT, Complex *pTwiddleMatrix, double complex *pTwiddleMatrix_z)
{
	// Exponent return value
	double complex m_result;

	// Capture Twiddle Matrix reference for resetting pointer before return
	Complex *pTwiddleRef = pTwiddleMatrix;
	double complex *pTwiddleRef_z = pTwiddleMatrix_z;

	// Condition exponent sign bit and scaling on DFT or IDFT
	int sign = -1;
	float scale = 1;
	if ( !isIDFT )
	{
		sign = 1;
		scale = 1 / static_cast<float> ( N ) ;
	}

	for ( int i = 0; i < N; ++i )
		for ( int j = 0; j < N; ++j, ++pTwiddleMatrix, ++pTwiddleMatrix_z )
		{
			m_result = cexp( sign * 2 * M_PI * i * j / N * I ) * scale;
			*pTwiddleMatrix_z = m_result;
			pTwiddleMatrix->x = creal( m_result );
			pTwiddleMatrix->y = cimag( m_result );
		}

	// Reset the Matrix pointers
	pTwiddleMatrix = pTwiddleRef;
	pTwiddleMatrix_z = pTwiddleRef_z;
}

// Function to generate lowpass/highpass filter coefficients
//! @param isLowpass 	Lowpass or highpass option
//! @param wc			Filter cutoff frequency - expressed in fraction of N
//! @param pCoeffs		Generated 2D filter coefficients handle
void filterCoeffGen(bool isLowpass, float wc, float *pCoeffs)
{
	// Pass band count based on cutoff frequency wc
	float fPassband = wc * N;

	// Capture reference for local iteration
	float *m_pCoeffs = pCoeffs;

	// Loop over matrix and set the filter coefficients based on filter type
	// Configure both filters to have unity gain and exponential rolloff @ wc
	if (isLowpass)
	{
		for ( int i = 0; i < N; ++i )
			for (int j = 0; j < N; ++j, ++m_pCoeffs )
			{
				if ( i < fPassband && j < fPassband)
					*m_pCoeffs = 1;
				else
					*m_pCoeffs = 0;
			}
	}
	else
	{
		for ( int i = 0; i < N; ++i )
			for (int j = 0; j < N; ++j, ++m_pCoeffs )
			{
				if ( i > fPassband && j > fPassband )
					*m_pCoeffs = 1;
				else
					*m_pCoeffs = 0;
			}
	}
}

// Function to generate CPU reference filter
//! @param pIn		 			Input Image
//! @param pTwiddleDFT			DFT twiddle matrix
//! @param pTwiddleIDFT			IDFT twiddle matrix
//! @param pFilterCoeffs		Filter coefficients
void calcGoldRef( unsigned char *pIn, double complex *pTwiddleDFT, double complex *pTwiddleIDFT,
				  float *pFilterCoeffs )
{
	// Output image
	char *fnameOut = "data/lena_filtered_cpu.pgm";

	// Benchmark time stamps
	clock_t timer;
	timer = clock();

	// Capture local iterators
	unsigned char *m_pIn = pIn;
	double complex *m_pTwiddleDFT = pTwiddleDFT;
	double complex *m_pTwiddleIDFT = pTwiddleIDFT;
	float *m_pFilterCoeffs = pFilterCoeffs;

	// Calculate row-wise 1D DFT
	double complex *m_pTemp, *m_pTempRef;
	m_pTemp = new double complex[N*N];
	m_pTempRef = m_pTemp;
	memset( m_pTemp, 0, sizeof( double complex ) * N * N );
	for ( int i = 0; i < N; ++i )
	{
		m_pIn = pIn + i * N;
		for (int j = 0; j < N; ++j, ++m_pTemp )
		{
			m_pTwiddleDFT = pTwiddleDFT + j;
			m_pIn = pIn + i * N;
			for ( int k = 0; k < N; ++k, ++m_pIn )
			{
				*m_pTemp += (double)*m_pIn * *m_pTwiddleDFT;
				m_pTwiddleDFT += N;
			}
		}
	}
	std::cout << "Finished 1D DFT .." << std::endl;

	// Reset pointers for next stage
	m_pTemp = m_pTempRef;
	m_pTwiddleDFT = pTwiddleDFT;

	// Next, calculate column-wise 1D DFT
	double complex *m_p2D_xfm, *m_p2D_xfmRef;
	m_p2D_xfm = new double complex[N*N];
	m_p2D_xfmRef = m_p2D_xfm;
	memset( m_p2D_xfm, 0, sizeof( double complex ) * N * N );
	for ( int i = 0; i < N; ++i )
	{
		m_pTwiddleDFT = pTwiddleDFT + i * N;
		for (int j = 0; j < N; ++j, ++m_p2D_xfm )
		{
			m_pTwiddleDFT = pTwiddleDFT + i * N;
			m_pTemp = m_pTempRef + j;
			for ( int k = 0; k < N; ++k, ++m_pTwiddleDFT )
			{
				*m_p2D_xfm += *m_pTemp * *m_pTwiddleDFT;
				m_pTemp += N;
			}
		}
	}
	std::cout << "Finished 2D DFT .." << std::endl;

	// Reset pointers for next stages
	m_p2D_xfm = m_p2D_xfmRef;
	m_pTemp = m_pTempRef;

	// Next, apply filter to the transformed data
	double complex *m_pFilterOut, *m_pFilterOutRef;
	m_pFilterOut = new double complex[N*N];
	m_pFilterOutRef = m_pFilterOut;
	memset( m_pFilterOut, 0, sizeof( double complex ) * N * N );
	// Perform element multiplication for filter application
	for ( int i = 0; i < N * N; ++i, ++m_pFilterCoeffs, ++m_p2D_xfm, ++m_pFilterOut )
		*m_pFilterOut = *m_pFilterCoeffs * *m_p2D_xfm;

	std::cout << "Finished Filtering .." << std::endl;
	// Reset pointers for next stages
	m_pFilterOut = m_pFilterOutRef;
	m_p2D_xfm = m_p2D_xfmRef;

	// Next, apply row-wise 1D IDFT - dump in already allocated Temp matrix
	memset( m_pTemp, 0, sizeof( double complex ) * N * N );
	for ( int i = 0; i < N; ++i )
	{
		m_pFilterOut = m_pFilterOutRef + i * N;
		for ( int j = 0; j < N; ++j, ++m_pTemp )
		{
			m_pTwiddleIDFT = pTwiddleIDFT + j;
			m_pFilterOut = m_pFilterOutRef + i * N;
			for ( int k = 0; k < N; ++k, ++m_pFilterOut )
			{
				*m_pTemp += *m_pFilterOut * *m_pTwiddleIDFT;
				m_pTwiddleIDFT += N;
			}
		}
	}
	std::cout << "Finished 1D IDFT .." << std::endl;
	// Reset pointer for next stage
	m_pTemp = m_pTempRef;
	m_pTwiddleIDFT = pTwiddleIDFT;

	// Next, apply column-wise 1D IDFT - dump in already allocated 2D_xfm matrix
	memset( m_p2D_xfm, 0, sizeof( double complex ) * N * N );
	for ( int i = 0; i < N; ++i )
	{
		m_pTwiddleIDFT = pTwiddleIDFT + i * N;
		for ( int j = 0; j < N; ++j, ++m_p2D_xfm )
		{
			m_pTemp = m_pTempRef + j;
			m_pTwiddleIDFT = pTwiddleIDFT + i * N;
			for ( int k = 0; k < N; ++k, ++m_pTwiddleIDFT )
			{
				*m_p2D_xfm += *m_pTemp * *m_pTwiddleIDFT;
				m_pTemp += N;
			}
		}
	}
	std::cout << "Finished 2D IDFT .." << std::endl;
	// Reset output pointer
	m_p2D_xfm = m_p2D_xfmRef;

	// Compute magnitude and convert output result to unsigned char for pgm write
	unsigned char *m_pOut = new unsigned char[N*N];
	unsigned char *m_pOutRef = m_pOut;
	for ( int i = 0; i < N * N; ++i, ++m_p2D_xfm, ++m_pOut )
		*m_pOut = (unsigned char) sqrt( pow( creal(*m_p2D_xfm), 2 ) + pow( cimag(*m_p2D_xfm), 2 ) );

	// Generate end time stamp and report performance
	timer = clock() - timer;
	std::cout << "Total CPU execution time = " << static_cast<float>(timer) / CLOCKS_PER_SEC << " seconds" << std::endl;

	// Write output to file
	if( !sdkSavePGM( fnameOut, m_pOutRef, N, N ) )
		std::cout << "Error Saving CPU output file!!" << std::endl;
	else
		std::cout << "Finished writing to CPU output!" << std::endl;

	// Cleanup local memory allocations
	delete[] m_pTempRef;
	delete[] m_p2D_xfmRef;
	delete[] m_pFilterOutRef;

}

int main(int argc, char *argv[])
{
	char *fnameOut = "data/lena_filtered_gpu.pgm";	// Output image
	unsigned char *h_pImgResult = NULL;				// Output handle
	Complex *h_pTwiddleDFT = NULL;					// Host DFT Twiddle Matrix
	Complex *h_pTwiddleIDFT = NULL;					// Host IDFT Twiddle Matrix
	double complex *h_pTwiddleDFT_z = NULL;			// Host CPU DFT Twiddle Matrix
	double complex *h_pTwiddleIDFT_z = NULL;		// Host CPU IDFT Twiddle Matrix
	float *h_pFilterCoeffs = NULL;					// Host Filter Coefficent Matrix
	static const int dev = 0;						// Hard code to use device 0
	cudaEvent_t start, stop;						// Cuda events for benchmarking
	float time;										// Performance timer for benchmarking
	std::string filterOption ("highpass");			// Cmd line filter option


	// Load the image
	loadDefaultImage( argv[0] );

	// Capture image reference
	unsigned char *m_imgRef = h_pImage;

	std::cout << "Image width = " << imWidth << " and image height = " << imHeight << std::endl;

	// Define number of threads and blocks to be mapped to the GPU
	dim3 nThreads = dim3( blkSize_x, blkSize_y, 1 );
	dim3 nBlocks = dim3( ceil( imWidth / nThreads.x ), ceil( imHeight / nThreads.y ) );
	int nPixels = imWidth * imHeight;
	unsigned int imSz = sizeof( unsigned char ) * nPixels;

	// Allocate host memory for the result
	h_pImgResult = static_cast<unsigned char*>( malloc( imSz ) );
	unsigned char *m_imgOut = h_pImgResult;

	// Generate the DFT and IDFT twiddle matrix to be dumped to the GPU
	// and for CPU reference implementation
	unsigned int twiddleSz = sizeof( Complex ) * nPixels;
	unsigned int twiddleSz_z = sizeof( double complex ) * nPixels;
	h_pTwiddleDFT = static_cast<Complex*>( malloc( twiddleSz ) );
	h_pTwiddleDFT_z = static_cast<double complex*>( malloc( twiddleSz_z ) );
	twiddleMatrixGen( false, h_pTwiddleDFT, h_pTwiddleDFT_z );

	h_pTwiddleIDFT = static_cast<Complex*>( malloc ( twiddleSz ) );
	h_pTwiddleIDFT_z = static_cast<double complex*>( malloc( twiddleSz_z ) );
	twiddleMatrixGen( true, h_pTwiddleIDFT, h_pTwiddleIDFT_z );

	// Allocate and generate the Filter Coefficient Matrix
	unsigned int filterSz = sizeof( float ) * nPixels;
	h_pFilterCoeffs = static_cast<float*>( malloc( filterSz ) );

	if ( filterOption == "lowpass" )
		filterCoeffGen( true, W_C, h_pFilterCoeffs );
	else
		filterCoeffGen( false, W_C, h_pFilterCoeffs );

	// Initialize the result buffer
	for ( int i = 0; i < nPixels; ++i )
		h_pImgResult[i] = 0;

	// Allocate device memory
	unsigned char *d_pImage, *d_pImgResult;
	Complex *d_pTwiddleDFT, *d_pTwiddleIDFT, *d_pTempMatrix, *d_p2D_xfm, *d_pFilterOut;
	float *d_pFilterCoeffs;

	checkCudaErrors( cudaMalloc( (void**) &d_pImage, imSz ) );
	checkCudaErrors( cudaMalloc( (void**) &d_pImgResult, imSz ) );
	checkCudaErrors( cudaMalloc( (void**) &d_pTwiddleDFT, twiddleSz ) );
	checkCudaErrors( cudaMalloc( (void**) &d_pTwiddleIDFT, twiddleSz ) );
	checkCudaErrors( cudaMalloc( (void**) &d_pTempMatrix, twiddleSz ) );
	checkCudaErrors( cudaMalloc( (void**) &d_p2D_xfm, twiddleSz ) );
	checkCudaErrors( cudaMalloc( (void**) &d_pFilterCoeffs, filterSz ) );
	checkCudaErrors( cudaMalloc( (void**) &d_pFilterOut, filterSz ) );

	// Copy host memory to the device - use h_pImgResult for zeroing device result buffer
	checkCudaErrors( cudaMemcpy( d_pImage, h_pImage, imSz, cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy( d_pImgResult, h_pImgResult, imSz, cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy( d_pTwiddleDFT, h_pTwiddleDFT, twiddleSz, cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy( d_pTwiddleIDFT, h_pTwiddleIDFT, twiddleSz, cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy( d_pFilterCoeffs, h_pFilterCoeffs, filterSz, cudaMemcpyHostToDevice ) );

	// Create CUDA events for the timer
	checkCudaErrors ( cudaEventCreate( &start ) );
	checkCudaErrors ( cudaEventCreate( &stop ) );

	// Kickoff start timer and dispatch the row-wise DFT kernel, block until the kernel returns
	checkCudaErrors( cudaEventRecord( start, NULL ) );
	dispatchDFTkernel( d_pImage, d_pTwiddleDFT, d_pTempMatrix, d_p2D_xfm, false, true, d_pImgResult, nBlocks, nThreads );
	cudaDeviceSynchronize();

	// Dispatch the column-wise DFT kernel, block until the kernel returns
	dispatchDFTkernel( d_pImage, d_pTwiddleDFT, d_pTempMatrix, d_p2D_xfm, false, false, d_pImgResult, nBlocks, nThreads );
	cudaDeviceSynchronize();

	// Dispatch the filter kernel, block until the kernel returns
	dispatchFilterKernel( d_p2D_xfm, d_pFilterCoeffs, d_pTempMatrix, nBlocks, nThreads );
	cudaDeviceSynchronize();

	// Dispatch the row-wise IDFT kernel, block until the kernel returns
	dispatchDFTkernel( d_pImage, d_pTwiddleIDFT, d_pTempMatrix, d_p2D_xfm, true, true, d_pImgResult, nBlocks, nThreads );
	cudaDeviceSynchronize();

	// Dispatch the column-wise IDFT kernel, block until the kernel returns
	dispatchDFTkernel( d_pImage, d_pTwiddleIDFT, d_pTempMatrix, d_p2D_xfm, true, false, d_pImgResult, nBlocks, nThreads );
	cudaDeviceSynchronize();

	// Test
	checkCudaErrors( cudaMemcpy( h_pImgResult, d_pImgResult, imSz, cudaMemcpyDeviceToHost ) );

	// Generate stop event and record execution time
	checkCudaErrors( cudaEventRecord( stop, NULL ) );
	checkCudaErrors( cudaEventSynchronize( stop ) );
	checkCudaErrors( cudaEventElapsedTime( &time, start, stop ) );
	checkCudaErrors( cudaEventDestroy( start ) );
	checkCudaErrors( cudaEventDestroy( stop ) );

	// Print out time results
	std::cout << "Total GPU Execution time = " << time/1000 << " seconds" << std::endl;

	// Save our result
	if( !sdkSavePGM( fnameOut, m_imgOut, imWidth, imHeight ) )
		std::cout << "Error Saving output file!!" << std::endl;
	else
		std::cout << "Finished writing to output!" << std::endl;

    cudaDeviceReset();

    // Calculate CPU reference output and benchmark
    calcGoldRef( h_pImage, h_pTwiddleDFT_z, h_pTwiddleIDFT_z, h_pFilterCoeffs );

    // Host cleanup - CUDA device reset reclaims device memory allocations
    if ( h_pImgResult != NULL ) free ( h_pImgResult );
    if ( h_pTwiddleDFT != NULL ) free ( h_pTwiddleDFT );
    if ( h_pTwiddleDFT_z != NULL ) free ( h_pTwiddleDFT_z );
    if ( h_pTwiddleIDFT != NULL ) free ( h_pTwiddleIDFT );
    if ( h_pTwiddleIDFT_z != NULL ) free ( h_pTwiddleIDFT_z );
    if ( h_pFilterCoeffs != NULL ) free ( h_pFilterCoeffs );

    exit(EXIT_SUCCESS);
}

