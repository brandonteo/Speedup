// CS 61C Fall 2015 Project 4

// include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <x86intrin.h>
#endif

// include OpenMP
#if !defined(_MSC_VER)
#include <pthread.h>
#endif
#include <omp.h>

#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#include "calcDepthOptimized.h"
#include "calcDepthNaive.h"

/* DO NOT CHANGE ANYTHING ABOVE THIS LINE. */

#include <string.h>

#define MAX(a, b) ((a > b)? a : b)
#define MIN(a, b) ((a < b)? a : b)

void calcDepthOptimized(float *depth, float *left, float *right, int imageWidth, int imageHeight, int featureWidth, int featureHeight, int maximumDisplacement) 
{
	memset(depth, 0, imageHeight*imageWidth*sizeof(float));
	int first_threshold = (2 * featureWidth + 1) / 8 * 8;
	int second_threshold = (2 * featureWidth + 1) / 4 * 4;
	int third_threshold = (2 * featureWidth + 1);

 	/* The two outer for loops iterate through each pixel */
	#pragma omp parallel for
	for (int y = featureHeight; y < imageHeight - featureHeight; y++)
	{
		for (int x = featureWidth; x < imageWidth - featureWidth; x++)
		{	

			float minimumSquaredDifference = -1;
			int minimumDy = 0;
			int minimumDx = 0;

			/* Iterate through all feature boxes that fit inside the maximum displacement box. 
			   centered around the current pixel. */
			int dyLBound = MAX(-maximumDisplacement, featureHeight - y);
			int dxLBound = MAX(-maximumDisplacement, featureWidth - x);
			int dyUBound = MIN(maximumDisplacement, imageHeight - y - 1 - featureHeight);
			int dxUBound = MIN(maximumDisplacement, imageWidth - x - 1 - featureWidth);

			for (int dy = dyLBound; dy <= dyUBound; dy++)
			{
				for (int dx = dxLBound; dx <= dxUBound; dx++)
				{

					float squaredDifference = 0;
					__m128 sumBlock = _mm_setzero_ps();
					float tempDiff[4];

					/* Sum the squared difference within a box of +/- featureHeight and +/- featureWidth. */
					for (int boxX = 0; boxX < first_threshold; boxX += 8)
					//for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)
					{
						//for (int boxX = 0; boxX < first_threshold; boxX += 8)
						for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)
						{
							int leftX = x + boxX - featureWidth;
							int leftY = y + boxY;
							int rightX = x + dx + boxX - featureWidth;
							int rightY = y + dy + boxY;
							int leftIndex = leftY * imageWidth + leftX;
							int rightIndex = rightY * imageWidth + rightX;

							__m128 leftBlock = _mm_loadu_ps(left + leftIndex);
							__m128 rightBlock = _mm_loadu_ps(right + rightIndex);
							__m128 differenceBlock = _mm_sub_ps(leftBlock, rightBlock);
							__m128 squareBlock = _mm_mul_ps(differenceBlock, differenceBlock);
							sumBlock = _mm_add_ps(squareBlock, sumBlock);

                                                        leftBlock = _mm_loadu_ps(left + leftIndex + 4);
							rightBlock = _mm_loadu_ps(right + rightIndex + 4);
							differenceBlock = _mm_sub_ps(leftBlock, rightBlock);
							squareBlock = _mm_mul_ps(differenceBlock, differenceBlock);
							sumBlock = _mm_add_ps(squareBlock, sumBlock);
						}
					}

					for (int boxX = first_threshold; boxX < second_threshold; boxX += 4)
					//for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)
					{
						//for (int boxX = first_threshold; boxX < second_threshold; boxX += 4)
						for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)
						{
							int leftX = x + boxX - featureWidth;
							int leftY = y + boxY;
							int rightX = x + dx + boxX - featureWidth;
							int rightY = y + dy + boxY;
							int leftIndex = leftY * imageWidth + leftX;
							int rightIndex = rightY * imageWidth + rightX;

							__m128 leftBlock = _mm_loadu_ps(left + leftIndex);
							__m128 rightBlock = _mm_loadu_ps(right + rightIndex);
							__m128 differenceBlock = _mm_sub_ps(leftBlock, rightBlock);
							__m128 squareBlock = _mm_mul_ps(differenceBlock, differenceBlock);
							sumBlock = _mm_add_ps(squareBlock, sumBlock);
						}
					}

					for (int boxX = second_threshold; boxX < third_threshold; boxX += 1)
					//for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)
					{
						//for (int boxX = second_threshold; boxX < third_threshold; boxX += 1)
						for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)
						{
							int leftX = x + boxX - featureWidth;
							int leftY = y + boxY;
							int rightX = x + dx + boxX - featureWidth;
							int rightY = y + dy + boxY;
							int leftIndex = leftY * imageWidth + leftX;
							int rightIndex = rightY * imageWidth + rightX;

							float diff = left[leftIndex] - right[rightIndex];
							squaredDifference += diff * diff;
						}
					}

					_mm_storeu_ps(tempDiff, sumBlock);
					squaredDifference += tempDiff[0] + tempDiff[1] + tempDiff[2] + tempDiff[3];

					/* 
					Check if you need to update minimum square difference. 
					This is when either it has not been set yet, the current
					squared displacement is equal to the min and but the new
					displacement is less, or the current squared difference
					is less than the min square difference.
					*/
					if ((minimumSquaredDifference == -1) || ((minimumSquaredDifference == squaredDifference) && (displacementNaive(dx, dy) < displacementNaive(minimumDx, minimumDy))) || (minimumSquaredDifference > squaredDifference))
					{
						minimumSquaredDifference = squaredDifference;
						minimumDx = dx;
						minimumDy = dy;
					}
				}
			}

			/* 
			Set the value in the depth map. 
			If max displacement is equal to 0, the depth value is just 0.
			*/
			if (minimumSquaredDifference == -1 || (minimumSquaredDifference != -1 &&maximumDisplacement == 0))
			{
				depth[y * imageWidth + x] = 0;
			}
			else
			{
				depth[y * imageWidth + x] = displacementNaive(minimumDx, minimumDy);
			}
			
		}
	}
}
