/************************************************************************* 
    > File Name: sharp.cpp
    > Author: chefang
    > Mail: chefang.com
    > Created Time: 2015年10月28日 星期三 09时20分32秒
 ************************************************************************/

#include<iostream>
#include <cmath>
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
//#include "opencv2/precomp.hpp"
#include <stdio.h>

using namespace std;
using namespace cv;
using namespace cv::cuda;

void help()
{
	printf("\nimput a number according the list :");
	printf("\n1.Filter");
	printf("\n2.Sobel");
	printf("\n3.Scharr");
	printf("\n4.Laplacian");
	printf("thr number:");
}

int main(int argc, char * argv[])
{
	Mat src = imread("test.jpg");
	//Mat src = imread("test.jpg", IMREAD_GRAYSCALE);
	Mat dst;
	double start, time;
	char ch;
	cuda::GpuMat src_gpu, dst_gpu;
	
	src_gpu.upload(src);
		
	cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
	help();

	if(src_gpu.channels() == 3)
	{
		cuda::GpuMat src4ch;
		cuda::cvtColor(src_gpu, src4ch, COLOR_BGR2BGRA);
		src_gpu = src4ch;
	}

	while((ch = getchar()) != '0')
	{
		switch(ch){
			case '1'://Filter2D LinearFilter
			{
				Mat KernRobert = (Mat_<char>(2,2) << -1, 0,
													0, 1);
			
				start = getTickCount();
				filter2D(src, dst, src.depth(), KernRobert);
				time = (getTickCount() - start) / getTickFrequency();
				cout << "cpu time about Filter2D: " << time * 1000 <<"ms" << endl;
				
				Ptr<cuda::Filter> filter = cuda::createLinearFilter(src_gpu.type(), src_gpu.type(), KernRobert);
				start = getTickCount();
				filter->apply(src_gpu, dst_gpu);
				time = (getTickCount() - start) / getTickFrequency();
				cout << "gpu time about LinearFilter : " << time * 1000 << "ms"<< endl;
			}break;
			
			case '2'://sobel
			{
				start = static_cast<double>(getTickCount());
				Sobel(src, dst, CV_16S, 1, 0);
				time = ((double)getTickCount() - start) / getTickFrequency();
				cout << "cpu runtime about sobel: " << time*1000 << "ms" << endl;

				Ptr<cuda::Filter> filter = cuda::createSobelFilter(src_gpu.type(), dst_gpu.type(), 1, 0);
				start = static_cast<double>(getTickCount());
				filter->apply(src_gpu, dst_gpu);
				time = (getTickCount() - start) / getTickFrequency();
				cout << "gpu run time about sobel: " << time * 1000 << "ms" << endl;

			}break;
			case '3'://scharr
			{
				start = static_cast<double>(getTickCount());
				Scharr(src, dst, src.depth(), 1, 0);
				time = ((double)getTickCount() - start) / getTickFrequency();
				cout << "cpu runtime about scharr: " << time*1000 << "ms" << endl;

				Ptr<cuda::Filter> filter = cuda::createScharrFilter(src_gpu.type(), src_gpu.type(),1, 0 );
				start = static_cast<double>(getTickCount());
				filter->apply(src_gpu, dst_gpu);
				time = ((double)getTickCount() - start)/getTickFrequency();
				cout << "gpu time about scharr: " << time * 1000 << "ms" << endl;
			}break;
			case '4'://Laplacian
			{
				start = static_cast<double>(getTickCount());
				Laplacian(src, dst, src.depth(), 3);
				time = ((double)getTickCount() - start) / getTickFrequency();
				cout << "cpu runtime about laplacian: " << time*1000 << "ms" << endl;
				
				Ptr<cuda::Filter> filter = cuda::createLaplacianFilter(src_gpu.type(), src_gpu.type(), 3);
				start = static_cast<double>(getTickCount());
				filter->apply(src_gpu, dst_gpu);
				time = ((double)getTickCount() - start) / getTickFrequency();
				cout << "gpu runtime about laplacian: " << time*1000 << "ms" << endl;
			}break;
			default:
				break;

		}
		printf("please input a number:");
		getchar();
	}

	return 0;
}

