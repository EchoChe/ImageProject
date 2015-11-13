/************************************************************************* 
    > File Name: convolve.cpp
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
#include <stdio.h>

using namespace std;
using namespace cv;
using namespace cv::cuda;

void help()
{
	printf("\nimput a number according the list :");
	printf("\n1.boxFilter(MeanValue)");
	printf("\n2.blur(MeanValue)");
	printf("\n3.Gaussian(Filter)");
	printf("\n4.medianBlur");
	printf("\n5.bilateralFilter");
	printf("\n0.exit\n");
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
			case '1':
			{

				start = getTickCount();
				boxFilter(src, dst, -1, Size(5, 5));
				time = (getTickCount() - start) / getTickFrequency();
				cout << "cpu time " << time * 1000 <<"ms" << endl;
				
				Ptr<cuda::Filter> filter = cuda::createBoxFilter(src_gpu.type(), src_gpu.type(), Size(5, 5));
				start = getTickCount();
				filter->apply(src_gpu, dst_gpu);
				time = (getTickCount() - start) / getTickFrequency();
				cout << "gpu time " << time * 1000 << "ms"<< endl;
			}break;
			
			case '2':
			{
				start = static_cast<double>(getTickCount());
				blur(src, dst, Size(5, 5) );
				time = ((double)getTickCount() - start) / getTickFrequency();
				cout << "cpu runtime" << time*1000 << "ms" << endl;
			}break;
			case '3':
			{
				start = static_cast<double>(getTickCount());
				GaussianBlur(src, dst, Size(3,3), 0, 0);
				time = ((double)getTickCount() - start) / getTickFrequency();
				cout << "cpu runtime" << time*1000 << "ms" << endl;

				Ptr<cuda::Filter> filter = cuda::createGaussianFilter(src_gpu.type(), src_gpu.type(), Size(3,3), 0, 0);
				start = static_cast<double>(getTickCount());
				filter->apply(src_gpu, dst_gpu);
				time = ((double)getTickCount() - start)/getTickFrequency();
				cout << "gpu time" << time * 1000 << "ms" << endl;
			}break;
			case '4':
			{
				start = static_cast<double>(getTickCount());
				medianBlur(src, dst, 7);
				time = ((double)getTickCount() - start) / getTickFrequency();
				cout << "cpu runtime" << time*1000 << "ms" << endl;
			}break;
			case '5':
			{
				start = static_cast<double>(getTickCount());
				cv::bilateralFilter(src, dst, 25, 25*2, 25/2);
				time = ((double)getTickCount() - start) / getTickFrequency();
				cout << "cpu runtime" << time*1000 << "ms" << endl;
				
				start = static_cast<double>(getTickCount());
				cuda::bilateralFilter(src_gpu, dst_gpu, 25, 25*2, 25/2);
				time = ((double)getTickCount() - start) / getTickFrequency();
				cout << "gpu runtime" << time*1000 << "ms" << endl;
			}break;
		default:
			break;

		}
		printf("please input a number:");
		getchar();
	}

	return 0;
}

