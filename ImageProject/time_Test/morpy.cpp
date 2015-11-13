/************************************************************************* 
    > File Name: morpy.cpp
    > Author: chefang
    > Mail: chefang.com
    > Created Time: 2015年10月28日 星期三 11时23分29秒
 ************************************************************************/

#include<iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include <stdio.h>

using namespace std;
using namespace cv;

void help()
{
	printf("\n------------------------------");
	printf("\n input a number according the list and choose what you want:\n");
	printf("1.dilate\n");
	printf("2.erode\n");
	printf("3.open\n");
	printf("4.close\n");
	printf("5.morphological gradient\n");
	printf("6.tophat\n");
	printf("7.blackhat\n");
}

int main(int argc, char * argv[])
{
	Mat src_cpu = imread("1.jpg");
	Mat dst_cpu;
	cuda::GpuMat src_gpu, dst_gpu;
	//src_gpu.upload(src_cpu);
	
	double start, time;
	char ch;

	help();

	if(src_cpu.channels() == 3)
	{
		Mat src4ch;
		cvtColor(src_cpu, src4ch, COLOR_BGR2BGRA);
		src_cpu = src4ch;
	}
	src_gpu.upload(src_cpu);

#if 0
	if(src_gpu.channels() == 3)
	{
		cuda::GpuMat src4ch;
		cuda::cvtColor(src_gpu, src4ch, COLOR_BGR2BGRA);
		src_gpu = src4ch;
	}
#endif	
	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5), Point(2, 2));
	printf("\nplease input a number:");
	
	while( (ch = getchar()) != '0')
	{
		switch(ch)
		{
			case '1'://dilate
				{
					start = static_cast<double>(getTickCount());
					//morphologyEx(src_cpu, dst_cpu, MORPH_DILATE, element );
					dilate(src_cpu, dst_cpu, element);
					time = (getTickCount() - start) / getTickFrequency();
					cout << "start" << start<< endl;
					cout << "cpu about dilate runtime:" << time*1000 << endl;
					imshow("cpu_dst  dilate", dst_cpu);
					
					imshow("src_image", src_cpu);

					Ptr<cuda::Filter> filter = cuda::createMorphologyFilter(MORPH_DILATE, src_gpu.type(), element);
					start = static_cast<double>(getTickCount());
					filter->apply(src_gpu, dst_gpu);
					time = ((double)getTickCount() - start) / getTickFrequency();
					cout << "start" << start<< endl;
					cout << "gpu about dilate runtime:" << time*1000 << endl;
					Mat h_dst(dst_gpu);
					imshow("gpu_dst dilate", h_dst);
				}break;
			case '2'://erode
				{
				start = static_cast<double>(getTickCount());
					//morphologyEx(src_cpu, dst_cpu, MORPH_ERODE, element );
					erode(src_cpu, dst_cpu, element);
					time = ((double)getTickCount() - start) / getTickFrequency();
					cout << "cpu about erode runtime:" << time*1000 << endl;
					
					
					Ptr<cuda::Filter> filter = cuda::createMorphologyFilter(MORPH_ERODE, src_gpu.type(), element);
					start = static_cast<double>(getTickCount());
					filter->apply(src_gpu, dst_gpu);
					time = ((double)getTickCount() - start) / getTickFrequency();
					cout << "gpu about erode runtime:" << time*1000 << endl;
				}break;
			case '3'://open
				{
					start = static_cast<double>(getTickCount());
					morphologyEx(src_cpu, dst_cpu, MORPH_OPEN, element );
					time = ((double)getTickCount() - start) / getTickFrequency();
					cout << "cpu about open runtime:" << time*1000 << endl;
					
					
					Ptr<cuda::Filter> filter = cuda::createMorphologyFilter(MORPH_OPEN, src_gpu.type(), element);
					start = static_cast<double>(getTickCount());					
					filter->apply(src_gpu, dst_gpu);
					time = ((double)getTickCount() - start) / getTickFrequency();
					cout << "gpu about open runtime:" << time*1000 << endl;
				}break;
			case '4'://close
				{
					start = static_cast<double>(getTickCount());
					morphologyEx(src_cpu, dst_cpu, MORPH_CLOSE, element );
					time = ((double)getTickCount() - start) / getTickFrequency();
					cout << "cpu about close runtime:" << time*1000 << endl;
					
					
					Ptr<cuda::Filter> filter = cuda::createMorphologyFilter(MORPH_CLOSE, src_gpu.type(), element);
					start = static_cast<double>(getTickCount());					
					filter->apply(src_gpu, dst_gpu);
					time = ((double)getTickCount() - start) / getTickFrequency();
					cout << "gpu about close runtime:" << time*1000 << endl;
				}break;
			case '5'://Morphological Gradient
				{
					start = static_cast<double>(getTickCount());
					morphologyEx(src_cpu, dst_cpu, MORPH_GRADIENT, element );
					time = ((double)getTickCount() - start) / getTickFrequency();
					cout << "cpu about Morphological Gradient runtime:" << time*1000 << endl;
					
					
					Ptr<cuda::Filter> filter = cuda::createMorphologyFilter(MORPH_GRADIENT, src_gpu.type(), element);
					start = static_cast<double>(getTickCount());
					filter->apply(src_gpu, dst_gpu);
					time = ((double)getTickCount() - start) / getTickFrequency();
					cout << "gpu about Morphological gradient runtime:" << time*1000 << endl;
				}break;
			case '6'://tophat
				{
					start = static_cast<double>(getTickCount());
					morphologyEx(src_cpu, dst_cpu, MORPH_TOPHAT, element );
					time = ((double)getTickCount() - start) / getTickFrequency();
					cout << "cpu about tophat runtime:" << time*1000 << endl;
					
					Ptr<cuda::Filter> filter = cuda::createMorphologyFilter(MORPH_TOPHAT, src_gpu.type(), element);
					start = static_cast<double>(getTickCount());					
					filter->apply(src_gpu, dst_gpu);
					time = ((double)getTickCount() - start) / getTickFrequency();
					cout << "gpu about tophat runtime:" << time*1000 << endl;
				}break;
			case '7'://blackhat
				{
					start = static_cast<double>(getTickCount());
					morphologyEx(src_cpu, dst_cpu, MORPH_BLACKHAT, element );
					time = ((double)getTickCount() - start) / getTickFrequency();
					cout << "cpu about blackhat runtime:" << time*1000 << endl;
					
					
					Ptr<cuda::Filter> filter = cuda::createMorphologyFilter(MORPH_BLACKHAT, src_gpu.type(), element);
					start = static_cast<double>(getTickCount());
					filter->apply(src_gpu, dst_gpu);
					time = ((double)getTickCount() - start) / getTickFrequency();
					cout << "gpu about blackhat runtime:" << time*1000 << endl;
				}break;
			default:
				break;
		}
		printf("Please input a number:");
		getchar();
	}

	//Mat h_dst(dst_gpu);
	//imshow("open/close", h_dst);
	waitKey(0);
	return 0;
}
