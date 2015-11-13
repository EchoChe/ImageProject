/************************************************************************* 
    > File Name: test.cpp
    > Author: chefang
    > Mail: chefang.com
    > Created Time: 2015年10月29日 星期四 16时46分11秒
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


int main(int argc, char * argv[])
{
	Mat src_cpu = imread("1.jpg");
	Mat dst_cpu;
	cuda::GpuMat src_gpu, dst_gpu;
	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5), Point(2, 2));
	double start, time;
	char ch;

	src_gpu.upload(src_cpu);
		
	cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

	if(src_gpu.channels() == 3)
	{
		cuda::GpuMat src4ch;
		cuda::cvtColor(src_gpu, src4ch, COLOR_BGR2RGBA);
		src_gpu = src4ch;
	}

#if  0
	if(src_cpu.channels() == 3)
	{
		Mat src4ch;
		cv::cvtColor(src_cpu, src4ch, COLOR_BGR2BGRA);
		src_cpu = src4ch;
	}
	src_gpu.upload(src_cpu);
#endif
		//boxFilter
			{
				start = getTickCount();
				boxFilter(src_cpu, dst_cpu, -1, Size(5, 5));
				time = (getTickCount() - start) / getTickFrequency();
				cout << "cpu time boxFilter " << time * 1000 <<"ms" << endl;
				
				Ptr<cuda::Filter> filter = cuda::createBoxFilter(src_gpu.type(), src_gpu.type(), Size(5, 5));
				start = getTickCount();
				filter->apply(src_gpu, dst_gpu);
				time = (getTickCount() - start) / getTickFrequency();
				cout << "gpu time boxFilter " << time * 1000 << "ms"<< endl;
			}
			
		//blur	
			{
				start = static_cast<double>(getTickCount());
				blur(src_cpu, dst_cpu, Size(5, 5) );
				time = ((double)getTickCount() - start) / getTickFrequency();
				cout << "cpu runtime blur" << time*1000 << "ms" << endl;
			}
		//Gaussian	
			{
				start = static_cast<double>(getTickCount());
				GaussianBlur(src_cpu, dst_cpu, Size(3,3), 0, 0);
				time = ((double)getTickCount() - start) / getTickFrequency();
				cout << "cpu runtime Gaussian " << time*1000 << "ms" << endl;

				Ptr<cuda::Filter> filter = cuda::createGaussianFilter(src_gpu.type(), src_gpu.type(), Size(3,3), 0, 0);
				start = static_cast<double>(getTickCount());
				filter->apply(src_gpu, dst_gpu);
				time = ((double)getTickCount() - start)/getTickFrequency();
				cout << "gpu time Gaussian " << time * 1000 << "ms" << endl;
			}
			//medianBlur
			{
				start = static_cast<double>(getTickCount());
				medianBlur(src_cpu, dst_cpu, 7);
				time = ((double)getTickCount() - start) / getTickFrequency();
				cout << "cpu runtime medianBlur " << time*1000 << "ms" << endl;
			}
			//bilateraFilter
			{
				start = static_cast<double>(getTickCount());
				cv::bilateralFilter(src_cpu, dst_cpu, 25, 25*2, 25/2);
				time = ((double)getTickCount() - start) / getTickFrequency();
				cout << "cpu runtime bilateraFilter " << time*1000 << "ms" << endl;
				
				start = static_cast<double>(getTickCount());
				cuda::bilateralFilter(src_gpu, dst_gpu, 25, 25*2, 25/2);
				time = ((double)getTickCount() - start) / getTickFrequency();
				cout << "gpu runtime bilateraFilter " << time*1000 << "ms" << endl;
			}
			
			//dilate
				{
					start = static_cast<double>(getTickCount());
					//morphologyEx(src_cpu, dst_cpu, MORPH_DILATE, element );
					dilate(src_cpu, dst_cpu, element);
					time = (getTickCount() - start) / getTickFrequency();
					cout << "cpu about dilate runtime:" << time*1000 << "ms" << endl;
					//imshow("cpu_dst  dilate", dst_cpu);
					
					//imshow("src_image", src_cpu);

					Ptr<cuda::Filter> filter = cuda::createMorphologyFilter(MORPH_DILATE, src_gpu.type(), element);
					start = static_cast<double>(getTickCount());
					filter->apply(src_gpu, dst_gpu);
					time = ((double)getTickCount() - start) / getTickFrequency();
					cout << "gpu about dilate runtime:" << time*1000 << "ms" << endl;
					Mat h_dst(dst_gpu);
					imshow("gpu_dst dilate", h_dst);
				}
			//erode
				{
					start = static_cast<double>(getTickCount());
					//morphologyEx(src_cpu, dst_cpu, MORPH_ERODE, element );
					erode(src_cpu, dst_cpu, element);
					time = ((double)getTickCount() - start) / getTickFrequency();
					cout << "cpu about erode runtime:" << time*1000 << "ms" << endl;
					
					
					Ptr<cuda::Filter> filter = cuda::createMorphologyFilter(MORPH_ERODE, src_gpu.type(), element);
					start = static_cast<double>(getTickCount());
					filter->apply(src_gpu, dst_gpu);
					time = ((double)getTickCount() - start) / getTickFrequency();
					cout << "gpu about erode runtime:" << time*1000 << "ms" << endl;
				}
			//open
				{
					start = static_cast<double>(getTickCount());
					morphologyEx(src_cpu, dst_cpu, MORPH_OPEN, element );
					time = ((double)getTickCount() - start) / getTickFrequency();
					cout << "cpu about open runtime:" << time*1000 << "ms" << endl;
					
					
					Ptr<cuda::Filter> filter = cuda::createMorphologyFilter(MORPH_OPEN, src_gpu.type(), element);
					start = static_cast<double>(getTickCount());					
					filter->apply(src_gpu, dst_gpu);
					time = ((double)getTickCount() - start) / getTickFrequency();
					cout << "gpu about open runtime:" << time*1000 << "ms" << endl;
				}
			//close
				{
					start = static_cast<double>(getTickCount());
					morphologyEx(src_cpu, dst_cpu, MORPH_CLOSE, element );
					time = ((double)getTickCount() - start) / getTickFrequency();
					cout << "cpu about close runtime:" << time*1000 << "ms" << endl;
					
					
					Ptr<cuda::Filter> filter = cuda::createMorphologyFilter(MORPH_CLOSE, src_gpu.type(), element);
					start = static_cast<double>(getTickCount());					
					filter->apply(src_gpu, dst_gpu);
					time = ((double)getTickCount() - start) / getTickFrequency();
					cout << "gpu about close runtime:" << time*1000 << "ms" << endl;
				}
			//Morphological Gradient
				{
					start = static_cast<double>(getTickCount());
					morphologyEx(src_cpu, dst_cpu, MORPH_GRADIENT, element );
					time = ((double)getTickCount() - start) / getTickFrequency();
					cout << "cpu about Morphological Gradient runtime:" << time*1000 << "ms" << endl;
					
					
					Ptr<cuda::Filter> filter = cuda::createMorphologyFilter(MORPH_GRADIENT, src_gpu.type(), element);
					start = static_cast<double>(getTickCount());
					filter->apply(src_gpu, dst_gpu);
					time = ((double)getTickCount() - start) / getTickFrequency();
					cout << "gpu about Morphological gradient runtime:" << time*1000 << "ms" << endl;
				}
			//tophat
				{
					start = static_cast<double>(getTickCount());
					morphologyEx(src_cpu, dst_cpu, MORPH_TOPHAT, element );
					time = ((double)getTickCount() - start) / getTickFrequency();
					cout << "cpu about tophat runtime:" << time*1000 << "ms" << endl;
					
					Ptr<cuda::Filter> filter = cuda::createMorphologyFilter(MORPH_TOPHAT, src_gpu.type(), element);
					start = static_cast<double>(getTickCount());					
					filter->apply(src_gpu, dst_gpu);
					time = ((double)getTickCount() - start) / getTickFrequency();
					cout << "gpu about tophat runtime:" << time*1000 << "ms" << endl;
				}
			//blackhat
				{
					start = static_cast<double>(getTickCount());
					morphologyEx(src_cpu, dst_cpu, MORPH_BLACKHAT, element );
					time = ((double)getTickCount() - start) / getTickFrequency();
					cout << "cpu about blackhat runtime:" << time*1000 << "ms" << endl;
					
					
					Ptr<cuda::Filter> filter = cuda::createMorphologyFilter(MORPH_BLACKHAT, src_gpu.type(), element);
					start = static_cast<double>(getTickCount());
					filter->apply(src_gpu, dst_gpu);
					time = ((double)getTickCount() - start) / getTickFrequency();
					cout << "gpu about blackhat runtime:" << time*1000 << "ms" << endl;
				}
		//Filter2D LinearFilter
			{
				Mat KernRobert = (Mat_<char>(2,2) << -1, 0,
													0, 1);
			
				start = getTickCount();
				filter2D(src_cpu, dst_cpu, src_cpu.depth(), KernRobert);
				time = (getTickCount() - start) / getTickFrequency();
				cout << "cpu time about Filter2D: " << time * 1000 <<"ms" << endl;
				
				Ptr<cuda::Filter> filter = cuda::createLinearFilter(src_gpu.type(), src_gpu.type(), KernRobert);
				start = getTickCount();
				filter->apply(src_gpu, dst_gpu);
				time = (getTickCount() - start) / getTickFrequency();
				cout << "gpu time about LinearFilter : " << time * 1000 << "ms"<< endl;
			}
			
			//sobel
			{
				start = static_cast<double>(getTickCount());
				Sobel(src_cpu, dst_cpu, CV_16S, 1, 0);
				time = ((double)getTickCount() - start) / getTickFrequency();
				cout << "cpu runtime about sobel: " << time*1000 << "ms" << endl;

				Ptr<cuda::Filter> filter = cuda::createSobelFilter(src_gpu.type(), dst_gpu.type(), 1, 0);
				start = static_cast<double>(getTickCount());
				filter->apply(src_gpu, dst_gpu);
				time = (getTickCount() - start) / getTickFrequency();
				cout << "gpu run time about sobel: " << time * 1000 << "ms" << endl;

			}
			//scharr
			{
				start = static_cast<double>(getTickCount());
				Scharr(src_cpu, dst_cpu, src_cpu.depth(), 1, 0);
				time = ((double)getTickCount() - start) / getTickFrequency();
				cout << "cpu runtime about scharr: " << time*1000 << "ms" << endl;

				Ptr<cuda::Filter> filter = cuda::createScharrFilter(src_gpu.type(), src_gpu.type(),1, 0 );
				start = static_cast<double>(getTickCount());
				filter->apply(src_gpu, dst_gpu);
				time = ((double)getTickCount() - start)/getTickFrequency();
				cout << "gpu time about scharr: " << time * 1000 << "ms" << endl;
			}
			//Laplacian
			{
				start = static_cast<double>(getTickCount());
				Laplacian(src_cpu, dst_cpu, src_cpu.depth(), 3);
				time = ((double)getTickCount() - start) / getTickFrequency();
				cout << "cpu runtime about laplacian: " << time*1000 << "ms" << endl;
				
				Ptr<cuda::Filter> filter = cuda::createLaplacianFilter(src_gpu.type(), src_gpu.type(), 3);
				start = static_cast<double>(getTickCount());
				filter->apply(src_gpu, dst_gpu);
				time = ((double)getTickCount() - start) / getTickFrequency();
				cout << "gpu runtime about laplacian: " << time*1000 << "ms" << endl;
			}
	return 0;
}
