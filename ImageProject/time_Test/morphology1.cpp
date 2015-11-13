
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cv;

static void help()
{

printf("\nShow off image morphology: erosion, dialation, open and close\n"
    "Call:\n   morphology2 [image]\n"
    "This program also shows use of rect, ellipse and cross kernels\n\n");
printf( "Hot keys: \n"
    "\tESC - quit the program\n"
    "\tr - use rectangle structuring element\n"
    "\te - use elliptic structuring element\n"
    "\tc - use cross-shaped structuring element\n"
    "\tSPACE - loop through all the options\n" );
}

cuda::GpuMat src, dst;
Mat cpu_src, cpu_dst;

int element_shape = MORPH_RECT;

//the address of variable which receives trackbar position update
int max_iters = 12;
int open_close_pos = 0;
int erode_dilate_pos = 0;

// callback function for erode/dilate trackbar
static void ErodeDilate(int, void*)
{
    int n = erode_dilate_pos - max_iters;
    int an = n > 0 ? n : -n;
	double start = getTickCount();
    Mat element = getStructuringElement(element_shape, Size(an*2+1, an*2+1), Point(an, an) );
    if( n < 0 ) {
        Ptr<cuda::Filter> erodeFilter = cuda::createMorphologyFilter(MORPH_ERODE, src.type(), element);
		erodeFilter->apply(src, dst);
	}
    else {
        Ptr<cuda::Filter> dilateFilter = cuda::createMorphologyFilter(MORPH_DILATE, src.type(), element);
		dilateFilter->apply(src, dst);
	}
    double timeSec = (getTickCount() - start) / getTickFrequency();
	std::cout << "GPU Time : " << timeSec * 1000 << " ms" << endl;
    imshow("Erode/Dilate",(Mat)dst);
	
	start = getTickCount();
	if (n < 0)
		erode(cpu_src, cpu_dst, element);
	else
		dilate(cpu_src, cpu_dst, element);
	timeSec = (getTickCount() - start) / getTickFrequency();
	std::cout << "CPU Time : " << timeSec * 1000 << " ms" << endl;
	imshow("CPU Erode/Dilate", cpu_dst);
}


int main( int argc, char** argv )
{
    char* filename = argc == 2 ? argv[1] : (char*)"baboon.jpg";
    if (string(argv[1]) == "--help")
    {
        help();
        return -1;
    }

	cpu_src = imread(filename);
    src.upload(imread(filename, 1));
    if (src.empty())
    {
        help();
        return -1;
    }

    //cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());

    //help();


    if (src.channels() == 3)
    {
        // gpu support only 4th channel images
		cuda::GpuMat src4ch;
        cuda::cvtColor(src, src4ch, CV_BGR2BGRA);
        src = src4ch;
    }

    //create windows for output images
    //namedWindow("Open/Close",1);
    //namedWindow("Erode/Dilate",1);

    open_close_pos = erode_dilate_pos = max_iters;
    //createTrackbar("iterations", "Open/Close",&open_close_pos,max_iters*2+1,OpenClose);
    //createTrackbar("iterations", "Erode/Dilate",&erode_dilate_pos,max_iters*2+1,ErodeDilate);

    //for(;;)
    {
        int c;

		erode_dilate_pos = 8;
        //OpenClose(open_close_pos, 0);
        ErodeDilate(erode_dilate_pos, 0);
        c = cvWaitKey(0);

        //if( (char)c == 27 )
            //break;
        /*if( (char)c == 'e' )
            element_shape = MORPH_ELLIPSE;
        else if( (char)c == 'r' )
            element_shape = MORPH_RECT;
        else if( (char)c == 'c' )
            element_shape = MORPH_CROSS;
        else if( (char)c == ' ' )
            element_shape = (element_shape + 1) % 3;*/
    }

    return 0;
}
