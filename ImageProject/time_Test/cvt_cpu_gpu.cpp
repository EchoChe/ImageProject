/*
 *            File name               :cvt.cpp
 *            Author                  :kangkangliang
 *            File desc               :
 *            Mail                    :liangkangkang@paag.com
 *            Create time             :2015-10-28
 */

/*!
 *                                             headfile
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iterator>
#include <opencv2/core/core.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;
using namespace cv::cuda;
/*!
 *                                            main function
 */
int
main(int argc,char **argv)
{
	Mat		cpu_src = imread( argv[1] );
	//!		error deal with
	if ( cpu_src.empty() )
	{
		cout << "read image error" <<  endl;
		return -1;
	}
	cout << "the image rows " << cpu_src.rows << endl;
	cout << "the image cols " << cpu_src.cols << endl;
	cout << "the image channel " << cpu_src.channels() << endl;
	Mat		gray;

	//!			cpu  process
    const int64 start1 = getTickCount();
	cv::cvtColor(cpu_src,gray,COLOR_BGR2GRAY);
    const double timeSec1 = (getTickCount() - start1) / getTickFrequency();
    cout << "CPU Time : " << timeSec1 * 1000 << " ms" << endl;
	//!			gpu  process
	GpuMat		gpu_src(cpu_src);
	GpuMat		gpu_gray;
    const int64 start = getTickCount();
	cuda::cvtColor(gpu_src,gpu_gray,COLOR_BGR2GRAY,4);
    const double timeSec = (getTickCount() - start) / getTickFrequency();
    cout << "GPU Time : " << timeSec * 1000 << " ms" << endl;

	return 0;
}
