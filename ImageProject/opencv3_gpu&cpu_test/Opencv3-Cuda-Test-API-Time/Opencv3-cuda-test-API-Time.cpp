# include <opencv2/core/core.hpp>
# include <opencv2/highgui/highgui.hpp>
# include <opencv2/imgproc/imgproc.hpp>
# include <iostream>
# include <time.h>
# include <opencv2/core/utility.hpp>
# include <opencv2/cudaarithm.hpp>
# include <opencv2/cudabgsegm.hpp>
# include <opencv2/cudacodec.hpp>
# include <opencv2/cudafeatures2d.hpp>
# include <opencv2/cudafilters.hpp>
# include <opencv2/cudaimgproc.hpp>
//# include <opencv2/cudalegacy.hpp>
# include <opencv2/cudaobjdetect.hpp>
# include <opencv2/cudaoptflow.hpp>
# include <opencv2/cudastereo.hpp>
# include <opencv2/cudawarping.hpp>
//# include <opencv2/cudev.hpp>
using namespace std;
using namespace cv;
using namespace cv::cuda;

int main(int argc, char **argv)
{

	Mat srcImg_0, srcImg, dstImg, warp_mat, warp_mat_invert, grayImg, patch;
	Point2f src_point[4], dst_point[4];
	GpuMat srcImg_gpu_0, srcImg_gpu, dstImg_gpu, grayImg_gpu, patch_gpu;
	GpuMat warp_mat_x_gpu, warp_mat_y_gpu;
	double speed1, speed2, speedUp;
	int i, j;
	long start, finish, s, f;
	float total_time, t;
	float total_time0, total_time1;
	//!			error  argc deal with
	if ( 2!= argc )
	{
		cout << "Usage:./Opencv3-cuda-test-API-Time test-image" <<endl;
		return -1;
	}
	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5), Point(2, 2));

	srcImg_0 = imread(argv[1]);
	cv::cvtColor(srcImg_0, srcImg, CV_RGB2GRAY);
	srcImg_gpu.upload(srcImg);
	srcImg_gpu_0.upload(srcImg_0);

	warp_mat = getRotationMatrix2D(Point(0, 0), 30, 1);

//!	author:liusheng

// ********************    Affine Transform  *********
	cuda::resetDevice();
				s = getTickCount();
				cv::warpAffine( srcImg, dstImg, warp_mat, srcImg.size());   									// warpAffine()  CPU
				f = getTickCount();
				t = ( ( f - s ) / getTickFrequency()) * 1000.0;
				cout << "warpAffine  "<< endl;;
				cout << "CPU Time: " << t << " ms  " <<endl;
				speed1 = t;

				s = getTickCount();
				cuda::warpAffine( srcImg_gpu, dstImg_gpu, warp_mat, srcImg_gpu.size());   							// warpAffine() GPU
				f = getTickCount();
				t = ( ( f - s ) / getTickFrequency()) * 1000.0;
				cout << "GPU Time: " << t << " ms"<< endl;
				speed2 = t;
				speedUp = speed1 / speed2;
				cout << "SpeedUp: " << speedUp << endl << endl;

			// *******************  Rotate Transform  **************
				s = getTickCount();
				cv::warpAffine( srcImg, dstImg, warp_mat, srcImg.size());   									// rotate CPU
				f = getTickCount();
				t = ( ( f - s ) / getTickFrequency()) * 1000.0;
				cout << "rotate " << endl;
				cout << "CPU Time: " << t << " ms" << endl;
				speed1 = t;
				s = getTickCount();
				cuda:: rotate( srcImg_gpu, dstImg_gpu, srcImg.size(), 30, 0, 0);
				f = getTickCount();
				t = ( ( f - s ) / getTickFrequency()) * 1000.0;
				speed2 = t;
				speedUp = speed1 / speed2;
				cout << "GPU Time: " << t << " ms" << endl;
				cout << "SpeedUp: " << speedUp << endl << endl;

			// ******************  perspective AffineTransform *********
				src_point[0] = Point2f(0, 0);
				src_point[1] = Point2f(0, 426);
				src_point[2] = Point2f(640, 0);
				src_point[3] = Point2f(640, 426);

				dst_point[0] = Point2f(50,100);
				dst_point[1] = Point2f(100,400);
				dst_point[2] = Point2f(320,20);
				dst_point[3] = Point2f(450,200);

				s = getTickCount();
				warp_mat = getPerspectiveTransform(src_point, dst_point);           // getPerspectiveTransform()   CPU
				f = getTickCount();
				t = ( ( f - s ) / getTickFrequency()) * 1000.0;
				cout << "getPerspectiveTransform" << endl;
				cout << "CPU Time: " << t << " ms" << endl;
				speed1 = t;

				s = getTickCount();
				cuda::buildWarpPerspectiveMaps( warp_mat, 0, srcImg.size(), warp_mat_x_gpu, warp_mat_y_gpu);//buildWarpPerspectiveMaps()    GPU
				f = getTickCount();
				t = ( ( f - s ) / getTickFrequency()) * 1000.0;
				cout << "GPU Time: " << t << " ms" << endl;
				speed2 = t;
				speedUp = speed1 / speed2;
				cout << "SpeedUp: " << speedUp << endl << endl;

				s = getTickCount();
				cv::warpPerspective( srcImg, dstImg, warp_mat, Size(srcImg.cols , srcImg.rows));   			//  warpPerspective()     CPU
				f = getTickCount();
				t = ( ( f -s ) / getTickFrequency()) * 1000.0;
				cout << "warpPerspective" << endl;
				cout << "CPU Time: " << t << " ms" << endl;
				speed1 = t;

				s = getTickCount();
				cuda::warpPerspective( srcImg_gpu, dstImg_gpu, warp_mat, srcImg.size());
				f = getTickCount();																			// warpPerspective()   GPU
				t = ( ( f - s ) / getTickFrequency()) * 1000.0;
				cout << "GPU Time: " << t << " ms" << endl;
				speed2 = t;
				speedUp = speed1 / speed2;
				cout << "SpeedUp: " << speedUp << endl << endl;
				finish = clock();
				total_time =  (finish - start) / getTickFrequency() * 1000.0;
			// ******************   remap   ******************
				Mat warp_mat_x;
				Mat warp_mat_y;
				warp_mat_x.create( srcImg.size(), CV_32FC1 );
				warp_mat_y.create( dstImg.size(), CV_32FC1 );
				for ( i = 0; i < srcImg.rows; i++)
					for( j = 0; j < srcImg.cols; j++)
					{
						warp_mat_x.at<float>(i,j) = j * 2.48544 + 0.621359 * i - 372.816;
						warp_mat_x.at<float>(i,j) = 1.40621 * j + 2.48155 * i - 636.932;
					}


				s = getTickCount();
				cv::remap( srcImg, dstImg, warp_mat_x, warp_mat_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar( 0, 0, 0) );  				// remap()  CPU
				f = getTickCount();
				t = ( ( f - s ) / getTickFrequency()) * 1000.0;
				cout << "remap" << endl;
				cout << "CPU Time: " <<  t << " ms" << endl;
				speed1 = t;

				warp_mat_x_gpu.upload(warp_mat_x);
				warp_mat_y_gpu.upload(warp_mat_y);
				s = getTickCount();
				cuda::remap( srcImg_gpu, dstImg_gpu, warp_mat_x_gpu, warp_mat_y_gpu, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar( 0, 0, 0) );   //  remap() GPU
				f = getTickCount();
				t = ( ( f - s ) / getTickFrequency()) * 1000.0;
				cout << "GPU Time: " << t << " ms" << endl;
				speed2 = t;
				speedUp = speed1 / speed2;
				cout << "SpeedUp: " << speedUp << endl;
				cout << endl;
				Mat warp_mat_x_new, warp_mat_y_new;
			// *****************  resize  **********************
				s = getTickCount();
				cv::resize(srcImg, dstImg, Size(0, 0), 0.5, 0.5, INTER_LINEAR);   													// resize()  CPU
				f = getTickCount();
				t = ( ( f - s ) / getTickFrequency()) * 1000.0;
				speed1 = t;
				cout << "resize" << endl;
				cout << "CPU Time: " << t <<  "ms" <<endl;
				s = getTickCount();
				cuda::resize(srcImg_gpu, dstImg_gpu, Size(0, 0), 0.5, 0.5, INTER_LINEAR);   										// resize()  GPU
				f = getTickCount();
				t = ( ( f - s ) / getTickFrequency()) * 1000.0;
				cout << "GPU Time: " << t << "ms" <<endl;
				speed2 = t;
				speedUp = speed1 / speed2;
				cout << "SpeedUp: " << speedUp << endl << endl;

			// *****************  adaptiveThreshold ************
			// *****************  cvtColor  ********************
				s = getTickCount();
				cv::cvtColor(srcImg_0, dstImg, CV_BGR2GRAY);   // cvtColor()     CPU
				f = getTickCount();
				t = ( ( f - s) / getTickFrequency()) * 1000.0;
				cout << "cvtColor" << endl;
				cout << "CPU Time: " << t << " ms" << endl;
				speed1 = t;

				s = getTickCount();
				cuda::cvtColor(srcImg_gpu_0, dstImg_gpu, CV_BGR2GRAY);  // cvtColor()    GPU
				f = getTickCount();
				t = ( ( f - s) / getTickFrequency()) * 1000.0;
				cout << "GPU Time: " << t << " ms" << endl;
				speed2 = t;
				speedUp = speed1 / speed2;
				cout << "SpeedUp: " << speedUp << endl << endl;

			// ****************   distanceTransform ***********
				s = getTickCount();
				cv::threshold(srcImg, dstImg, 150, 255, THRESH_BINARY);  														 // threshold()   CPU
				f = getTickCount();
				t = ( ( f - s ) / getTickFrequency()) * 1000.0;
				cout << "threshold" << endl;
				cout << "CPU Time: " << t << " ms" << endl;
				speed1 = t;

				s = getTickCount();
				cuda::threshold( srcImg_gpu, dstImg_gpu, 150.0, 255.0, THRESH_BINARY ); 										 // threshold()   GPU
				f = getTickCount();
				t = ( ( f - s ) / getTickFrequency()) * 1000.0;
				speed2 = t;
				cout << "GPU Time: " << t << " ms" << endl;
				speedUp = speed1 / speed2;
				cout << "SpeedUp: " << speedUp << endl << endl;
			// ****************   floodFill   ****************
			// ****************   integral  ******************

				s = getTickCount();
				cv::integral(srcImg, dstImg);      // integral()    CPU
				f = getTickCount();
				t = ( ( f - s ) / getTickFrequency()) * 1000.0;
				cout << "integral" << endl;
				cout << "CPU Time: " << t << " ms" << endl;
				speed1 = t;

				s = getTickCount();
				cuda::integral( srcImg_gpu, dstImg_gpu );   // integral()    GPU
				f = getTickCount();
				t = ( ( f - s ) / getTickFrequency()) * 1000.0;
				cout << "GPU Time: " << t <<  " ms" << endl;
				speed2 = t;
				speedUp = speed1 / speed2;
				cout << "SpeedUp: " << speedUp << endl << endl;
				finish = clock();
				total_time =  (finish - start) / getTickFrequency() * 1000.0;
			// ****************  threshold  ******************
				Mat imgs[1] = srcImg;
				int channels[] = {0, 1, 2};
				MatND hist;
				int histSize[] = {256, 256, 256};
				float range[] = { 0, 255 };
				const float *ranges[]={range};
				s = getTickCount();
				cv::calcHist(imgs, 1, channels, Mat(), hist, 1, histSize, ranges, true, false);  							// calcHist()    CPU
				f = getTickCount();
				t = ((f - s) / getTickFrequency()) * 1000.0;
				cout << "calcHist" << endl;
				cout << "CPU Time: " << t << " ms" << endl;
				speed1 = t;

				GpuMat hist_gpu;
				s = getTickCount();
				cuda::calcHist( srcImg_gpu, hist_gpu);                                     									 // calcHist    GPU
				f = getTickCount();
				t = ((f - s) / getTickFrequency()) * 1000.0;
				cout << "GPU Time: " << t << " ms" << endl;
				speed2 = t;
				speedUp = speed1 / speed2;
				cout << "SpeedUp: " << speedUp << endl << endl;

				// *************************** calcBackProject
				s = getTickCount();
				cv::normalize( srcImg, dstImg, 53, 125, NORM_MINMAX, -1 );     												 // normalize()  CPU
				f = getTickCount();
				t = ((f - s) / getTickFrequency()) * 1000.0;
				cout << "normalize" << endl;
				cout << "CPU Time: " << t << " ms" << endl;
				speed1 = t;

				s = getTickCount();
				cuda::normalize( srcImg_gpu, dstImg_gpu, 0, 255, NORM_MINMAX, -1);   					// normalize()  GPU
				f = getTickCount();
				t = ((f - s) / getTickFrequency()) * 1000.0;
				cout << "GPU Time: " <<  t << " ms" << endl;
				speed2 = t;
				speedUp = speed1 / speed2;
				cout << "SpeedUp: " << speedUp << endl << endl;

			// *************************equalizeHist************

				s = getTickCount();
				cv::equalizeHist( srcImg, dstImg);     // equalizeHist()   CPU
				f = getTickCount();
				t = ((f - s) / getTickFrequency()) * 1000.0;
				cout << "equalizeHist" << endl;
				cout << "CPU Time: " << t << " ms" << endl;
				speed1 = t;

				s = getTickCount();
				cuda::equalizeHist( srcImg_gpu, dstImg_gpu);// equalizeHist()   GPU
				f = getTickCount();
				t = ((f - s) / getTickFrequency()) * 1000.0;
				cout << "GPU Time: " << t << " ms" << endl;
				speed2 = t;
				speedUp = speed1 / speed2;
				cout << "SpeedUp: " << speedUp << endl << endl;
				cout << endl;


				if(srcImg_gpu.channels() == 3)
				{
					cuda::GpuMat src4ch;
					cuda::cvtColor(srcImg_gpu, src4ch, COLOR_BGR2RGBA);
					srcImg_gpu = src4ch;
				}

			//boxFilter
				{
					cout << "boxFilter"<<endl;
					start = getTickCount();
					boxFilter(srcImg, dstImg, -1, Size(5, 5));
					total_time0 = (getTickCount() - start) / getTickFrequency();
					cout << "CPU Time :" << total_time0*1000 << "ms" << endl;

					Ptr<cuda::Filter> filter = cuda::createBoxFilter(srcImg_gpu.type(), srcImg_gpu.type(), Size(5, 5));
					start = getTickCount();
					filter->apply(srcImg_gpu, dstImg_gpu);
					total_time1 = (getTickCount() - start) / getTickFrequency();
					cout << "GPU Time : " << total_time1 * 1000 << " ms" << endl;
					cout << "SpeedUp:" << total_time0 / total_time1 << endl;
					cout << endl;
				}

			//blur
				{
					cout << "blur" <<endl;
					start = static_cast<double>(getTickCount());
					blur(srcImg, dstImg, Size(5, 5) );
					total_time0 = (getTickCount() - start) / getTickFrequency();
					cout << "CPU Time" << total_time0*1000 << "ms" << endl;
					cout << endl;
				}
			//Gaussian
				{
					cout << "GaussianBlur" <<endl;
					start = static_cast<double>(getTickCount());
					GaussianBlur(srcImg, dstImg, Size(3,3), 0, 0);
					total_time0 = (getTickCount() - start) / getTickFrequency();
					cout << "CPU Time : " << total_time0 * 1000 << " ms" << endl;

					Ptr<cuda::Filter> filter = cuda::createGaussianFilter(srcImg_gpu.type(), srcImg_gpu.type(), Size(3,3), 0, 0);
					start = static_cast<double>(getTickCount());
					filter->apply(srcImg_gpu, dstImg_gpu);
					total_time1 = (getTickCount() - start)/getTickFrequency();
					cout << "GPU Time : " << total_time1 * 1000 << " ms" << endl;
					cout << "SpeedUp:" << total_time0 / total_time1 << endl;
					cout << endl;
				}
				//medianBlur
				{
					cout << "medianBlur" <<endl;
					start = static_cast<double>(getTickCount());
					medianBlur(srcImg, dstImg, 7);
					total_time = (getTickCount() - start) / getTickFrequency();
					cout << "CPU Time : " << total_time0 * 1000 << " ms" << endl;
					cout << endl;
				}
				//bilateraFilter
				{
					cout <<"bilateralFilter" <<endl;
					start = static_cast<double>(getTickCount());
					cv::bilateralFilter(srcImg, dstImg, 25, 25*2, 25/2);
					total_time0 = (getTickCount() - start) / getTickFrequency();
					cout << "CPU Time : " << total_time0 * 1000 << " ms" << endl;

					start = static_cast<double>(getTickCount());
					cuda::bilateralFilter(srcImg_gpu, dstImg_gpu, 25, 25*2, 25/2);
					total_time1 = (getTickCount() - start) / getTickFrequency();
					cout << "GPU Time : " << total_time1 * 1000 << " ms" << endl;
					cout << "SpeedUp:" << total_time0 / total_time1 << endl;
					cout << endl;
				}

			//dilate
				{
					cout << "dilate"<<endl;
					start = static_cast<double>(getTickCount());
					//morphologyEx(srcImg, dstImg, MORPH_DILATE, element );
					dilate(srcImg, dstImg, element);
					total_time0 = (getTickCount() - start) / getTickFrequency();
					cout << "CPU Time : " << total_time0 * 1000 << " ms" << endl;

					Ptr<cuda::Filter> filter = cuda::createMorphologyFilter(MORPH_DILATE, srcImg_gpu.type(), element);
					start = static_cast<double>(getTickCount());
					filter->apply(srcImg_gpu, dstImg_gpu);
					total_time1 = (getTickCount() - start) / getTickFrequency();
					cout << "GPU Time : " << total_time1 * 1000 << " ms" << endl;
					cout << "SpeedUp:" << total_time0 / total_time1 << endl;
					cout << endl;
				}
			//erode
				{
					cout << "erode" <<endl;
					start = static_cast<double>(getTickCount());
					//morphologyEx(srcImg, dstImg, MORPH_ERODE, element );
					erode(srcImg, dstImg, element);
					total_time0 = (getTickCount() - start) / getTickFrequency();
					cout << "CPU Time : " << total_time0 * 1000 << " ms" << endl;


					Ptr<cuda::Filter> filter = cuda::createMorphologyFilter(MORPH_ERODE, srcImg_gpu.type(), element);
					start = static_cast<double>(getTickCount());
					filter->apply(srcImg_gpu, dstImg_gpu);
					total_time1 = (getTickCount() - start) / getTickFrequency();
					cout << "GPU Time : " << total_time1 * 1000 << " ms" << endl;
					cout << "SpeedUp:" << total_time0 / total_time1 << endl;
					cout << endl;
				}
			//open
				{
					cout << "open" <<endl;
					start = static_cast<double>(getTickCount());
					morphologyEx(srcImg, dstImg, MORPH_OPEN, element );
					total_time0 = (getTickCount() - start) / getTickFrequency();
					cout << "CPU Time : " << total_time0 * 1000 << " ms" << endl;

					Ptr<cuda::Filter> filter = cuda::createMorphologyFilter(MORPH_OPEN, srcImg_gpu.type(), element);
					start = static_cast<double>(getTickCount());
					filter->apply(srcImg_gpu, dstImg_gpu);
					total_time1 = (getTickCount() - start) / getTickFrequency();
					cout << "GPU Time : " << total_time1 * 1000 << " ms" << endl;
					cout << "SpeedUp:" << total_time0 / total_time1 << endl;
					cout << endl;
				}
			//close
				{
					cout << "close" <<endl;
					start = static_cast<double>(getTickCount());
					morphologyEx(srcImg, dstImg, MORPH_CLOSE, element );
					total_time0 = (getTickCount() - start) / getTickFrequency();
					cout << "CPU Time : " << total_time0 * 1000 << " ms" << endl;


					Ptr<cuda::Filter> filter = cuda::createMorphologyFilter(MORPH_CLOSE, srcImg_gpu.type(), element);
					start = static_cast<double>(getTickCount());
					filter->apply(srcImg_gpu, dstImg_gpu);
					total_time1 = (getTickCount() - start) / getTickFrequency();
					cout << "GPU Time : " << total_time1 * 1000 << " ms" << endl;
					cout << "SpeedUp:" << total_time0 / total_time1 << endl;
					cout << endl;
				}
			//Morphological Gradient
				{
					cout << "Morphological Gradient"<<endl;
					start = static_cast<double>(getTickCount());
					morphologyEx(srcImg, dstImg, MORPH_GRADIENT, element );
					total_time0 = (getTickCount() - start) / getTickFrequency();
					cout << "CPU Time : " << total_time0 * 1000 << " ms" << endl;


					Ptr<cuda::Filter> filter = cuda::createMorphologyFilter(MORPH_GRADIENT, srcImg_gpu.type(), element);
					start = static_cast<double>(getTickCount());
					filter->apply(srcImg_gpu, dstImg_gpu);
					total_time1 = (getTickCount() - start) / getTickFrequency();
					cout << "GPU Time : " << total_time1 * 1000 << " ms" << endl;
					cout << "SpeedUp:" << total_time0 / total_time1 << endl;
					cout << endl;
				}
			//tophat
				{
					cout <<"tophat"<<endl;
					start = static_cast<double>(getTickCount());
					morphologyEx(srcImg, dstImg, MORPH_TOPHAT, element );
					total_time0 = (getTickCount() - start) / getTickFrequency();
					cout << "CPU Time : " << total_time0 * 1000 << " ms" << endl;

					Ptr<cuda::Filter> filter = cuda::createMorphologyFilter(MORPH_TOPHAT, srcImg_gpu.type(), element);
					start = static_cast<double>(getTickCount());
					filter->apply(srcImg_gpu, dstImg_gpu);
					total_time1 = (getTickCount() - start) / getTickFrequency();
					cout << "GPU Time : " << total_time1 * 1000 << " ms" << endl;
					cout << "SpeedUp:" << total_time0 / total_time1 << endl;
					cout << endl;
				}
			//blackhat
				{
					cout <<"blackhat"<<endl;
					start = static_cast<double>(getTickCount());
					morphologyEx(srcImg, dstImg, MORPH_BLACKHAT, element );
					total_time0 = (getTickCount() - start) / getTickFrequency();
					cout << "CPU Time : " << total_time0 * 1000 << " ms" << endl;

					Ptr<cuda::Filter> filter = cuda::createMorphologyFilter(MORPH_BLACKHAT, srcImg_gpu.type(), element);
					start = static_cast<double>(getTickCount());
					filter->apply(srcImg_gpu, dstImg_gpu);
					total_time1 = (getTickCount() - start) / getTickFrequency();
					cout << "GPU Time : " << total_time1 * 1000 << " ms" << endl;
					cout << "SpeedUp:" << total_time0 / total_time1 << endl;
					cout << endl;
				}
			//Filter2D LinearFilter
				{
					Mat KernRobert = (Mat_<char>(2,2) << -1, 0,
														0, 1);

					cout <<"Filter2D"<<endl;
					start = getTickCount();
					filter2D(srcImg, dstImg, srcImg.depth(), KernRobert);
					total_time0 = (getTickCount() - start) / getTickFrequency();
					cout << "CPU Time : " << total_time0 * 1000 << " ms" << endl;

					Ptr<cuda::Filter> filter = cuda::createLinearFilter(srcImg_gpu.type(), srcImg_gpu.type(), KernRobert);
					start = getTickCount();
					filter->apply(srcImg_gpu, dstImg_gpu);
					total_time1 = (getTickCount() - start) / getTickFrequency();
					cout << "GPU Time : " << total_time1 * 1000 << " ms" << endl;
					cout << "SpeedUp:" << total_time0 / total_time1 << endl;
					cout << endl;
				}

				//sobel
				{
					cout << "Sobel" << endl;
					start = static_cast<double>(getTickCount());
					Sobel(srcImg, dstImg, CV_16S, 1, 0);
					total_time0 = (getTickCount() - start) / getTickFrequency();
					cout << "CPU Time : " << total_time0 * 1000 << " ms" << endl;

					Ptr<cuda::Filter> filter = cuda::createSobelFilter(srcImg_gpu.type(), dstImg_gpu.type(), 1, 0);
					start = static_cast<double>(getTickCount());
					filter->apply(srcImg_gpu, dstImg_gpu);
					total_time1 = (getTickCount() - start) / getTickFrequency();
					cout << "GPU Time : " << total_time1 * 1000 << " ms" << endl;
					cout << "SpeedUp:" << total_time0 / total_time1 << endl;
					cout << endl;
				}
				//scharr
				{
					cout << "Scharr" << endl;
					start = static_cast<double>(getTickCount());
					Scharr(srcImg, dstImg, srcImg.depth(), 1, 0);
					total_time0 = (getTickCount() - start) / getTickFrequency();
					cout << "CPU Time : " << total_time0 * 1000 << " ms" << endl;

					Ptr<cuda::Filter> filter = cuda::createScharrFilter(srcImg_gpu.type(), srcImg_gpu.type(),1, 0 );
					start = static_cast<double>(getTickCount());
					filter->apply(srcImg_gpu, dstImg_gpu);
					total_time1 = (getTickCount() - start)/getTickFrequency();
					cout << "GPU Time : " << total_time1 * 1000 << " ms" << endl;
					cout << "SpeedUp:" << total_time0 / total_time1 << endl;
					cout << endl;
				}
				//Laplacian
				{
					cout << "Laplacian" << endl;
					start = static_cast<double>(getTickCount());
					Laplacian(srcImg, dstImg, srcImg.depth(), 3);
					total_time0 = (getTickCount() - start) / getTickFrequency();
					cout << "CPU Time : " << total_time0 * 1000 << " ms" << endl;

					Ptr<cuda::Filter> filter = cuda::createLaplacianFilter(srcImg_gpu.type(), srcImg_gpu.type(), 3);
					start = static_cast<double>(getTickCount());
					filter->apply(srcImg_gpu, dstImg_gpu);
					total_time1 = (getTickCount() - start) / getTickFrequency();
					cout << "GPU Time : " << total_time1 * 1000 << " ms" << endl;
					cout << "SpeedUp:" << total_time0 / total_time1 << endl;
					cout << endl;
				}

				//					harris detect
				{
					//!						cpu		process
					cout << "harris detect " << endl;
					start = getTickCount();
					//!		test cornerHarris test
					cornerHarris(srcImg, dstImg,2,3,0.04,BORDER_DEFAULT);
					total_time0 = (getTickCount() - start) / getTickFrequency();
					cout << "CPU Time : " << total_time0 * 1000 << " ms" << endl;

					//!						gpu process image
					start = getTickCount();
					Ptr<cuda::CornernessCriteria> harris_gpu = cuda::createHarrisCorner(srcImg_gpu.type(),2,3,0.04,BORDER_DEFAULT);
					harris_gpu->compute(srcImg_gpu,dstImg_gpu);
					total_time1 = (getTickCount() - start) / getTickFrequency();
					cout << "GPU Time : " << total_time1 * 1000 << " ms" << endl;
					cout << "SpeedUp:" << total_time0 / total_time1 << endl;
					cout << endl;
				}

				//!					Canny detect
				{
					//!				cpu test
					cout << "Canny detect "<<endl;
					start = getTickCount();
					cv::Canny(srcImg, dstImg, 3, 9, 3);
					total_time0 = (getTickCount() - start) / getTickFrequency();
					cout << "CPU Time : " << total_time0 * 1000 << " ms" << endl;

					//!				gpu test
					start = getTickCount();
					Ptr<cuda::CannyEdgeDetector> Canny = cuda::createCannyEdgeDetector(3,9,3);
					Canny->detect(srcImg_gpu, dstImg_gpu);
					total_time1 = (getTickCount() - start) / getTickFrequency();
					cout << "GPU Time : " << total_time1 * 1000 << " ms" << endl;
					cout << "SpeedUp:" << total_time0 / total_time1 << endl;
					cout << endl;
				}

					//!				gpu test
				//					HoughLines
				{
					//!				cpu  test
					cout << "HoughLines " << endl;
					vector<Vec2f> lines_cpu;
					start = getTickCount();
					cv::Canny(srcImg, dstImg, 100, 200, 3);
					cv::HoughLines(dstImg, lines_cpu, 1, CV_PI / 180, 150, 0, 0);
					total_time0 = (getTickCount() - start) / getTickFrequency();
					cout << "CPU Time : " << total_time0 * 1000 << " ms" << endl;

					//!				gpu test
					Ptr<cuda::HoughLinesDetector> houghlines = cuda::createHoughLinesDetector(1, (float) (CV_PI / 180.0f), 150);
					start = getTickCount();
					houghlines->detect(srcImg_gpu, dstImg_gpu);
					total_time1 = (getTickCount() - start) / getTickFrequency();
					cout << "GPU Time : " << total_time1 * 1000 << " ms" << endl;
					cout << "SpeedUp:" << total_time0 / total_time1 << endl;
					cout << endl;
				}

				//!					HoughLinesP test
				{
					//			cput test
					cout << "HoughLinesP " <<endl;
					vector<Vec4i> lines_cpu;
					start = getTickCount();
					cv::Canny(srcImg, dstImg, 100, 200, 3);
					cv::HoughLinesP(dstImg, lines_cpu, 1, CV_PI / 180, 50, 60, 5);
					total_time0 = (getTickCount() - start) / getTickFrequency();
					cout << "CPU Time : " << total_time0 * 1000 << " ms" << endl;

					//			gpu test
					Ptr<cuda::HoughSegmentDetector> hough = cuda::createHoughSegmentDetector(1.0f, (float) (CV_PI / 180.0f), 50, 5);
					start = getTickCount();
					hough->detect(srcImg_gpu, dstImg_gpu);
					total_time1 = (getTickCount() - start) / getTickFrequency();
					cout << "GPU Time : " << total_time1 * 1000 << " ms" << endl;
					cout << "SpeedUp:" << total_time0 / total_time1 << endl;
					cout << endl;
				}

				//!					goodFeaturesToTrack test
				{
						//!			cpu test
						cout << "goodFeaturesToTrack " << endl;
						double	minDistance = 10;
						int		max_corner = 8000;
						//!		convert image to gray
						//!		cpu test
						vector<Point2f> corners;
						double	qualityLevel = 0.01;
						int		blockSize = 3;
						bool	useHarrisDetector = false;
						double	k = 0.04;
						start = getTickCount();
						goodFeaturesToTrack( srcImg,
									corners,
									max_corner,
									qualityLevel,
									minDistance,
									Mat(),
									blockSize,
									useHarrisDetector,
									k );
						total_time0 = (getTickCount() - start) / getTickFrequency();
						cout << "CPU Time : " << total_time0 * 1000 << " ms" << endl;

					//!			gpu test
					cv::Ptr<cuda::CornersDetector> d_detector = cuda::createGoodFeaturesToTrackDetector(srcImg_gpu.type(),max_corner,0.01,minDistance );
					start = getTickCount();
					d_detector->detect(srcImg_gpu,dstImg_gpu);
					total_time1 = (getTickCount() - start) / getTickFrequency();
					cout << "GPU Time :" << total_time1 * 1000 << " ms" << endl;
					cout << "SpeedUp  :" << total_time0 / total_time1 << endl;
				}
	//GGh#endif
	return 0;

}
