
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui.hpp>

#include <Windows.h>

#include<iostream>
#include<stdlib.h>
#include<stdio.h>
#include<cmath>

using namespace std;
using namespace cv;

//__global__ void kernel(void)
//{
//
//}

void waitImage(string const& title, Mat const& img)
{
	namedWindow(title, WINDOW_NORMAL);
	imshow(title, img);
	waitKey();
	destroyWindow(title);
}

static void drawHsv(const Mat& flow, Mat& bgr) {
	//extract x and y channels
	Mat xy[2]; //X,Y
	split(flow, xy);

	//calculate angle and magnitude
	Mat magnitude, angle, hsv;
	cartToPolar(xy[0], xy[1], magnitude, angle, true);

	//translate magnitude to range [0;1]
	double mag_max;
	minMaxLoc(magnitude, 0, &mag_max);
	magnitude.convertTo(
		magnitude,    // output matrix
		-1,           // type of the ouput matrix, if negative same type as input matrix
		1.0 / mag_max // scaling factor
	);


	//build hsv image
	Mat _hsv[3];
	_hsv[0] = angle;
	_hsv[1] = magnitude;
	_hsv[2] = cv::Mat::ones(angle.size(), CV_32F);

	merge(_hsv, 3, hsv);
	//convert to BGR and show
	cvtColor(hsv, bgr, COLOR_HSV2BGR);
}

static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, double scale, int step, const Scalar& color)
{
	for (int y = 0; y < cflowmap.rows; y += step)
		for (int x = 0; x < cflowmap.cols; x += step)
		{
			const Point2f& fxy = flow.at<Point2f>(y, x) * scale;
			line(cflowmap, Point(x, y), Point(cvRound(x + fxy.x), cvRound(y + fxy.y)),
				color);
			circle(cflowmap, Point(x, y), 2, color, -1);
		}
}




int main(/*void*/)
{
	
	//kernel <<<1, 1 >>> ();
	Mat img = imread("C:\\Users\\User\\Downloads\\Lenna.png", 0);
	if (!img.data) // Yes or no image loading;
	{
		return -1;
	}
	Mat sub_mat(img.size(), CV_8UC1);
	int neighPixel;
	int maxPix;
	int N = 0;   // Step for image;
	int K;
	int z = 1;
	int ddepth = CV_16S;

	
	for (int x = 1; x < img.cols - 1; x++)   //delta[i,j] = x[i + 1, j] - x[i , j]; 
	{
		for (int y = 1; y < img.rows - 1; y++)
		{
			Point curr_point(x, y);
			auto delta1 = img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y));
			sub_mat.at<uint8_t>(curr_point) = delta1;
		}
	
	}

	GaussianBlur(sub_mat, sub_mat, Size(25, 25), 2.5);
	waitImage("Gradiend Field", sub_mat);
	
	
	
	for (int i = 1; i < sub_mat.rows - 1; i++) // search pixel
	{
		for (int j = 1; j < sub_mat.cols - 1; j++)
			for (int N = -z; N <= z; N++)
			{
				{
					neighPixel = sub_mat.at<uint8_t>(i + N, j + N);
					int maxPix = sub_mat.at<uint8_t>(0, 0);
					if (sub_mat.at<uint8_t>(i, j) > maxPix)
					{
						maxPix = sub_mat.at<uint8_t>(i, j);
						neighPixel = maxPix;
					}

				}
			}
		
	}
	waitImage("Used matrix 3X3", sub_mat);
	

	Sobel(sub_mat, sub_mat, ddepth, 0,1,3,10,1, Border_Default);
	waitImage("Sobel", sub_mat);
	//erode(sub_mat, sub_mat, getStructuringElement(MORPH_RECT, Size(3, 3)));
	//Canny(sub_mat, sub_mat, 100, 200);
	//Laplacian(sub_mat,sub_mat, CV_8UC1,0);
	//waitImage("New_pic", sub_mat);
	
	return 0;


}