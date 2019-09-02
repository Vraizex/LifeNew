
#define _USE_MATH_DEFINES

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui.hpp>

#include <Windows.h>

#include<iostream>
#include<stdlib.h>
#include<stdio.h>
#include<math.h>
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


//double angle(const Point& curr_point(x1,y1), const Point& curr_point(x2,y2))
//{
//	double arctan = atan2(x2-x1,y2-y1);
//	if (arctan > 1.0)
//		return 0.0;
//	else if (arctan < -1.0)
//		return M_PI;
//	return atan2(arctan);
//}

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
	int N = 0;   // Step for image;
	int z = 1;



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

	waitImage("Angle", sub_mat);

	//Sobel(sub_mat,sub_mat, CV_8UC1, 1, 0, 3, 1, 0, BORDER_DEFAULT);

	//waitImage("Sobel", sub_mat);
	//erode(sub_mat, sub_mat, getStructuringElement(MORPH_RECT, Size(3, 3)));
	//Canny(sub_mat, sub_mat, 100, 200);
	/*Laplacian(sub_mat,sub_mat, CV_8UC1,0);*/
	waitImage("New_pic", sub_mat);


	return 0;


}