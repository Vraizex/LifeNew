
#define _USE_MATH_DEFINES

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui.hpp>


#include <iostream>
//#include <stdlib.h>
//#include <stdio.h>
#include <cmath>
#include <limits>

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

	//Mat mask = sub_mat.clone();

	//dilate(mask, mask, Mat());
	//dilate(mask, mask, Mat());
	//erode(mask, mask, Mat());
	//erode(mask, mask, Mat());

	//erode(mask, mask, Mat());
	//erode(mask, mask, Mat());
	//dilate(mask, mask, Mat());
	//dilate(mask, mask, Mat());

	//Mat median;
	//medianBlur(sub_mat, median, 7);

	//Mat resizedIn;
	//Mat resizedMask;
	//Mat resizedMedian;
	//resize(mask, resizedMask, Size(), 0.5, 0.5);
	//resize(median, resizedMedian, Size(), 0.5, 0.5);
	//resize(sub_mat, resizedIn, Size(), 0.5, 0.5);

	//waitImage("sub_mat", resizedIn);
	//waitImage("mask", resizedMask);
	//waitImage("median", resizedMedian);

	for (int i = 1; i < sub_mat.rows - 1; i++) // search pixel
	{
		for (int j = 1; j < sub_mat.cols - 1; j++)
			
			for (int N = -z; N <= z; N++)
			{
				{
					Point curr_point(i, j);
					neighPixel = sub_mat.at<uint8_t>(i + N, j + N);
					int maxPix = sub_mat.at<uint8_t>(0, 0);
					if (sub_mat.at<uint8_t>(i + N, j + N) > maxPix)
					{
						maxPix = sub_mat.at<uint8_t>(i, j);
						neighPixel = maxPix;
						sub_mat.at<uint8_t>(i,j) = neighPixel;
						sub_mat.at<uint8_t>(curr_point) = pow((pow(neighPixel, 2) + pow(maxPix, 2)), 1 / 2);
					}

					
				}
			}
	
	}
	waitImage("Used matrix 3X3", sub_mat);
	
	double max_angle = numeric_limits<double>::min();
	double min_angle = numeric_limits<double>::max();

	for (int i = 1; i < sub_mat.rows - 1; i++) // search pixel
	{
		for (int j = 1; j < sub_mat.cols - 1; j++)
		{
			for (int N = -z; N <= z; N++)
			{
						Point curr_point(i, j);
						
						double angle = atan2(sub_mat.at<uint8_t>(i - N) - sub_mat.at<uint8_t>(i), sub_mat.at<uint8_t>(j - N) - sub_mat.at<uint8_t>(j));
						double angles = angle * 180 / M_PI;

						max_angle = max(angle, max_angle);
						min_angle = min(angle, min_angle);

						if (angles > 1)
						{
							sub_mat.at<uint8_t>(curr_point) = 1;
						}

						else
						{
							sub_mat.at<uint8_t>(curr_point) = 0;
						}
						uint8_t val = static_cast<uint8_t>((angle + 3.2) * 35);
						//std::cout << static_cast<uint32_t>(val) << endl;
						sub_mat.at<uint8_t>(curr_point) = val;// angles;
			
			}
		}
	}

	std::cout << "max angle: " << min_angle << " : " << max_angle << endl;

	waitImage("Angle", sub_mat);

	//Sobel(sub_mat,sub_mat, CV_8UC1, 1, 0, 3, 1, 0, BORDER_DEFAULT);

	//waitImage("Sobel", sub_mat);
	//erode(sub_mat, sub_mat, getStructuringElement(MORPH_RECT, Size(3, 3)));
	//waitImage("Angle", sub_mat);
	//Canny(sub_mat, sub_mat, 100, 200);
	//waitImage("New_pic", sub_mat);


	return 0;


}

}