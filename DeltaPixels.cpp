
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

void waitImage(string const& title, Mat const& img)
{
	namedWindow(title, cv::WINDOW_NORMAL);
	imshow(title, img);
	waitKey();
	destroyWindow(title);
}


int main()
{
	
	Mat img = imread("C:\\Users\\User\\Downloads\\Lenna.png", 0);
	Mat sub_mat(img.size(), CV_8UC1);
	
	for (int x = 1; x < img.cols - 1; x++)
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
	
	/*int neighPixel;
	int maxPix;
	for (int i = 1; i < sub_mat.rows - 1; i++) {
		for (int j = 1; j < sub_mat.cols - 1; j++) {
			
			
			neighPixel = 0;
			if (sub_mat.at<uint8_t>(i, j) ) {
				
				if (sub_mat.at<uint8_t>(i - 1, j - 1) < maxPix) {
					neighPixel++;
				}
				if (sub_mat.at<uint8_t>(i - 1, j) < maxPix) {
					neighPixel++;
				}
				if (sub_mat.at<uint8_t>(i - 1, j + 1) < maxPix) {
					neighPixel++;
				}
				if (sub_mat.at<uint8_t>(i + 1, j) < maxPix) {
					neighPixel++;
				}
				if (sub_mat.at<uint8_t>(i + 1, j - 1) < maxPix) {
					neighPixel++;
				}
				if (sub_mat.at<uint8_t>(i + 1, j + 1) < maxPix) {
					neighPixel++;
				}
				if (sub_mat.at<uint8_t>(i , j - 1) < maxPix) {
					neighPixel++;
				}				
				if (neighPixel == 1) {
					neighPixel == 0;
				}
			}
		}
	}*/


	
	erode(sub_mat, sub_mat, getStructuringElement(MORPH_RECT, Size(5, 5)));

	/*Canny(sub_mat, sub_mat, 100, 200);*/
	//Laplacian(sub_mat,sub_mat, CV_8UC1,0);
	waitImage("New_pic", sub_mat);
	
	return 0;


}

