
#define _USE_MATH_DEFINES

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui.hpp>


#include <iostream>
#include <cmath>
#include <limits>
#include "Pixs.h"

using namespace std;
using namespace cv;

void waitImage(string const& title, Mat const& img)
{
	namedWindow(title, WINDOW_NORMAL);
	imshow(title, img);
	waitKey();
	destroyWindow(title);
}

cv::Mat calcHGradient(cv::Mat& img);
std::pair<cv::Mat, cv::Mat> calc3x3Gradient(cv::Mat& img);

int main()
{
	try {
		setlocale(LC_ALL, "Russian");
		std::string file_path = "C:\\Users\\User\\Downloads\\Lenna.png";
		Mat img = imread(file_path, 0);
		if (!img.data) // no image loading;
		{
			throw std::system_error(errno, std::system_category(), file_path);
		}

		Mat mat_horizontal = calcHGradient(img);
		waitImage("Gradiend Field horizontal", mat_horizontal);

		auto mat3x3 = calc3x3Gradient(img);
		waitImage("Gradiend Field 3x3 module", mat3x3.first);
		waitImage("Gradiend Field 3x3 angle", mat3x3.second);

/*
		//GaussianBlur(sub_mat, sub_mat, Size(25, 25), 2.5);
		

		erode(sub_mat, sub_mat, getStructuringElement(MORPH_RECT, Size(5, 5)));
		waitImage("Angle", sub_mat);

		//Canny(sub_mat, sub_mat, 100, 200);
		//waitImage("New_pic", sub_mat);

		double max_angle = numeric_limits<double>::min();
		double min_angle = numeric_limits<double>::max();


		for (int x = 1; x < sub_mat.cols - 1; x++)
		{
			for (int y = 1; y < sub_mat.rows - 1; y++)
			{
				//for (int N = -1; N < 2; N++)


				Point curr_point(x, y);

				uint8_t neighPixel1 = sub_mat.at<uint8_t>(x, y);
				uint8_t neighPixel2 = sub_mat.at<uint8_t>(x + 1, y + 1);

				double b = pow((pow(neighPixel1, 2) - pow(neighPixel2, 2)), 1 / 2);

				cout << " b = " << b << endl;


				double angle = atan2(sub_mat.at<uint8_t>(x) - sub_mat.at<uint8_t>(x + 1), sub_mat.at<uint8_t>(y) - sub_mat.at<uint8_t>(y + 1));
				double angles = angle * 180 / M_PI;

				max_angle = max(angle, max_angle);
				min_angle = min(angle, min_angle);

				uint8_t val = static_cast<uint8_t>((angle + 3.2) * 35);

				sub_mat.at<uint8_t>(curr_point) = val;

			}
		}

		cout << "max angle: " << min_angle << " : " << max_angle << endl;
		waitImage("Used matrix 3X3", sub_mat);
*/
	}
	catch (std::exception const& ex) {
		std::cout << "fail: " << ex.what() << std::endl;
	}
}

cv::Mat calcHGradient(cv::Mat& img)
{
	cv::Mat sub_mat(img.size(), CV_8UC1, Scalar(0));
	for (int x = 1; x < img.cols - 1; x++)
	{
		for (int y = 1; y < img.rows - 1; y++)
		{
			Point curr_point(x, y);
			sub_mat.at<uint8_t>(curr_point) = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y)) + 255) / 2;
		}
	}

	return sub_mat;
}


std::pair<cv::Mat, cv::Mat> calc3x3Gradient(cv::Mat& img)
{
	cv::Mat mat_mod(img.size(), CV_8UC1, Scalar(0));
	cv::Mat mat_angle(img.size(), CV_8UC1, Scalar(0));
	for (int x = 1; x < img.cols - 1; x++)
	{
		for (int y = 1; y < img.rows - 1; y++)
		{
			for (int N = -1; N <= 1; N++)
			{
				Point curr_point(x, y);			
				
				mat_mod.at<uint8_t>(curr_point) = (pow(pow(img.at<uint8_t>(Point(x,y)), 2) - pow(img.at<uint8_t>(Point(x + N, y + N)), 2), 1 / 2) + 255) / 2;
				
				double angle = atan2(mat_angle.at<uint8_t>(Point(x + N)) - mat_angle.at<uint8_t>(Point(x)), mat_angle.at<uint8_t>(Point(y + N)) - mat_angle.at<uint8_t>(Point(y)));
				double angles = angle * 180 / M_PI;
				uint8_t val = static_cast<uint8_t>((angle + 3.2) * 35);
				mat_angle.at<uint8_t>(curr_point) = val;
			



			}
			

			/*double angle = atan2(mat_angle.at<uint8_t>(x) - mat_angle.at<uint8_t>(x + 1), mat_angle.at<uint8_t>(y) - mat_angle.at<uint8_t>(y + 1));*/
			/*double angles = angle * 180 / M_PI;
			
			uint8_t val = static_cast<uint8_t>((angle + 3.2) * 35);
			mat_angle.at<uint8_t>(curr_point) = val;*/
		}
	}

	return {mat_mod, mat_angle};
}
