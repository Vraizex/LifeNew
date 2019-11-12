#pragma once
#define _USE_MATH_DEFINES
#define HARD 1  // Жесткий
#define SOFT 2  // Мягкий
#define GARROT 3  // Фильтр Гаррота
#define MAX_ITERS 10000
#define ALPHA 0

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/video/tracking.hpp"
#include <opencv2/highgui.hpp>

#include <iostream>
#include <cmath>
#include <limits>
#include <vector>
#include <fstream> 
#include <sstream>
#include <random>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
using namespace cv;

class Window
{
public:
	Window(std::string const& title) : _title(title), _is_opened(false) 
	{
		namedWindow(_title, cv::WINDOW_NORMAL);
	}
	void show(cv::Mat const& m, bool wait_flag = false) 
	{
		imshow(_title, m);
		_is_opened = true;
		if (wait_flag) {
			waitKey(0);
		}
	}
	static void wait(int timeout = 0) 
	{
		waitKey(timeout);
	}
	~Window() 
	{
		if (_is_opened) 
		{
			destroyWindow(_title);
		}
	}
protected:
	std::string _title;
	bool _is_opened;

	//EXAMPLE 
	//Mat new_sobel4 = MatrixGrad(dst, 10);
	//Window win_sobel4("Sobel ABS5");
	//win_sobel4.show(new_sobel4);
};

//Methods release
Mat_<Vec3b> k_means(Mat_<Vec3b> & src);

cv::Mat NewSobol(cv::Mat& img);
cv::Mat NewShar(cv::Mat& img);
cv::Mat NewGradientPrevitta(cv::Mat& img);

cv::Mat newfil(cv::Mat& img);
cv::Mat MatrixGrad(cv::Mat& img, int h);
cv::Mat calcHough(cv::Mat& img);
cv::Mat cvHaarWavelet(cv::Mat &img, cv::Mat &dst, int NIter);


cv::Mat calcLoGDiskretWeights(cv::Mat& img);


cv::Mat cvInvHaarWavelet(Mat &src, Mat &dst, int NIter, int SHRINKAGE_TYPE = 0, float SHRINKAGE_T = 50); // Вейвлет-преобразование
cv::Mat calcKircsha(cv::Mat& img, int k, int z);
cv::Mat calcRobinsone(cv::Mat& img, int k);

pair<cv::Mat, cv::Mat> calc3x3Gradient(cv::Mat& img);

cv::Mat MarrHildeth(cv::Mat& img, float sigm);

cv::Mat NewFilter(cv::Mat& img);
