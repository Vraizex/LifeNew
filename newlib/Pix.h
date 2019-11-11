#pragma once
#define _USE_MATH_DEFINES
#define HARD 1  // ∆есткий
#define SOFT 2  // ћ€гкий
#define GARROT 3  // ‘ильтр √аррота
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

cv::Mat Binarizathion(cv::Mat& img);
//Methods release
Mat_<Vec3b> k_means(Mat_<Vec3b> & src);
