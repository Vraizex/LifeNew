#pragma once
#define _USE_MATH_DEFINES
#define HARD 1  // Жесткий
#define SOFT 2  // Мягкий
#define GARROT 3  // Фильтр Гаррота

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/video/tracking.hpp"
#include <opencv2/highgui.hpp>

#include <iostream>
#include <cmath>
#include <limits>
#include <stdio.h>
#include <vector>


using namespace std;
using namespace cv;

class Window
{
public:
	Window(std::string const& title) : _title(title), _is_opened(false) {
		namedWindow(_title, cv::WINDOW_NORMAL);
	}
	void show(cv::Mat const& m, bool wait_flag = false) {
		imshow(_title, m);
		_is_opened = true;
		if (wait_flag) {
			waitKey(0);
		}
	}
	static void wait(int timeout = 0) {
		waitKey(timeout);
	}
	~Window() {
		if (_is_opened) {
			destroyWindow(_title);
		}
	}
protected:
	std::string _title;
	bool _is_opened;
};


Mat calc3x3GradientSOBOL(cv::Mat& img);
cv::Mat calcLoG(cv::Mat& img);
cv::Mat calcLoGDiskret(cv::Mat& img);
cv::Mat calcLoGDiskretWeights(cv::Mat& img);
cv::Mat calcLoGDiskretWeightsProg(cv::Mat& img);
Mat calc3x3GradientPrevitta(cv::Mat& img);
Mat calcRobertsa(cv::Mat& img);
Mat calcHGradient(Mat& img);
Mat cvHaarWavelet(Mat &img, Mat &dst, int NIter);
pair<cv::Mat, cv::Mat> calc3x3Gradient(cv::Mat& img);
Mat cvInvHaarWavelet(Mat &src, Mat &dst, int NIter, int SHRINKAGE_TYPE = 0, float SHRINKAGE_T = 50); // Вейвлет-преобразование


