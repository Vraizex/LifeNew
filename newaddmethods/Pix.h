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


cv::Mat calc3x3GradientSOBOL(cv::Mat& img);
cv::Mat dilateXY(cv::Mat& img);
cv::Mat dilate3X3Y(cv::Mat& img);
cv::Mat dilateGOA(cv::Mat& img, int k);
cv::Mat dilateAndErozia(cv::Mat& img, int k, int z);
cv::Mat EroziaAndDilate(cv::Mat& img, int k, int z);
cv::Mat Bin(cv::Mat& img);
cv::Mat dilateMXN(cv::Mat& img, int k, int z);
cv::Mat dilateEroziaLevel(cv::Mat& img, int k);
cv::Mat calcLoG(cv::Mat& img);
cv::Mat calcLoGDiskret(cv::Mat& img);
cv::Mat calcLoGDiskretWeights(cv::Mat& img);
cv::Mat calcLoGDiskretWeightsProg(cv::Mat& img);
cv::Mat calc3x3GradientSOBOLBinAndMatrix(cv::Mat& img);
cv::Mat calc3x3GradientPrevitta(cv::Mat& img);
cv::Mat calcRobertsa(cv::Mat& img);
cv::Mat calcHGradient(cv::Mat& img);
cv::Mat dilateY(cv::Mat& img);
cv::Mat dilateX(cv::Mat& img);
cv::Mat cvHaarWavelet(cv::Mat &img, cv::Mat &dst, int NIter);
pair<cv::Mat, cv::Mat> calc3x3Gradient(cv::Mat& img);
cv::Mat cvInvHaarWavelet(Mat &src, Mat &dst, int NIter, int SHRINKAGE_TYPE = 0, float SHRINKAGE_T = 50); // Вейвлет-преобразование


cv::Mat NewSobol(cv::Mat& img);
cv::Mat NewShar(cv::Mat& img);
cv::Mat NewGradientPrevitta(cv::Mat& img);
cv::Mat NewLoG(cv::Mat& img);
cv::Mat BinandDeleteOnlyPixels(cv::Mat& img);
cv::Mat lagrange2(cv::Mat& img);
cv::Mat Catmull_Rom(cv::Mat& img);
cv::Mat B_Spline(cv::Mat& img);
cv::Mat dilateBiz(cv::Mat& img, int k);
cv::Mat Splines(cv::Mat& img);
cv::Mat LOGLith(cv::Mat& img);

cv::Mat dilateAndEroziaProbMatrix(cv::Mat& img, int k, int z);
cv::Mat dilateAndEroziaRobinsone(cv::Mat& img, int k);
cv::Mat MarrHildeth(cv::Mat& img, float sigm);

cv::Mat MarrHildethNew(cv::Mat& img, float sigm);
cv::Mat NewFilter(cv::Mat& img);