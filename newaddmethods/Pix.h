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
Mat dilateXY(Mat& img);
Mat dilate3X3Y(Mat& img);
Mat dilateGOA(Mat& img, int k);
Mat dilateAndErozia(Mat& img, int k, int z);
Mat EroziaAndDilate(Mat& img, int k, int z);
Mat Bin(Mat& img);
Mat dilateMXN(Mat& img, int k, int z);
Mat dilateEroziaLevel(Mat& img, int k);
Mat dilateGOAEroze(Mat& img, int k);
cv::Mat calcLoG(cv::Mat& img);
cv::Mat calcLoGDiskret(cv::Mat& img);
cv::Mat calcLoGDiskretWeights(cv::Mat& img);
cv::Mat calcLoGDiskretWeightsProg(cv::Mat& img);
cv::Mat calc3x3GradientSOBOLBinAndMatrix(cv::Mat& img);
cv::Mat calc3x3GradientPrevitta(cv::Mat& img);
cv::Mat calcRobertsa(cv::Mat& img);
cv::Mat calcHGradient(Mat& img);
Mat dilateY(Mat& img);
Mat dilateX(Mat& img);
cv::Mat cvHaarWavelet(Mat &img, Mat &dst, int NIter);
pair<cv::Mat, cv::Mat> calc3x3Gradient(cv::Mat& img);
cv::Mat cvInvHaarWavelet(Mat &src, Mat &dst, int NIter, int SHRINKAGE_TYPE = 0, float SHRINKAGE_T = 50); // Вейвлет-преобразование


cv::Mat NewSobol(cv::Mat& img);
cv::Mat NewShar(cv::Mat& img);
cv::Mat NewGradientPrevitta(cv::Mat& img);
cv::Mat NewLoG(cv::Mat& img);
Mat BinandDeleteOnlyPixels(Mat& img);
Mat lagrange2(Mat& img);
Mat Catmull_Rom(Mat& img);
Mat B_Spline(Mat& img);
Mat dilateBiz(Mat& img, int k);
Mat Splines(Mat& img);
Mat LOGLith(Mat& img);

Mat dilateAndEroziaProbMatrix(Mat& img, int k, int z);
Mat dilateAndEroziaRobinsone(Mat& img, int k);
Mat MarrHildeth(Mat& img, float sigm);

Mat MarrHildethNew(Mat& img, float sigm);
Mat NewFilter(Mat& img);