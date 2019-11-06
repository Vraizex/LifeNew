#pragma once
#include "Pix.h"


cv::Mat calc3x3GradientSOBOLBinAndMatrix(cv::Mat& img);
pair<cv::Mat, cv::Mat> calc3x3Gradient(cv::Mat& img);
cv::Mat cvInvHaarWavelet(Mat &src, Mat &dst, int NIter, int SHRINKAGE_TYPE = 0, float SHRINKAGE_T = 50); // Вейвлет-преобразование
cv::Mat calcVGradient(cv::Mat& img);
cv::Mat LOGLith(cv::Mat& img);
cv::Mat calcKircsha(cv::Mat& img, int k, int z);
cv::Mat calcRobinsone(cv::Mat& img, int k);
cv::Mat MarrHildeth(cv::Mat& img, float sigm);
cv::Mat MarrHildrethNew(cv::Mat& img, float sigm);
cv::Mat NewFilter(cv::Mat& img);