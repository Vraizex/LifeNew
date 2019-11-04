#pragma once
#include "Pix.h"

cv::Mat calcLoGDiskret(cv::Mat& img);
cv::Mat calcLoGDiskretWeights(cv::Mat& img);
cv::Mat calcLoGDiskretWeightsProg(cv::Mat& img);
cv::Mat calc3x3GradientSOBOLBinAndMatrix(cv::Mat& img);
cv::Mat cvHaarWavelet(cv::Mat &img, cv::Mat &dst, int NIter);
pair<cv::Mat, cv::Mat> calc3x3Gradient(cv::Mat& img);
cv::Mat cvInvHaarWavelet(Mat &src, Mat &dst, int NIter, int SHRINKAGE_TYPE = 0, float SHRINKAGE_T = 50); // Вейвлет-преобразование
cv::Mat calcVGradient(cv::Mat& img);
cv::Mat BinandDeleteOnlyPixels(cv::Mat& img);
cv::Mat lagrange2(cv::Mat& img);
cv::Mat Catmull_Rom(cv::Mat& img);
cv::Mat B_Spline(cv::Mat& img);
cv::Mat Splines(cv::Mat& img);
cv::Mat LOGLith(cv::Mat& img);
cv::Mat calcKircsha(cv::Mat& img, int k, int z);
cv::Mat calcRobinsone(cv::Mat& img, int k);
cv::Mat MarrHildeth(cv::Mat& img, float sigm);
cv::Mat MarrHildrethNew(cv::Mat& img, float sigm);
cv::Mat NewFilter(cv::Mat& img);