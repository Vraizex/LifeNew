#pragma once
#include "Pix.h"

cv::Mat calcSOBOL(cv::Mat& img);
cv::Mat calcLoG(cv::Mat& img);
cv::Mat calcPrevitta(cv::Mat& img);
cv::Mat calcRobertsa(cv::Mat& img);
cv::Mat calcHGradient(cv::Mat& img);
cv::Mat NewSobol(cv::Mat& img);
cv::Mat NewShar(cv::Mat& img);
cv::Mat NewGradientPrevitta(cv::Mat& img);
cv::Mat NewLoG(cv::Mat& img);
cv::Mat dilateBiz(cv::Mat& img, int k);
std::pair<cv::Mat, cv::Mat> calc5x5Gradient(cv::Mat& img);
cv::Mat newfil(cv::Mat& img);
cv::Mat MatrixGrad(cv::Mat& img, int h);
cv::Mat calcHough(cv::Mat& img);