#pragma once
#include "Pix.h"

cv::Mat dilateXY(cv::Mat& img);
cv::Mat dilate3X3Y(cv::Mat& img);
cv::Mat dilateGOA(cv::Mat& img, int k);
cv::Mat dilateAndErozia(cv::Mat& img, int k, int z);
cv::Mat EroziaAndDilate(cv::Mat& img, int k, int z);
cv::Mat dilateY(cv::Mat& img);
cv::Mat dilateX(cv::Mat& img);
cv::Mat dilateMXN(cv::Mat& img, int k, int z);
cv::Mat dilateEroziaLevel(cv::Mat& img, int k);