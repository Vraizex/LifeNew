#pragma once
#include "Pix.h"

namespace sobel_image
{
int xGradient(Mat & img, int x, int y) //Gradient dx
{
return img.at<uchar>(y - 1, x - 1) +
2 * img.at<uchar>(y, x - 1) +
img.at<uchar>(y + 1, x - 1) -
img.at<uchar>(y - 1, x + 1) -
2 * img.at<uchar>(y, x + 1) -
img.at<uchar>(y + 1, x + 1);
}

int yGradient(Mat & img, int x, int y) //Gradient dy
{
return img.at<uchar>(y - 1, x - 1) +
2 * img.at<uchar>(y - 1, x) +
img.at<uchar>(y - 1, x + 1) -
img.at<uchar>(y + 1, x - 1) -
2 * img.at<uchar>(y + 1, x) -
img.at<uchar>(y + 1, x + 1);
}

}

cv::Mat calcPrevitta(cv::Mat& img);
cv::Mat calcRobertsa(cv::Mat& img);

cv::Mat NewSobol(cv::Mat& img);
cv::Mat NewShar(cv::Mat& img);
cv::Mat NewGradientPrevitta(cv::Mat& img);

cv::Mat newfil(cv::Mat& img);
cv::Mat MatrixGrad(cv::Mat& img, int h);
cv::Mat calcHough(cv::Mat& img);
cv::Mat cvHaarWavelet(cv::Mat &img, cv::Mat &dst, int NIter);

cv::Mat lagrange(cv::Mat& img);
cv::Mat Catmull_Rom(cv::Mat& img);
cv::Mat B_Spline(cv::Mat& img);
cv::Mat Splines(cv::Mat& img);

cv::Mat calcLoGDiskret(cv::Mat& img);
cv::Mat calcLoGDiskretWeights(cv::Mat& img);
cv::Mat calcLoGDiskretWeightsProg(cv::Mat& img);


cv::Mat cvInvHaarWavelet(Mat &src, Mat &dst, int NIter, int SHRINKAGE_TYPE = 0, float SHRINKAGE_T = 50); // Вейвлет-преобразование
cv::Mat calcKircsha(cv::Mat& img, int k, int z);
cv::Mat calcRobinsone(cv::Mat& img, int k);

