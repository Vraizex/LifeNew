#pragma once
#include "Pix.h"


namespace convolution
{
int reflect(int M, int x) //Отраженная индексация
{
if (x < 0)
{
return -x - 1;
}
if (x >= M)
{
return 2 * M - x - 1;
}
return x;
}

int circular(int M, int x) //Круговая индексация
{
if (x < 0)
return x + M;
if (x >= M)
return x - M;
return x;
}

void noBorderProcessing(Mat & src, Mat & dst, float Kernel[][3]) //пикселями на границе просто пренебрегают
{
float sum;
for (int y = 1; y < src.rows - 1; y++) 
{
for (int x = 1; x < src.cols - 1; x++) 
{
sum = 0.0;
for (int k = -1; k <= 1; k++) 
{
for (int j = -1; j <= 1; j++) 
{
sum += Kernel[j + 1][k + 1] * src.at<uchar>(y - j, x - k);
}
}
dst.at<uchar>(y, x) = sum;
}
}
}

void refletedIndexing(Mat & src, Mat & dst, float Kernel[][3]) //пиксель лежащий вне изображения, т.е. ( x - j, y - k ), отражается обратно в изображение 
{
float sum, x1, y1;
for (int y = 0; y < src.rows; y++) 
{
for (int x = 0; x < src.cols; x++) 
{
sum = 0.0;
for (int k = -1; k <= 1; k++) 
{
for (int j = -1; j <= 1; j++) 
{
x1 = reflect(src.cols, x - j);
y1 = reflect(src.rows, y - k);
sum += Kernel[j + 1][k + 1] * src.at<uchar>(y1, x1);
}
}
dst.at<uchar>(y, x) = sum;
}
}
}

void circularIndexing(Mat & src, Mat  & dst, float Kernel[][3]) //координаты которые выходят за границы изображения переходят на противоположную сторону
{
float sum, x1, y1;
for (int y = 0; y < src.rows; y++) 
{
for (int x = 0; x < src.cols; x++) 
{
sum = 0.0;
for (int k = -1; k <= 1; k++) 
{
for (int j = -1; j <= 1; j++) 
{
x1 = circular(src.cols, x - j);
y1 = circular(src.rows, y - k);
sum += Kernel[j + 1][k + 1] * src.at<uchar>(y1, x1);
}
}
dst.at<uchar>(y, x) = sum;
}
}
}
};

pair<cv::Mat, cv::Mat> calc3x3Gradient(cv::Mat& img);

cv::Mat MarrHildeth(cv::Mat& img, float sigm);

cv::Mat NewFilter(cv::Mat& img);