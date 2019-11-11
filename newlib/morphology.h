#pragma once

#include "Pix.h"

using namespace std;
using namespace cv;


namespace morphology
{
struct Kernel
{
int start_size;
int end_size;
};

cv::Mat dilate(cv::Mat const&  img, Kernel const& kernel)// int start_size_dilate, int end_size_dilate)
{
cv::Mat mat_start(img.size(), img.type(), Scalar(0));
for (int x = 1; x < img.cols - 1; x++)
{
for (int y = 1; y < img.rows - 1; y++)
{
Point curr_point(x, y);
float dx = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y))) / 2;
float dy = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x, y + 1))) / 2;
float sobol = (sqrt(pow(dx, 2) + pow(dy, 2))) - 250;
mat_start.at<uint8_t>(curr_point) = static_cast<uint8_t>(sobol - 6) * 10;
if (30 < mat_start.at<uint8_t>(curr_point) && mat_start.at<uint8_t>(curr_point) < 255)
{
mat_start.at<uint8_t>(curr_point) = 255;
}
else
{
mat_start.at<uint8_t>(curr_point) = 0;
}
//Delete only Pixels
if (mat_start.at<uint8_t>(Point(x, y)) == 2)
{
mat_start.at<uint8_t>(Point(x, y)) = 0;
}
if (mat_start.at<uint8_t>(curr_point) == 255)
{
for (int l = x - kernel.start_size; l <= x + kernel.start_size; l++)
{
for (int m = y - kernel.end_size; m <= y + kernel.end_size; m++)
{
if (l >= 0 &&
m >= 0 &&
l < mat_start.cols &&
m < mat_start.rows &&
mat_start.at<uint8_t>(Point(l, m)) == 0)
{
mat_start.at<uint8_t>(Point(l, m)) = 2;
}
}
}
}
}
}
for (int x = 0; x < img.cols; x++)
{
for (int y = 0; y < img.rows; y++)
{
if (mat_start.at<uint8_t>(Point(x, y)) == 2)
{
mat_start.at<uint8_t>(Point(x, y)) = 255;
}
}
}
return mat_start;
}

cv::Mat eroze(cv::Mat const& img, Kernel const& kernel)
{
cv::Mat mat_start(img.size(), img.type(), Scalar(0));
for (int x = 1; x < img.cols - 1; x++)
{
for (int y = 1; y < img.rows - 1; y++)
{
Point curr_point(x, y);
float dx = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y))) / 2;
float dy = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x, y + 1))) / 2;
float sobol = (sqrt(pow(dx, 2) + pow(dy, 2))) - 250;
mat_start.at<uint8_t>(curr_point) = static_cast<uint8_t>(sobol - 6) * 10;

if (30 < mat_start.at<uint8_t>(curr_point) && mat_start.at<uint8_t>(curr_point) < 255)
{
mat_start.at<uint8_t>(curr_point) = 255;
}
else
{
mat_start.at<uint8_t>(curr_point) = 0;
}
if (mat_start.at<uint8_t>(curr_point) == 0)
{
for (int l = x - kernel.start_size; l <= x + kernel.start_size; l++)
{
for (int m = y - kernel.end_size; m <= y + kernel.end_size; m++)
{
if (l >= 0 &&
m >= 0 &&
l < mat_start.cols &&
m < mat_start.rows &&
mat_start.at<uint8_t>(Point(l, m)) == 255)
{
mat_start.at<uint8_t>(Point(l, m)) = 2;
}
}
}
}
}
}
for (int x = 0; x < img.cols; x++)
{
for (int y = 0; y < img.rows; y++)
{
if (mat_start.at<uint8_t>(Point(x, y)) == 2)
{
mat_start.at<uint8_t>(Point(x, y)) = 0;
}
}
}
return mat_start;
}
};

