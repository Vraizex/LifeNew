#pragma once
#include "Pix.h"


static int xGradient(Mat const& img, int x, int y) //Gradient dx
{
	return img.at<uchar>(y - 1, x - 1) +
		2 * img.at<uchar>(y, x - 1) +
		img.at<uchar>(y + 1, x - 1) -
		img.at<uchar>(y - 1, x + 1) -
		2 * img.at<uchar>(y, x + 1) -
		img.at<uchar>(y + 1, x + 1);
}

static int yGradient(Mat const& img, int x, int y) //Gradient dy
{
	return img.at<uchar>(y - 1, x - 1) +
		2 * img.at<uchar>(y - 1, x) +
		img.at<uchar>(y - 1, x + 1) -
		img.at<uchar>(y + 1, x - 1) -
		2 * img.at<uchar>(y + 1, x) -
		img.at<uchar>(y + 1, x + 1);
}


namespace IP
{
	struct Thresholds
	{
		int threshold_min;
		int threshold_max;
	};

	cv::Mat sobol(cv::Mat& img)
	{
		cv::Mat dst(img.size(), img.type(), Scalar(0));
		int gx, gy, sum;

		for (int y = 0; y < img.rows; y++)
		{
			for (int x = 0; x < img.cols; x++)
			{
				dst.at<uchar>(y, x) = 0.0;
			}
		}
		for (int y = 1; y < img.rows - 1; y++)
		{
			for (int x = 1; x < img.cols - 1; x++)
			{
				gx = xGradient(img, x, y);
				gy = yGradient(img, x, y);
				sum = abs(gx) + abs(gy);
				sum = sum > 255 ? 255 : sum;
				sum = sum < 0 ? 0 : sum;
				dst.at<uchar>(y, x) = sum;
			}
		}
		return dst;
	}

	cv::Mat Threshold(cv::Mat const& img, Thresholds const& threshold)
	{
		cv::Mat mat_start(img.size(), img.type(), Scalar(0));

		for (int x = 1; x < img.cols - 1; x++)
		{
			for (int y = 1; y < img.rows - 1; y++)
			{
				Point curr_point(x, y);
				if (threshold.threshold_min < mat_start.at<uint8_t>(curr_point) && mat_start.at<uint8_t>(curr_point) < threshold.threshold_max)
				{
					mat_start.at<uint8_t>(curr_point) = threshold.threshold_max;
				}
				else
				{
					mat_start.at<uint8_t>(curr_point) = 0;
				}
			}
		}
		return mat_start;
	}
}


