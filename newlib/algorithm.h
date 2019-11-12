#pragma once
#include "Pix.h"

using namespace std;
using namespace cv;

namespace
{
	cv::Mat lagrange(cv::Mat const& img)
	{
		cv::Mat mat_start(img.size(), img.type(), Scalar(0));

		for (int x = 2; x < img.cols - 2; x++)
		{
			for (int y = 2; y < img.rows - 2; y++)
			{
				Point curr_point(x, y);
				float t = 3.8f;
				mat_start.at<uint8_t>(curr_point) += 
					(((1 - t)*mat_start.at<uint8_t>(Point(x - 1, y - 1)) +
					3 * t*mat_start.at<uint8_t>(Point(x + 1, y - 1))
					*(1 - t) + 3 * t*t*mat_start.at<uint8_t>(Point(x + 1, y + 1))*
					(1 - t) + t * t*t*mat_start.at<uint8_t>(Point(x - 1, y + 1))
					));
			}
		}
		return mat_start;
	}

	cv::Mat Catmull_Rom(cv::Mat const& img)
	{
		cv::Mat mat_start(img.size(), img.type(), Scalar(0));

		for (int x = 1; x < img.cols - 1; x++)
		{
			for (int y = 1; y < img.rows - 1; y++)
			{
				Point curr_point(x, y);

				float t = 1.5;
				mat_start.at<uint8_t>(curr_point) += 1 / 2 * (-t * (1 - t)*(1 - t))*
					mat_start.at<uint8_t>(Point(x - 1, y - 1)) + (2 - 5 * t*t + 3 * t*t*t)*
					mat_start.at<uint8_t>(Point(x + 1, y - 1)) + t * (1 + 4 * t - 3 * t*t)*
					mat_start.at<uint8_t>(Point(x + 1, y + 1)) - t * t*(1 - t)*
					mat_start.at<uint8_t>(Point(x - 1, y + 1));
			}
		}
		return mat_start;
	}

	cv::Mat B_Spline(cv::Mat const& img)
	{
		cv::Mat mat_start(img.size(), img.type(), Scalar(0));

		for (int x = 2; x < img.cols - 2; x++)
		{
			for (int y = 2; y < img.rows - 2; y++)
			{
				Point curr_point(x, y);

				float t = 0.25;
				float b1 = 15;
				float b2 = 100;
				float alpha = 2 * b1*b1*b1 + 4 * b1*b1 + 4 * b1 + b2 + 2;
				float bet0 = (2 * b1*b1*b1 / alpha)*(1 - t)*(1 - t)*(1 - t);
				float bet1 = (1 / alpha)*(2 * b1*b1*b1*t*(t*t - 3 * t + 3) + 2 * b1*b1*(t*t*t - 3 * t + 2) + 2 * b1 * (t*t*t - 3 * t + 2) + b2 * (2 * t*t*t - 3 * t + 1));
				float bet2 = (1 / alpha)*(2 * b1*b1*t*t*(-t + 3) + 2 * b1*t*(-t * (-t) + 3) + 2 * b2*t*t*(-2 * t + 3) + 2 * (-t * (-t)*(-t) + 1));
				float bet3 = 2 * t*t*t / alpha;
				mat_start.at<uint8_t>(curr_point) +=
					bet0 *
					mat_start.at<uint8_t>(Point(x - 1, y - 1)) +
					bet1 *
					mat_start.at<uint8_t>(Point(x + 1, y - 1)) +
					bet2 *
					mat_start.at<uint8_t>(Point(x + 1, y + 1)) +
					bet3 *
					mat_start.at<uint8_t>(Point(x - 1, y + 1));
			}
		}
		return mat_start;
	}

	cv::Mat Splines(cv::Mat const& img)
	{
		cv::Mat mat_start(img.size(), img.type(), Scalar(0));

		for (int x = 1; x < img.cols - 1; x++)
		{
			for (int y = 1; y < img.rows - 1; y++)
			{
				Point curr_point(x, y);
		
				float t = 2.9f;
				mat_start.at<uint8_t>(curr_point) += 1 / 2 * (-t * (1 - t)*(1 - t))*
					mat_start.at<uint8_t>(Point(x - 1, y - 1)) + (2 - 5 * t*t + 3 * t*t*t)*
					mat_start.at<uint8_t>(Point(x + 1, y - 1)) + t * (1 + 4 * t - 3 * t*t)*
					mat_start.at<uint8_t>(Point(x + 1, y + 1)) - t * t*(1 - t)*
					mat_start.at<uint8_t>(Point(x - 1, y + 1));
			}
		}
		return mat_start;
	}
}