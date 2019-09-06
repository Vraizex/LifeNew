#include <cmath>
#include <iostream>
#include "Pix.h"


float sgn(float x) // сигноида
{
	float res = 0;
	if (x == 0)
	{
		res = 0;
	}
	if (x > 0)
	{
		res = 1;
	}
	if (x < 0)
	{
		res = -1;
	}
	return res;
}

float soft_shrink(float d, float T) // Мягкое ослабление
{
	float res;
	if (fabs(d) > T)
	{
		res = sgn(d)*(fabs(d) - T);
	}
	else
	{
		res = 0;
	}
	return res;
}

float hard_shrink(float d, float T) // Жесткое ослабление
{
	float res;
	if (fabs(d) > T)
	{
		res = d;
	}
	else
	{
		res = 0;
	}

	return res;
}


float Garrot_shrink(float d, float T) // Ослабление коэффициентов по Гаррота
{
	float res;
	if (fabs(d) > T)
	{
		res = d - ((T*T) / d);
	}
	else
	{
		res = 0;
	}

	return res;
}


cv::Mat calcHGradient(cv::Mat& img)
{
	cv::Mat sub_mat(img.size(), CV_8UC1, Scalar(0));
	for (int x = 1; x < img.cols - 1; x++)
	{
		for (int y = 1; y < img.rows - 1; y++)
		{
			Point curr_point(x, y);
			sub_mat.at<uint8_t>(curr_point) = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y)) + 255) / 2;
		}
	}
	return sub_mat;
}


std::pair<cv::Mat, cv::Mat> calc3x3Gradient(cv::Mat& img)
{
	cv::Mat mat_mod(img.size(), CV_8UC1, Scalar(0));
	cv::Mat mat_angle(img.size(), CV_8UC1, Scalar(0));

	int cols = img.cols;
	int rows = img.rows;

	float max_angle = numeric_limits<double>::min();
	float min_angle = numeric_limits<double>::max();

	for (int x = 1; x < cols - 1; x++)
	{
		for (int y = 1; y < rows - 1; y++)
		{
			vector<uint8_t> arr;
			for (int i = -1; i <= 1; i++)
			{
				for (int j = -1; j <= 1; j++)
				{
					auto cur = img.at<uint8_t>(Point(j + x, i + y));
					arr.push_back(cur);
				}
			}

			uint8_t const& cnt = arr[4];
			float sum = 0.;

			for (auto const it : arr)
			{
				sum += pow(it - cnt, 2);
			}
			float mod = sqrt(sum);

			Point curr_point(x, y);

			mat_mod.at<uint8_t>(curr_point) = 255 - static_cast<uint8_t>(mod);

			float dx = sqrt(pow(arr[0] - cnt, 2) + pow(arr[3] - cnt, 2) + pow(arr[6] - cnt, 2) + pow(arr[3] - cnt, 2) + pow(arr[5] - cnt, 2) + pow(arr[8] - cnt, 2));
			float dy = sqrt(pow(arr[0] - cnt, 2) + pow(arr[1] - cnt, 2) + pow(arr[2] - cnt, 2) + pow(arr[6] - cnt, 2) + pow(arr[7] - cnt, 2) + pow(arr[8] - cnt, 2));

			float angle_rad = atan2(dy, dx);

			float angle_grad = ((angle_rad * 180. / M_PI) + 3.2);

			max_angle = max(angle_grad, max_angle);
			min_angle = min(angle_grad, min_angle);

			mat_angle.at<uint8_t>(curr_point) = static_cast<uint8_t>(angle_grad);

		}
	}


	cout << "angle: " << min_angle << ", " << max_angle << endl;

	return { mat_mod, mat_angle };
}


Mat cvHaarWavelet(Mat &img, Mat &dst, int NIter) // Вейвлет-преобразование
{
	
	float c, dh, dv, dd;
	int width = img.cols;
	int height = img.rows;
	for (int k = 0; k < NIter; k++)
	{
		for (int y = 0; y < (height >> (k + 1)); y++)
		{
			for (int x = 0; x < (width >> (k + 1)); x++)
			{
				c = (img.at<uint8_t>(2 * y, 2 * x) + img.at<uint8_t>(2 * y, 2 * x + 1) + img.at<uint8_t>(2 * y + 1, 2 * x) + img.at<uint8_t>(2 * y + 1, 2 * x + 1))*0.5;
				dst.at<uint8_t>(y, x) = c;

				dh = (img.at<uint8_t>(2 * y, 2 * x) + img.at<uint8_t>(2 * y + 1, 2 * x) - img.at<uint8_t>(2 * y, 2 * x + 1) - img.at<uint8_t>(2 * y + 1, 2 * x + 1))*0.5;
				dst.at<uint8_t>(y, x + (width >> (k + 1))) = dh;

				dv = (img.at<uint8_t>(2 * y, 2 * x) + img.at<uint8_t>(2 * y, 2 * x + 1) - img.at<uint8_t>(2 * y + 1, 2 * x) - img.at<uint8_t>(2 * y + 1, 2 * x + 1))*0.5;
				dst.at<uint8_t>(y + (height >> (k + 1)), x) = dv;

				dd = (img.at<uint8_t>(2 * y, 2 * x) - img.at<uint8_t>(2 * y, 2 * x + 1) - img.at<uint8_t>(2 * y + 1, 2 * x) + img.at<uint8_t>(2 * y + 1, 2 * x + 1))*0.5;
				dst.at<uint8_t>(y + (height >> (k + 1)), x + (width >> (k + 1))) = dd;
			}
		}
		dst.copyTo(img);

		return dst;
	}
}
Mat cvInvHaarWavelet(Mat &img, Mat &dst, int NIter, int SHRINKAGE_TYPE, float SHRINKAGE_T )
{
	float c, dh, dv, dd;

	int width = img.cols;
	int height = img.rows;

	// NIter - Количество итераций преобразования

	for (int k = NIter; k > 0; k--)
	{
		for (int y = 0; y < (height >> k); y++)
		{
			for (int x = 0; x < (width >> k); x++)
			{
				c = img.at<uint8_t>(y, x);
				dh = img.at<uint8_t>(y, x + (width >> k));
				dv = img.at<uint8_t>(y + (height >> k), x);
				dd = img.at<uint8_t>(y + (height >> k), x + (width >> k));

				// Ослабляем коэффициенты (shrinkage)
				switch (SHRINKAGE_TYPE)
				{
				case HARD:
					dh = hard_shrink(dh, SHRINKAGE_T);
					dv = hard_shrink(dv, SHRINKAGE_T);
					dd = hard_shrink(dd, SHRINKAGE_T);
					break;
				case SOFT:
					dh = soft_shrink(dh, SHRINKAGE_T);
					dv = soft_shrink(dv, SHRINKAGE_T);
					dd = soft_shrink(dd, SHRINKAGE_T);
				case GARROT:
					dh = Garrot_shrink(dh, SHRINKAGE_T);
					dv = Garrot_shrink(dv, SHRINKAGE_T);
					dd = Garrot_shrink(dd, SHRINKAGE_T);
					break;
				}

				//-------------------
				dst.at<uint8_t>(y * 2, x * 2) = 0.5*(c + dh + dv + dd);
				dst.at<uint8_t>(y * 2, x * 2 + 1) = 0.5*(c - dh + dv - dd);
				dst.at<uint8_t>(y * 2 + 1, x * 2) = 0.5*(c + dh - dv - dd);
				dst.at<uint8_t>(y * 2 + 1, x * 2 + 1) = 0.5*(c - dh - dv + dd);
			}
		}
		Mat C = img(Rect(0, 0, width >> (k - 1), height >> (k - 1)));
		Mat D = dst(Rect(0, 0, width >> (k - 1), height >> (k - 1)));
		D.copyTo(C);
	}
	return img;
}
