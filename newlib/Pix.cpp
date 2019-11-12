
#include "Pix.h"

#include <random>
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include <vector>

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

void transpose(int matrix[3][3])
{
	int t;
	for (int i = 0; i < 3; ++i)
	{
		for (int j = i; j < 3; ++j)
		{
			t = matrix[i][j];
			matrix[i][j] = matrix[j][i];
			matrix[j][i] = t;
		}
	}
}

std::pair<cv::Mat, cv::Mat> calc3x3Gradient(cv::Mat& img)
{
	cv::Mat mat_mod(img.size(), img.type(), Scalar(0));
	cv::Mat mat_angle(img.size(), img.type(), Scalar(0));


	float max_angle = numeric_limits<float>::min();
	float min_angle = numeric_limits<float>::max();

	for (int x = 1; x < img.cols - 1; x++)
	{
		for (int y = 1; y < img.rows - 1; y++)
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

cv::Mat cvHaarWavelet(cv::Mat &img, cv::Mat &dst, int NIter) // Вейвлет-преобразование
{
	
	float c,b, dh, dv, dd;

	for (int k = 0; k < NIter; k++)
	{
		for (int y = 0; y < (img.rows >> (k + 1)); y++)
		{
			for (int x = 0; x < (img.cols >> (k + 1)); x++)
			{
				c = (img.at<uint8_t>( y, x) + img.at<uint8_t>( y,  x + 1) + img.at<uint8_t>( y + 1,  x) + img.at<uint8_t>( y + 1,  x + 1));
				dst.at<uint8_t>(y, x) = c;
				dst.at<uint8_t>(y + (img.rows >> (k + 1)), x) = c;
							
				dh = (img.at<uint8_t>( y,  x) + img.at<uint8_t>( y + 1,  x) - img.at<uint8_t>( y,  x + 1) - img.at<uint8_t>( y + 1, 2 * x + 1));
				dst.at<uint8_t>(y, x + (img.cols >> (k + 1))) = dh;
				
				dv = (img.at<uint8_t>(2 * y, 2 * x) + img.at<uint8_t>(2 * y, 2 * x + 1) - img.at<uint8_t>(2 * y + 1, 2 * x) - img.at<uint8_t>(2 * y + 1, 2 * x + 1));
				dst.at<uint8_t>(y + (img.rows >> (k + 1)), x) = dv;

				dd = (img.at<uint8_t>(2 * y, 2 * x) - img.at<uint8_t>(2 * y, 2 * x + 1) - img.at<uint8_t>(2 * y + 1, 2 * x) + img.at<uint8_t>(2 * y + 1, 2 * x + 1));
				dst.at<uint8_t>(y + (img.rows >> (k + 1)), x + (img.cols >> (k + 1))) = dd;
			
			}
		}
		dst.copyTo(img);

		return dst;
	}
}

cv::Mat cvInvHaarWavelet(cv::Mat &img, cv::Mat &dst, int NIter, int SHRINKAGE_TYPE, float SHRINKAGE_T )
{
	float c, dh, dv, dd;
	// NIter - Количество итераций преобразования

	for (int k = NIter; k > 0; k--)
	{
		for (int y = 0; y < (img.rows >> k); y++)
		{
			for (int x = 0; x < (img.cols >> k); x++)
			{
				c = img.at<uint8_t>(y, x);
				dh = img.at<uint8_t>(y, x + (img.cols >> k));
				dv = img.at<uint8_t>(y + (img.rows >> k), x);
				dd = img.at<uint8_t>(y + (img.rows >> k), x + (img.cols >> k));

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
				dst.at<uint8_t>(y * 2, x * 2) = 0.5*(c + dh + dv + dd);
				dst.at<uint8_t>(y * 2, x * 2 + 1) = 0.5*(c - dh + dv - dd);
				dst.at<uint8_t>(y * 2 + 1, x * 2) = 0.5*(c + dh - dv - dd);
				dst.at<uint8_t>(y * 2 + 1, x * 2 + 1) = 0.5*(c - dh - dv + dd);
			}
		}
		Mat C = img(Rect(0, 0, img.cols >> (k - 1), img.rows >> (k - 1)));
		Mat D = dst(Rect(0, 0, img.cols >> (k - 1), img.rows >> (k - 1)));
		D.copyTo(C);
	}
	return img;
}







cv::Mat calcKircsha(cv::Mat& img, int k, int z) //operator Kircsha
{
	cv::Mat mat_start(img.size(), img.type(), Scalar(0));

	float E[3][3] = {
	{ -3, -3, 5 },
	{ -3, 5, 5 },
	{ -3, -3, 5 } };
	float NE[3][3] = {
	{ -3, 5, 5 },
	{ -3, 0, 5 },
	{ -3, -3, -3 } };
	float N[3][3] = {
	{ 5, 5, 5 },
	{ -3, 0, -3 },
	{ -3, -3, -3 } };
	float NW[3][3] = {
	{ 5, 5, -3 },
	{ 5, 0, -3 },
	{ -3, -3, -3 } };
	float W[3][3] = {
	{ 5, -3, -3 },
	{ 5, 0, -3 },
	{ 5, -3, -3 } };
	float SW[3][3] = {
	{ -3, -3, -3 },
	{ 5, 0, -3 },
	{ 5, 5, -3 } };
	float S[3][3] = {
	{ -3, -3, -3 },
	{ -3, 0, -3 },
	{ 5, 5, 5 } };
	float SE[3][3] = {
	{ -3, -3, 5 },
	{ -3, 0, 5 },
	{ -3, 5, 5 } };

	for (int x = 1; x < img.cols - 1; x++)
	{
		for (int y = 1; y < img.rows - 1; y++)
		{
			Point curr_point(x, y);
			float dx4 = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x - 1, y)))*N[3][3];
			float dy4 = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x - 1, y + 1)))*NE[3][3];
			float dy1 = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x, y + 1)))*E[3][3];
			float dx2 = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y + 1)))*SE[3][3];
			float dx1 = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y)))*S[3][3];
			float dy2 = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y - 1)))*SW[3][3];
			float dx3 = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x, y - 1)))*W[3][3];
			float dy3 = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x - 1, y - 1)))*NW[3][3];
			float priv = sqrt(pow(dx1, 2) + pow(dy1, 2) + pow(dx2, 2) + pow(dy2, 2) + pow(dx3, 2) + pow(dy3, 2) + pow(dx4, 2) + pow(dy4, 2));

			mat_start.at<uint8_t>(curr_point) += 255 * log10f((img.at<uint8_t>(curr_point)) + 1);
			mat_start.at<uint8_t>(curr_point) = static_cast<uint8_t>(priv);
			
			if (10 <= mat_start.at<uint8_t>(curr_point) && mat_start.at<uint8_t>(curr_point) < 30)
			{
				mat_start.at<uint8_t>(curr_point) = 20;
			}
			else if (0 <= mat_start.at<uint8_t>(curr_point) && mat_start.at<uint8_t>(curr_point) <=10)
			{
				mat_start.at<uint8_t>(curr_point) = 0;
			}
			if (mat_start.at<uint8_t>(curr_point) == 20)
			{
				for (int l = x - k; l <= x + k; l++)
				{
					for (int m = y - k; m <= y + k; m++)
					{
						if (l >= 0 &&
							m >= 0 &&
							l < mat_start.cols &&
							m < mat_start.rows &&
							mat_start.at<uint8_t>(Point(l, m)) == 0)
							mat_start.at<uint8_t>(Point(l, m)) = 2;
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
				mat_start.at<uint8_t>(Point(x, y)) = 20;
			}
		}
	}
	return mat_start;
}

cv::Mat calcRobinsone(cv::Mat& img, int k) //operator Robinsone
{
	cv::Mat mat_start(img.size(), img.type(), Scalar(0));

	float E[3][3] = {
	{ -1, 0, 1 },
	{ -2, 0, 2 },
	{ -1, 0, 1 } };
	float NE[3][3] = {
	{ 0, 1, 2 },
	{ -1, 0, 1 },
	{ -2, -1, 0 } };
	float N[3][3] = {
	{ 1, 2, 1 },
	{ 0, 0, 0 },
	{ -1, -2, -1 } };
	float NW[3][3] = {
	{ 2, 1, 0 },
	{ 1, 0, -1 },
	{ -0, -1, -2 } };
	float W[3][3] = {
	{ 1, 0, -1 },
	{ 2, 0, -2 },
	{ 1, 0, -1 } };
	float SW[3][3] = {
	{ 0, -1, -1 },
	{ 1, 0, -1 },
	{ 2, 1, 0 } };
	float S[3][3] = {
	{ -1, -2, -1 },
	{ 0, 0, 0 },
	{ 1, 2, 1 } };
	float SE[3][3] = {
	{ -2, -1, 0 },
	{ -1, 0, 1 },
	{ 0, 1, 2 } };

	for (int x = 1; x < img.cols - 1; x++)
	{
		for (int y = 1; y < img.rows - 1; y++)
		{
			Point curr_point(x, y);
			float dx4 = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x - 1, y)))*N[3][3];
			float dy4 = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x - 1, y + 1)))*NE[3][3];
			float dy1 = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x, y + 1)))*E[3][3];
			float dx2 = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y + 1)))*SE[3][3];
			float dx1 = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y)))*S[3][3];
			float dy2 = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y - 1)))*SW[3][3];
			float dx3 = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x, y - 1)))*W[3][3];
			float dy3 = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x - 1, y - 1)))*NW[3][3];
			float priv = sqrt(pow(dx1, 2) + pow(dy1, 2) + pow(dx2, 2) + pow(dy2, 2) + pow(dx3, 2) + pow(dy3, 2) + pow(dx4, 2) + pow(dy4, 2));

			mat_start.at<uint8_t>(curr_point) += 255 * log10f((img.at<uint8_t>(curr_point)) + 1);
			mat_start.at<uint8_t>(curr_point) = static_cast<uint8_t>(priv);

			if (10 <= mat_start.at<uint8_t>(curr_point) && mat_start.at<uint8_t>(curr_point) < 20)

			{
				mat_start.at<uint8_t>(curr_point) = 20;
			}
			else if (0 <= mat_start.at<uint8_t>(curr_point) && mat_start.at<uint8_t>(curr_point) <= 10)
			{
				mat_start.at<uint8_t>(curr_point) = 0;
			}
			if (mat_start.at<uint8_t>(curr_point) == 20)
			{
				for (int l = x - k; l <= x + k; l++)
				{
					for (int m = y - k; m <= y + k; m++)
					{
						if (l >= 0 &&
							m >= 0 &&
							l < mat_start.cols &&
							m < mat_start.rows &&
							mat_start.at<uint8_t>(Point(l, m)) == 0)
							mat_start.at<uint8_t>(Point(l, m)) = 2;
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
				mat_start.at<uint8_t>(Point(x, y)) = 20;
			}
		}
	}
	return mat_start;
}

cv::Mat MarrHildeth(cv::Mat& img, float sigm)
{
	cv::Mat mat_start(img.size(), img.type(), Scalar(0));
	GaussianBlur(img, img, Size(3, 3), 25);
	//TODO create filter GaussianBlur(Size(3,3), 25);
	for (int x = 1; x < img.cols - 1; x++)
	{
		for (int y = 1; y < img.rows - 1; y++)
		{
			Point curr_point(x, y);
			float marr_hidd =  ((img.at<uint8_t>(Point(x + 1, y)) +
				img.at<uint8_t>(Point(x - 1, y)) +
				img.at<uint8_t>(Point(x , y + 1)) +
				img.at<uint8_t>(Point(x, y - 1)) -
				4 * img.at<uint8_t>(Point(x,y)))/2);
			mat_start.at<uint8_t>(curr_point) = static_cast<uint8_t>(marr_hidd * sigm) ;
		}
	}
	return mat_start;
}

cv::Mat newfil(Mat &img)
{
	cv::Mat mat_start(img.size(), img.type(), Scalar(0));
	cv::Mat mat_sub(img.size(), img.type(), Scalar(0));

	GaussianBlur(img, img, Size(5, 5), 25);
	for (int x = 1; x < img.cols - 1; x++)
	{
		for (int y = 1; y < img.rows - 1; y++)
		{
			Point curr_point(x, y);
			float Ax = img.at<uint8_t>(Point(x + 1, y - 1)) + 
				2 * img.at<uint8_t>(Point(x + 1, y)) +
				img.at<uint8_t>(Point(x + 1, y + 1)) -
				img.at<uint8_t>(Point(x - 1, y - 1)) -
				2 * img.at<uint8_t>(Point(x - 1, y)) -
				img.at<uint8_t>(Point(x - 1, y + 1));
			float Bx = img.at<uint8_t>(Point(x - 1, y + 1)) +
				2 * img.at<uint8_t>(Point(x, y + 1)) +
				img.at<uint8_t>(Point(x + 1, y + 1)) -
				img.at<uint8_t>(Point(x - 1, y - 1)) -
				2 * img.at<uint8_t>(Point(x, y - 1)) -
				img.at<uint8_t>(Point(x + 1, y - 1));
			float sobol = (pow(Ax, 2) + pow(Bx, 2)) / 8;
			mat_sub.at<uint8_t>(curr_point) = static_cast<uint8_t>(sobol);
		}
	}
	return mat_sub;

}

cv::Mat MatrixGrad(cv::Mat& img , int h)
{
	cv::Mat sub_mat(img.size(), img.type(), Scalar(0));
	for (int x = 2; x < img.cols - 2; x++)
	{
		for (int y = 2; y < img.rows - 2; y++)
		{
			Point curr_point(x, y);
			float DminX = (3 * (img.at<uint8_t>(Point(x, y))) - 4 * img.at<uint8_t>(Point(x - 1, y)) + img.at<uint8_t>(Point(x - 2, y))) /2*h;
			float DplusX =(-3 * (img.at<uint8_t>(Point(x, y))) + 4 * img.at<uint8_t>(Point(x + 1, y)) - img.at<uint8_t>(Point(x + 2, y)))/2 * h;
			float DminY = (3 * (img.at<uint8_t>(Point(x, y))) - 4 * img.at<uint8_t>(Point(x, y - 1)) + img.at<uint8_t>(Point(x, y - 2)))/2 * h;
			float DplusY = (-3 * (img.at<uint8_t>(Point(x, y))) + 4 * img.at<uint8_t>(Point(x, y + 1)) - img.at<uint8_t>(Point(x, y + 2)))/2 * h;
			float I = sqrt(pow(max(DminX, -DplusX),2) + pow(max(DminY, -DplusY),2));
			//float weigths = 1 / (0.1 + max(pow(I, k), pow(I, k)));
			sub_mat.at<uint8_t>(curr_point) = static_cast<uint8_t>(I) ;
		}
	}
	return sub_mat;
}

cv::Mat kMeans(cv::Mat& src)
{
	vector<uint8_t> colors;
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			colors.push_back(src.at<uint8_t>(i, j));
			colors.push_back(src.at<uint8_t>(i, j));
			colors.push_back(src.at<uint8_t>(i, j));
		}
	}
	cout << colors.size();
	vector<Point2i> points;
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			points.push_back(Point2d(i, j));
		}
	}
	int n = points.size();
	default_random_engine gen;
	uniform_int_distribution<int> dist_img(0, n);
	int k = 10;
	vector<int> randomses;
	imshow("image", src);
	waitKey();
}

cv::Mat calcHough(cv::Mat& img)
{
	cv::Mat sub_mat(img.size(), img.type(), Scalar(0));
	for (int x = 1; x < img.cols - 1; x++)
	{
		for (int y = 1; y < img.rows - 1; y++)
		{
			Point curr_point(x, y);
			float dx = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y))) / 2;
			float dy = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x, y + 1))) / 2;
			float angle = atan2(dy, dx);
			float distanse = abs(x * cos(angle) - y * sin(angle));
			sub_mat.at<uint8_t>(curr_point) += distanse;
		}
	}
	return sub_mat;
}