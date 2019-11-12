#pragma once
#include "Pix.h"


static int reflect(int M, int x) //Отраженная индексация
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
};

static int circular(int M, int x) //Круговая индексация
{
	if (x < 0)
		return x + M;
	if (x >= M)
		return x - M;
	return x;
}

namespace convolution
{

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

void circularIndexing(Mat & src, Mat & dst, float Kernel[][3]) //координаты которые выходят за границы изображения переходят на противоположную сторону
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


cv::Mat NewSobol(cv::Mat& img)
{
	cv::Mat mat_start(img.size(), img.type(), Scalar(0));

	float GX1[3][3] = { { 1, 2, 1 },{ 0, 0, 0 },{ -1, -2, -1 } };
	float GY1[3][3] = { { 1, 0, -1 },{ 2, 0, -2 },{ 1, 0, -1 } };
	float GX[3][3] = { { -3, -10, -3 },{ 0, 0, 0 },{ 3, 10, 3 } };
	float GY[3][3] = { { -3, 0, 3 },{ -10, 0, 10 },{ -3, 0, 3 } };

	for (int x = 1; x < img.cols - 1; x++)
	{
		for (int y = 1; y < img.rows - 1; y++)
		{
			Point curr_point(x, y);
			float dx = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y))) *  GX1[3][3];
			float dy = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x, y + 1)))* GY1[3][3];
			float sobol = (sqrt(pow(dx, 2) + pow(dy, 2))) - 250;

			mat_start.at<uint8_t>(curr_point) = static_cast<uint8_t>(sobol - 6) * 10;
		}
	}
	return mat_start;
}

cv::Mat NewShar(cv::Mat& img)
{
	cv::Mat mat_start(img.size(), img.type(), Scalar(0));
	float GX[3][3] = { { -3, -10, -3 },{ 0, 0, 0 },{ 3, 10, 3 } };
	float GY[3][3] = { { -3, 0, 3 },{ -10, 0, 10 },{ -3, 0, 3 } };
	for (int x = 1; x < img.cols - 1; x++)
	{
		for (int y = 1; y < img.rows - 1; y++)
		{
			Point curr_point(x, y);
			float dx = ((img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y))) *  GX[3][3]) / 2;
			float dy = ((img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x, y + 1)))* GY[3][3]) / 2;
			float sobol = (sqrt(pow(dx, 2) + pow(dy, 2))) - 250;
			mat_start.at<uint8_t>(curr_point) = static_cast<uint8_t>(sobol) * 10;
			if (120 < mat_start.at<uint8_t>(curr_point) && mat_start.at<uint8_t>(curr_point) < 255)
			{
				mat_start.at<uint8_t>(curr_point) = 255;
			}
			else
			{
				mat_start.at<uint8_t>(curr_point) = 0;
			}
		}
	}
	return mat_start;
}

cv::Mat NewGradientPrevitta(cv::Mat& img)
{
	cv::Mat mat_mod(img.size(), img.type(), Scalar(0));

	float E[3][3] =
	{
	{ -3, -3, 5},
	{ -3, 5, 5},
	{ -3, -3, 5} 
	};
	float NE[3][3] = 
	{
	{ -3, 5, 5},
	{ -3, 0, 5},
	{ -3, -3, -3} 
	};
	float N[3][3] = 
	{
	{ 5, 5, 5},
	{ -3, 0, -3},
	{ -3, -3, -3} 
	};
	float NW[3][3] = 
	{
	{ 5, 5, -3},
	{ 5, 0, -3},
	{ -3, -3, -3} 
	};
	float W[3][3] = 
	{
	{ 5, -3, -3},
	{ 5, 0, -3},
	{ 5, -3, -3} 
	};
	float SW[3][3] = 
	{
	{ -3, -3, -3},
	{ 5, 0, -3},
	{ 5, 5, -3} 
	};
	float S[3][3] = 
	{
	{ -3, -3, -3},
	{ -3, 0, -3},
	{ 5, 5, 5} 
	};
	float SE[3][3] = 
	{
	{ -3, -3, 5 },
	{ -3, 0, 5 },
	{ -3, 5, 5 } 
	};

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
			mat_mod.at<uint8_t>(curr_point) += 255 * log10f((img.at<uint8_t>(curr_point)) + 1);
			mat_mod.at<uint8_t>(curr_point) = static_cast<uint8_t>(priv);

			if (0 <= mat_mod.at<uint8_t>(curr_point) && mat_mod.at<uint8_t>(curr_point) < 50)
			{
				mat_mod.at<uint8_t>(curr_point) = 0;
			}
			if (50 <= mat_mod.at<uint8_t>(curr_point) && mat_mod.at<uint8_t>(curr_point) < 255)
			{
				mat_mod.at<uint8_t>(curr_point) = 255;
			}
			if (mat_mod.at<uint8_t>(curr_point) == 255 &&
				mat_mod.at<uint8_t>(Point(x - 1, y)) == 255 &&
				mat_mod.at<uint8_t>(Point(x, y + 1)) == 255 &&
				mat_mod.at<uint8_t>(Point(x - 1, y)) == 0 &&
				mat_mod.at<uint8_t>(x, y - 1) == 0)
			{
				mat_mod.at<uint8_t>(curr_point) = 0;
			}
		}
	}
	return  mat_mod;
}

cv::Mat NewFilter(cv::Mat& img) //used filter Laplassian and Gaussian
{
	cv::Mat mat_start(img.size(), img.type(), Scalar(0));

	float Gx[3][3] =
	{
		{ 0, -1, 0},
		{ -1, 4, -1},
		{ 0, -1, 0}
	};
	float Gy[3][3] =
	{
		{ -1, -1, -1},
		{ -1, 8, -1},
		{ -1, -1, -1}
	};

	for (int x = 1; x < img.cols - 1; x++)
	{
		for (int y = 1; y < img.rows - 1; y++)
		{
			Point curr_point(x, y);
			float ddx = (img.at<uint8_t>(Point(x + 1, y)) - 2 * (img.at<uint8_t>(Point(x, y))) +
				img.at<uint8_t>(Point(x - 1, y)));

			float ddy = (img.at<uint8_t>(Point(x, y + 1)) - 2 * (img.at<uint8_t>(Point(x, y))) +
				img.at<uint8_t>(Point(x, y - 1)));

			float lpCas = sqrt(pow(ddx * Gx[3][3], 2) + pow(ddy * Gy[3][3], 2));
			mat_start.at<uint8_t>(curr_point) = static_cast<uint8_t>(lpCas);
		}
	}
	return mat_start;
}

cv::Mat calcLoGDiskretWeights(cv::Mat& img)
{
	cv::Mat sub_mat(img.size(), img.type(), Scalar(0));
	for (int x = 1; x < img.cols - 1; x++)
	{
		for (int y = 1; y < img.rows - 1; y++)
		{
			Point curr_point(x, y);

			sub_mat.at<uint8_t>(curr_point) =
				(abs(img.at<uint8_t>(Point(x + 1, y)) +
					img.at<uint8_t>(Point(x - 1, y)) +
					img.at<uint8_t>(Point(x, y + 1)) +
					img.at<uint8_t>(Point(x, y - 1)) +
					img.at<uint8_t>(Point(x + 1, y + 1)) +
					img.at<uint8_t>(Point(x + 1, y - 1)) +
					img.at<uint8_t>(Point(x - 1, y + 1)) +
					img.at<uint8_t>(Point(x - 1, y - 1)) -
					8 * img.at<uint8_t>(curr_point))) / 3;

		}
	}
	return sub_mat;
}