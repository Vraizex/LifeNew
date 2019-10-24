
#include "Pix.h"

#include <random>
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include <vector>

float sgn(float x) // ��������
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
float soft_shrink(float d, float T) // ������ ����������
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
float hard_shrink(float d, float T) // ������� ����������
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
float Garrot_shrink(float d, float T) // ���������� ������������� �� �������
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
cv::Mat calcVGradient(cv::Mat& img)
{
	cv::Mat sub_mat(img.size(), CV_8UC1, Scalar(0));
	for (int x = 1; x < img.cols - 1; x++)
	{
		for (int y = 1; y < img.rows - 1; y++)
		{
			Point curr_point(x, y);
			sub_mat.at<uint8_t>(curr_point) = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x , y + 1)) + 255) / 2;
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

cv::Mat cvHaarWavelet(cv::Mat &img, cv::Mat &dst, int NIter) // �������-��������������
{
	
	float c,b, dh, dv, dd;
	int width = img.cols;
	int height = img.rows;
	for (int k = 0; k < NIter; k++)
	{
		for (int y = 0; y < (width >> (k + 1)); y++)
		{
			for (int x = 0; x < (width >> (k + 1)); x++)
			{
				c = (img.at<uint8_t>( y, x) + img.at<uint8_t>( y,  x + 1) + img.at<uint8_t>( y + 1,  x) + img.at<uint8_t>( y + 1,  x + 1));
				dst.at<uint8_t>(y, x) = c;
				dst.at<uint8_t>(y + (height >> (k + 1)), x) = c;
							
				dh = (img.at<uint8_t>( y,  x) + img.at<uint8_t>( y + 1,  x) - img.at<uint8_t>( y,  x + 1) - img.at<uint8_t>( y + 1, 2 * x + 1));
				dst.at<uint8_t>(y, x + (width >> (k + 1))) = dh;
				
				dv = (img.at<uint8_t>(2 * y, 2 * x) + img.at<uint8_t>(2 * y, 2 * x + 1) - img.at<uint8_t>(2 * y + 1, 2 * x) - img.at<uint8_t>(2 * y + 1, 2 * x + 1));
				dst.at<uint8_t>(y + (height >> (k + 1)), x) = dv;

				dd = (img.at<uint8_t>(2 * y, 2 * x) - img.at<uint8_t>(2 * y, 2 * x + 1) - img.at<uint8_t>(2 * y + 1, 2 * x) + img.at<uint8_t>(2 * y + 1, 2 * x + 1));
				dst.at<uint8_t>(y + (height >> (k + 1)), x + (width >> (k + 1))) = dd;
			
			}
		}
		dst.copyTo(img);

		return dst;
	}
}

cv::Mat cvInvHaarWavelet(cv::Mat &img, cv::Mat &dst, int NIter, int SHRINKAGE_TYPE, float SHRINKAGE_T )
{
	float c, dh, dv, dd;

	int width = img.cols;
	int height = img.rows;

	// NIter - ���������� �������� ��������������

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

				// ��������� ������������ (shrinkage)
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
		Mat C = img(Rect(0, 0, width >> (k - 1), height >> (k - 1)));
		Mat D = dst(Rect(0, 0, width >> (k - 1), height >> (k - 1)));
		D.copyTo(C);
	}
	return img;
}

cv::Mat calcSOBOL(cv::Mat& img)
{
	cv::Mat mat_mod(img.size(), CV_8UC1, Scalar(0));


	int cols = img.cols;
	int rows = img.rows;

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


			Point curr_point(x, y);

			float dx = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y)))/2;
			float dy = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x, y + 1)))/2;

			float sobol = (sqrt(pow(dx,2) + pow(dy,2))) - 250;


			mat_mod.at<uint8_t>(curr_point) = static_cast<uint8_t>(sobol - 6) * 10;

		}
	}

	return  mat_mod;
}

cv::Mat calcPrevitta(cv::Mat& img)
{
	cv::Mat mat_mod(img.size(), CV_8UC1, Scalar(0));


	int cols = img.cols;
	int rows = img.rows;



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


			Point curr_point(x, y);

			float dx = (img.at<uint8_t>(Point(x + 1, y)) - img.at<uint8_t>(Point(x - 1, y))) / 2;
			float dy = (img.at<uint8_t>(Point(x, y + 1)) - img.at<uint8_t>(Point(x, y - 1))) / 2;


			float sobol = sqrt(pow(dx,2) + pow(dy,2)) - 250;


			mat_mod.at<uint8_t>(curr_point) = static_cast<uint8_t>(sobol - 6) * 4 ;

		}
	}

	return  mat_mod;
}

cv::Mat calcRobertsa(cv::Mat& img)
{
	cv::Mat sub_mat(img.size(), CV_8UC1, Scalar(0));
	for (int x = 1; x < img.cols - 1; x++)
	{
		for (int y = 1; y < img.rows - 1; y++)
		{
			Point curr_point(x, y);
			float a = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y + 1))) / 2;
			float b = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x, y + 1))) / 2;

			sub_mat.at<uint8_t>(curr_point) = (sqrt(pow(a,2) + pow(b,2)) - 256)*10;
			

		}
	}
	return sub_mat;
}

//log(1+arg)

cv::Mat calcLoG(cv::Mat& img)
{
	int matrix[3][3] = { {-1,-1,-1}, {-1,8,-1}, {-1,-1,-1} };

	cv::Mat sub_mat(img.size(), CV_8UC1, Scalar(0));
	for (int x = 1; x < img.cols - 1; x++)
	{
		for (int y = 1; y < img.rows - 1; y++)
		{
			Point curr_point(x, y);

			sub_mat.at<uint8_t>(curr_point) = ((img.at<uint8_t>(Point(x + 1, y)) +
				img.at<uint8_t>(Point(x - 1, y)) -
				4 * img.at<uint8_t>(Point(x, y)) +
				img.at<uint8_t>(Point(x , y + 1)) + 
				img.at<uint8_t>(Point(x, y - 1)))/2);

			/*float ddx = (img.at<uint8_t>(Point(x + 1, y)) - 2 * (img.at<uint8_t>(Point(x, y))) +
				img.at<uint8_t>(Point(x - 1, y))) / 2;

			float ddy = (img.at<uint8_t>(Point(x, y + 1)) - 2 * (img.at<uint8_t>(Point(x, y))) +
				img.at<uint8_t>(Point(x, y - 1))) / 2;
			float lpCas = abs(img.at<uint8_t>(Point(x + 1, y)) + img.at<uint8_t>(Point(x - 1, y)) +
				img.at<uint8_t>(Point(x, y + 1)) + img.at<uint8_t>(Point(x, y - 1)) - 4 * img.at<uint8_t>(Point(x, y)));*/

			//sub_mat.at<uint8_t>(curr_point) = static_cast<uint8_t>((sqrt(pow(ddx , 2) + pow(ddy, 2))));
			/*sub_mat.at<uint8_t>(curr_point) = static_cast<uint8_t>(lpCas - 256 )*10;*/

			if(120<= sub_mat.at<uint8_t>(curr_point) &&
				sub_mat.at<uint8_t>(curr_point) <= 255)
			{
				sub_mat.at<uint8_t>(curr_point) = 255;
			}
			else
			{
				sub_mat.at<uint8_t>(curr_point) = 0;
			}

		}
	}
	return sub_mat;
}

cv::Mat calcLoGDiskret(cv::Mat& img)
{
	cv::Mat sub_mat(img.size(), CV_8UC1, Scalar(0));
	for (int x = 1; x < img.cols - 1; x++)
	{
		for (int y = 1; y < img.rows - 1; y++)
		{
			Point curr_point(x, y);

			 float b = ((((img.at<uint8_t>(Point(x + 1, y + 1)) - img.at<uint8_t>(curr_point)) /sqrt(2)) - 
				((img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point( x - 1, y - 1)) / sqrt(2))))/sqrt(2)) +
				((((img.at<uint8_t>(Point(x + 1, y - 1)) - img.at<uint8_t>(curr_point)) / sqrt(2)) -
				((img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x - 1, y + 1)) / sqrt(2)))) / sqrt(2));

			 sub_mat.at<uint8_t>(curr_point) = b / 2 - 250;

			/* if (248< sub_mat.at<uint8_t>(curr_point)&& sub_mat.at<uint8_t>(curr_point) < 256 )
			 
				 sub_mat.at<uint8_t>(curr_point) = 250 ;
			 else if (0 < sub_mat.at<uint8_t>(curr_point) && sub_mat.at<uint8_t>(curr_point) < 30)
				
				 sub_mat.at<uint8_t>(curr_point) = 15;
			 else if (30 < sub_mat.at<uint8_t>(curr_point) && sub_mat.at<uint8_t>(curr_point) < 60)
				 
					 sub_mat.at<uint8_t>(curr_point) = 45;
			 else if (60 < sub_mat.at<uint8_t>(curr_point) && sub_mat.at<uint8_t>(curr_point) < 120)
					 
						 sub_mat.at<uint8_t>(curr_point) = 90;
			 else if (120 < sub_mat.at<uint8_t>(curr_point) && sub_mat.at<uint8_t>(curr_point) < 150)
						 
							 sub_mat.at<uint8_t>(curr_point) = 135;
			 else if (150 < sub_mat.at<uint8_t>(curr_point) && sub_mat.at<uint8_t>(curr_point) < 190)
							 
								 sub_mat.at<uint8_t>(curr_point) = 170;
			 else if (190 < sub_mat.at<uint8_t>(curr_point) && sub_mat.at<uint8_t>(curr_point) < 210)
								 
									 sub_mat.at<uint8_t>(curr_point) = 200;
			 else if (210 < sub_mat.at<uint8_t>(curr_point) && sub_mat.at<uint8_t>(curr_point) < 230)
									 
										 sub_mat.at<uint8_t>(curr_point) = 220;
			 else if (231 < sub_mat.at<uint8_t>(curr_point) && sub_mat.at<uint8_t>(curr_point) < 248)

				 sub_mat.at<uint8_t>(curr_point) = 240;
								 */
			 
		}
	}
	return sub_mat;
}

cv::Mat calcLoGDiskretWeights(cv::Mat& img)
{
	cv::Mat sub_mat(img.size(), CV_8UC1, Scalar(0));
	for (int x = 1; x < img.cols - 1; x++)
	{
		for (int y = 1; y < img.rows - 1; y++)
		{
			Point curr_point(x, y);

			float b =(abs(img.at<uint8_t>(Point(x + 1, y ))+ 
				img.at<uint8_t>(Point(x - 1, y)) + 
				img.at<uint8_t>(Point(x, y + 1)) + 
				img.at<uint8_t>(Point(x, y - 1)) +
				img.at<uint8_t>(Point(x + 1, y + 1)) + 
				img.at<uint8_t>(Point(x + 1, y - 1)) + 
				img.at<uint8_t>(Point(x - 1, y + 1)) + 
				img.at<uint8_t>(Point(x - 1, y - 1)) - 
				8*img.at<uint8_t>(curr_point)))/3;

			sub_mat.at<uint8_t>(curr_point) = b ;
		}
	}
	return sub_mat;
}

cv::Mat calcLoGDiskretWeightsProg(cv::Mat& img)
{
	cv::Mat sub_mat(img.size(), CV_8UC1, Scalar(0));
	for (int x = 1; x < img.cols - 1; x++)
	{
		for (int y = 1; y < img.rows - 1; y++)
		{
			Point curr_point(x, y);

			float b = (abs(img.at<uint8_t>(Point(x + 1, y + 1)) +
				img.at<uint8_t>(Point(x + 1, y - 1)) +
				img.at<uint8_t>(Point(x - 1, y + 1)) +
				img.at<uint8_t>(Point(x - 1, y - 1)) -
				4 * img.at<uint8_t>(curr_point))) / 2;

			sub_mat.at<uint8_t>(curr_point) = b;
		}
	}
	return sub_mat;
}

//Mat GradientSOBOLBinAndMatrix(Mat& img)
//{
//
//	Mat mat_mod(img.size(), CV_8UC1, Scalar(0));
//	Mat matrix_X(img.size(), CV_8UC1, Scalar(0));
//	Mat matrix_Y(img.size(), CV_8UC1, Scalar(0));
//
//	int cols = img.cols;
//	int rows = img.rows;
//	for (int x = 1; x < cols - 1; x++)
//	{
//		for (int y = 1; y < rows - 1; y++)
//		{
//			vector<uint8_t> arr;
//			for (int i = -1; i <= 1; i++)
//			{
//				for (int j = -1; j <= 1; j++)
//				{
//					auto cur = img.at<uint8_t>(Point(j + x, i + y));
//					arr.push_back(cur);
//				}
//			}
//
//			Point curr_point(x, y);
//
//			float dx = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y))) / 2;
//			float dy = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x, y + 1))) / 2;
//			float sobol = (sqrt(pow(dx, 2) + pow(dy, 2)) - 250);
//			mat_mod.at<uint8_t>(curr_point) = static_cast<uint8_t>(sobol)*10;
//	
//
//			if (85 <= mat_mod.at<uint8_t>(curr_point) && mat_mod.at<uint8_t>(curr_point) <= 255) // Binary
//			{
//				mat_mod.at<uint8_t>(curr_point) = 255;
//			}
//			else
//			{
//				mat_mod.at<uint8_t>(curr_point) = 0;
//			}
//			
//			matrix_X.at<uint8_t>(curr_point) = mat_mod.at<uint8_t>(curr_point);
//			
//			if ((arr[0] && arr[8]) == 255 && (arr[2] && arr[6]) == 255)
//
//			{
//				arr[4] = 255;
//
//			}
//
//			else
//				arr[4] = 0;
//			
//			matrix_X.at<uint8_t>(curr_point) = arr[4];
//			
//			
//			
//			matrix_Y.at<uint8_t>(curr_point) = mat_mod.at<uint8_t>(curr_point);
//
//			if ((arr[1] && arr[7]) == 255 && (arr[3] && arr[5]) == 255)
//
//			{
//				arr[4] = 255;
//
//			}
//
//			else
//				arr[4] = 0;
//
//
//			matrix_Y.at<uint8_t>(curr_point) = arr[4];
//			
//			mat_mod.at<uint8_t>(curr_point) = matrix_Y.at<uint8_t>(curr_point) + matrix_X.at<uint8_t>(curr_point);
//		}
//
//	}
//	
//	
//	return  mat_mod;
//	   
//}

//cv::Mat calc3x3GradientSOBOLBinAndMatrix(cv::Mat& img)
//{
//	cv::Mat mat_mod(img.size(), CV_8UC1, Scalar(0));
//	cv::Mat matrix_X(img.size(), CV_8UC1, Scalar(0));
//	cv::Mat matrix_Y(img.size(), CV_8UC1, Scalar(0));
//
//	int cols = img.cols;
//	int rows = img.rows;
//
//	
//
//	for (int x = 1; x < cols - 1; x++)
//	{
//		for (int y = 1; y < rows - 1; y++)
//		{
//			vector<uint8_t> arr;
//			for (int i = -1; i <= 1; i++)
//			{
//				for (int j = -1; j <= 1; j++)
//				{
//					auto cur = img.at<uint8_t>(Point(j + x, i + y));
//					arr.push_back(cur);
//				}
//			}
//
//
//			Point curr_point(x, y);
//
//			float dx = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y))) / 2;
//			float dy = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x, y + 1))) / 2;
//
//			float sobol = (sqrt(pow(dx, 2) + pow(dy, 2)) - 250);
//
//
//			mat_mod.at<uint8_t>(curr_point) = static_cast<uint8_t>(sobol - 6) * 10;
//
//
//
//			if (60 <= mat_mod.at<uint8_t>(curr_point) && mat_mod.at<uint8_t>(curr_point) <= 255) // Binary
//
//			{
//				mat_mod.at<uint8_t>(curr_point) = 0;
//			}
//			else
//			{
//				mat_mod.at<uint8_t>(curr_point) = 255;
//			}
//
//			//matrix_X.at<uint8_t>(curr_point) = mat_mod.at<uint8_t>(curr_point);
//
//			//	if ((matrix_X.at<uint8_t>(Point(x - 1, y - 1)) == 1) && (matrix_X.at<uint8_t>(Point(x + 1, y + 1)) == 1) && (matrix_X.at<uint8_t>(Point(x + 1, y - 1)) == 1) && (matrix_X.at<uint8_t>(Point(x + 1, y + 1)) == 1)) // X
//			//	{
//			//		matrix_X.at<uint8_t>(curr_point) = 255;
//			//	}
//			//	else
//			//	{
//			//		matrix_X.at<uint8_t>(curr_point) = 0;
//			//	}
//
//
//			//matrix_Y.at<uint8_t>(curr_point) = mat_mod.at<uint8_t>(curr_point);
//
//			//	if ((matrix_Y.at<uint8_t>(Point(x, y - 1)) == 1) && (matrix_Y.at<uint8_t>(Point(x, y + 1)) == 1) && (matrix_Y.at<uint8_t>(Point(x - 1, y)) == 1) && (matrix_Y.at<uint8_t>(Point(x + 1, y)) == 1)) // +
//			//	{
//			//		matrix_Y.at<uint8_t>(curr_point) = 255;
//			//	}
//			//	else
//			//	{
//			//		matrix_Y.at<uint8_t>(curr_point) = 0;
//			//	}
//			//
//			//mat_mod.at<uint8_t>(curr_point) = matrix_X.at<uint8_t>(curr_point) + matrix_Y.at<uint8_t>(curr_point);
//
//
//
//
//			vector<uint8_t> arr1;
//			for (int i = -1; i <= 1; i++)
//			{
//				for (int j = -1; j <= 1; j++)
//				{
//					auto cur = img.at<uint8_t>(Point(j + x, i + y));
//					arr1.push_back(cur);
//				}
//			}
//
//			if ((arr1[0]==1 && arr1[8]==1) && (arr1[2] == 1 && arr1[6] == 1) )
//			{
//				arr1[4] = 255;
//			}
//			else
//			{
//				arr1[4] = 0;
//			}
//		/*	matrix_X.at<uint8_t>(curr_point) = mat_mod.at<uint8_t>(curr_point);*/
//
//
//		//	vector<uint8_t> arr2;
//		//	for (int i = -1; i <= 1; i++)
//		//	{
//		//		for (int j = -1; j <= 1; j++)
//		//		{
//		//			auto cur = img.at<uint8_t>(Point(j + x, i + y));
//		//			arr2.push_back(cur);
//		//		}
//		//	}
//		//	if ((arr2[1] == 1 && arr2[7] == 1) && (arr2[3] == 1 && arr2[5] == 1))
//		//	{
//		//		arr2[4] = 255;
//		//	}
//		//	else
//		//	{
//		//		arr2[4] = 0;
//		//	}
//		//	
//		//	matrix_Y.at<uint8_t>(curr_point) = mat_mod.at<uint8_t>(curr_point);
//
//		//	mat_mod.at<uint8_t>(curr_point) = matrix_Y.at<uint8_t>(curr_point) + matrix_X.at<uint8_t>(curr_point);
//		//}
//		}
//	}
//
//	return  mat_mod;
//}

cv::Mat calc3x3GradientSOBOLBinAndMatrix(cv::Mat& img)
{
	cv::Mat mat_mod(img.size(), CV_8UC1, Scalar(0));


	int cols = img.cols;
	int rows = img.rows;



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


			Point curr_point(x, y);

			float dx = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y))) / 2;
			float dy = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x, y + 1))) / 2;

			float sobol = (sqrt(pow(dx, 2) + pow(dy, 2))) - 250;


			mat_mod.at<uint8_t>(curr_point) = static_cast<uint8_t>(sobol - 6) * 10;



			if (90 <= mat_mod.at<uint8_t>(curr_point) && 
				mat_mod.at<uint8_t>(curr_point) <= 255)
			{
				mat_mod.at<uint8_t>(curr_point) = 255;
			}
			else
			{
				mat_mod.at<uint8_t>(curr_point) = 1;
			}


		/*	if ((mat_mod.at<uint8_t>(Point(x - 1, y - 1)) == 255) && 
				(mat_mod.at<uint8_t>(Point(x + 1, y + 1)) == 255) && 
				(mat_mod.at<uint8_t>(Point(x + 1, y - 1)) == 255) && 
				(mat_mod.at<uint8_t>(Point(x + 1, y + 1)) == 255) ||
				(mat_mod.at<uint8_t>(Point(x, y - 1)) == 255) && 
				(mat_mod.at<uint8_t>(Point(x, y + 1)) == 255) && 
				(mat_mod.at<uint8_t>(Point(x - 1, y)) == 255) && 
				(mat_mod.at<uint8_t>(Point(x + 1, y)) == 255))
			{
				mat_mod.at<uint8_t>(curr_point) = 255;
			}
			else
			{
				mat_mod.at<uint8_t>(curr_point) = 0;
			}*/



			/*if ((mat_mod.at<uint8_t>(Point(x, y - 1)) == 255) && (mat_mod.at<uint8_t>(Point(x, y + 1)) == 255) && (mat_mod.at<uint8_t>(Point(x - 1, y)) == 255) && (mat_mod.at<uint8_t>(Point(x + 1, y)) == 255))
			{
				mat_mod.at<uint8_t>(curr_point) = 255;
			}
			else
			{
				mat_mod.at<uint8_t>(curr_point) = 0;
			}
*/

		}
	}

	return  mat_mod;
}
//cv::Mat AB(cv::Mat& img)
//{
//	cv::Mat mat_mod(img.size(), CV_8UC1, Scalar(0));
//	cv::Mat start_mod(img.size(), CV_8UC1, Scalar(0));
//	cv::Mat mat_X(img.size(), CV_8UC1, Scalar(0));
//	cv::Mat mat_Y(img.size(), CV_8UC1, Scalar(0));
//	int cols = img.cols;
//	int rows = img.rows;
//
//
//
//	for (int x = 1; x < cols - 1; x++)
//	{
//		for (int y = 1; y < rows - 1; y++)
//		{
//			vector<uint8_t> arr;
//			for (int i = -1; i <= 1; i++)
//			{
//				for (int j = -1; j <= 1; j++)
//				{
//					auto cur = img.at<uint8_t>(Point(j + x, i + y));
//					arr.push_back(cur);
//				}
//			}
//
//
//			Point curr_point(x, y);
//			float dx = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y))) / 2;
//			float dy = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x, y + 1))) / 2;
//
//			float sobol = (sqrt(pow(dx, 2) + pow(dy, 2))) - 250;
//
//
//			start_mod.at<uint8_t>(curr_point) = static_cast<uint8_t>(sobol - 6) * 10;
//			
//
//			if (40 < start_mod.at<uint8_t>(curr_point) && start_mod.at<uint8_t>(curr_point) < 255)
//
//			{
//				start_mod.at<uint8_t>(curr_point) = 1;
//			}
//			else
//			{
//				start_mod.at<uint8_t>(curr_point) = 0;
//			}
//			
//
//			mat_X.at<uint8_t>(curr_point) = start_mod.at<uint8_t>(curr_point);
//
//			if ((arr[0] && arr[8] && arr[2] && arr[6]) == 1)
//			{
//				mat_X.at<uint8_t>(curr_point) = 1;
//			}
//
//			else
//			{
//				mat_X.at<uint8_t>(curr_point) = 0;
//			}
//
//			mat_Y.at<uint8_t>(curr_point) = start_mod.at<uint8_t>(curr_point);
//
//			if ((arr[1] && arr[3] && arr[5] && arr[7]) == 1)
//			{
//				mat_Y.at<uint8_t>(curr_point) = 1;
//			}
//			else
//			{
//				mat_Y.at<uint8_t>(curr_point) = 0;
//			}
//
//			mat_mod.at<uint8_t>(curr_point) = mat_X.at<uint8_t>(curr_point) + mat_Y.at<uint8_t>(curr_point)*100;
//
//		}
//	}
//
//	return { mat_mod };
//}
cv::Mat dilateY(cv::Mat& img) {


	cv::Mat mat_start(img.size(), CV_8UC1, Scalar(0));

	int cols = img.cols;
	int rows = img.rows;


	for (int x = 1; x < cols - 1; x++)
	{
		for (int y = 1; y < rows - 1; y++)
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

			if (mat_start.at<uint8_t>(curr_point) == 255)
			{

				if (x > 0 && 
					mat_start.at<uint8_t>(Point(x - 1, y)) == 0)
					mat_start.at<uint8_t>(Point(x - 1, y)) = 2;

				if (y > 0 &&
					mat_start.at<uint8_t>(Point(x, y - 1)) == 0)
					mat_start.at<uint8_t>(Point(x, y - 1)) = 2;

				if (x + 1 < mat_start.cols && 
					mat_start.at<uint8_t>(Point(x + 1, y)) == 0)
					mat_start.at<uint8_t>(Point(x + 1, y)) = 2;

				if (y + 1 < mat_start.rows && 
					mat_start.at<uint8_t>(Point(x, y + 1)) == 0)
					mat_start.at<uint8_t>(Point(x, y + 1)) = 2;
			}
		}
	}
	for (int x = 0; x < cols; x++)
	{
		for (int y = 0; y < rows; y++)
		{
			if (mat_start.at<uint8_t>(Point(x, y)) == 2)
			{
				mat_start.at<uint8_t>(Point(x, y)) = 255;
			}
		}
	}
	return mat_start;
}

cv::Mat dilateX(cv::Mat& img) {


	cv::Mat mat_start(img.size(), CV_8UC1, Scalar(0));

	int cols = img.cols;
	int rows = img.rows;


	for (int x = 1; x < cols - 1; x++)
	{
		for (int y = 1; y < rows - 1; y++)
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



			if (mat_start.at<uint8_t>(curr_point) == 255)
			{

				if (x > 0 && mat_start.at<uint8_t>(Point(x - 1, y - 1)) == 0)
					mat_start.at<uint8_t>(Point(x - 1, y - 1)) = 2;

				if (y > 0 && mat_start.at<uint8_t>(Point(x + 1, y - 1)) == 0)
					mat_start.at<uint8_t>(Point(x + 1, y - 1)) = 2;

				if (x + 1 < mat_start.cols && mat_start.at<uint8_t>(Point(x + 1, y + 1)) == 0)
					mat_start.at<uint8_t>(Point(x + 1, y + 1)) = 2;

				if (y + 1 < mat_start.rows && mat_start.at<uint8_t>(Point(x - 1, y + 1)) == 0)
					mat_start.at<uint8_t>(Point(x - 1, y + 1)) = 2;
			}
		}
	}
	for (int x = 0; x < cols; x++)
	{
		for (int y = 0; y < rows; y++)
		{
			if (mat_start.at<uint8_t>(Point(x, y)) == 2)
			{
				mat_start.at<uint8_t>(Point(x, y)) = 255;
			}
		}
	}
	return mat_start;

}

cv::Mat dilateXY(cv::Mat& img) {
	cv::Mat mat_start(img.size(), CV_8UC1, Scalar(0));
	int cols = img.cols;
	int rows = img.rows;
	for (int x = 2; x < cols - 2; x++)
	{
		for (int y = 2; y < rows - 2; y++)
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

			if (mat_start.at<uint8_t>(curr_point) == 255)
			{

				if (x > 0 && mat_start.at<uint8_t>(Point(x - 1, y - 1)) == 0)
					mat_start.at<uint8_t>(Point(x - 1, y - 1)) = 2;

				if (y > 0 && mat_start.at<uint8_t>(Point(x + 1, y - 1)) == 0)
					mat_start.at<uint8_t>(Point(x + 1, y - 1)) = 2;

				if (x + 1 < mat_start.cols && mat_start.at<uint8_t>(Point(x + 1, y + 1)) == 0)
					mat_start.at<uint8_t>(Point(x + 1, y + 1)) = 2;

				if (y + 1 < mat_start.rows && mat_start.at<uint8_t>(Point(x - 1, y + 1)) == 0)
					mat_start.at<uint8_t>(Point(x - 1, y + 1)) = 2;

				if (x > 0 && mat_start.at<uint8_t>(Point(x - 1, y)) == 0)
					mat_start.at<uint8_t>(Point(x - 1, y)) = 2;

				if (y > 0 && mat_start.at<uint8_t>(Point(x, y - 1)) == 0)
					mat_start.at<uint8_t>(Point(x, y - 1)) = 2;

				if (x + 1 < mat_start.cols && mat_start.at<uint8_t>(Point(x + 1, y)) == 0)
					mat_start.at<uint8_t>(Point(x + 1, y)) = 2;

				if (y + 1 < mat_start.rows && mat_start.at<uint8_t>(Point(x, y + 1)) == 0)
					mat_start.at<uint8_t>(Point(x, y + 1)) = 2;

			}
		}
	}
	for (int x = 0; x < cols; x++)
	{
		for (int y = 0; y < rows; y++)
		{
			if (mat_start.at<uint8_t>(Point(x, y)) == 2)
			{
				mat_start.at<uint8_t>(Point(x, y)) = 255;
			}
		}
	}

	
	return mat_start;
}

cv::Mat dilate3X3Y(cv::Mat& img) {


	cv::Mat mat_start(img.size(), CV_8UC1, Scalar(0));

	int cols = img.cols;
	int rows = img.rows;


	for (int x = 2; x < cols - 2; x++)
	{
		for (int y = 2; y < rows - 2; y++)
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

			if (mat_start.at<uint8_t>(curr_point) == 255)
			{

				if (x > 0 &&
					mat_start.at<uint8_t>(Point(x - 2, y)) == 0)
					mat_start.at<uint8_t>(Point(x - 2, y)) = 2;

				if (y > 0 &&
					mat_start.at<uint8_t>(Point(x, y + 2)) == 0)
					mat_start.at<uint8_t>(Point(x, y + 2)) = 2;

				if (x + 1 < mat_start.cols &&
					mat_start.at<uint8_t>(Point(x + 2, y)) == 0)
					mat_start.at<uint8_t>(Point(x + 2, y)) = 2;

				if (y + 1 < mat_start.rows &&
					mat_start.at<uint8_t>(Point(x, y + 2)) == 0)
					mat_start.at<uint8_t>(Point(x, y - 1)) = 2;

				if (x > 0 && mat_start.at<uint8_t>(Point(x - 2, y - 2)) == 0)
					mat_start.at<uint8_t>(Point(x - 2, y - 2)) = 2;

				if (y > 0 && mat_start.at<uint8_t>(Point(x + 2, y - 2)) == 0)
					mat_start.at<uint8_t>(Point(x + 2, y - 2)) = 2;

				if (x + 1 < mat_start.cols && mat_start.at<uint8_t>(Point(x + 2, y + 2)) == 0)
					mat_start.at<uint8_t>(Point(x + 2, y + 2)) = 2;

				if (y + 1 < mat_start.rows && mat_start.at<uint8_t>(Point(x - 2, y + 2)) == 0)
					mat_start.at<uint8_t>(Point(x - 2, y + 2)) = 2;
			}
		}
	}
	for (int x = 0; x < cols; x++)
	{
		for (int y = 0; y < rows; y++)
		{
			if (mat_start.at<uint8_t>(Point(x, y)) == 2)
			{
				mat_start.at<uint8_t>(Point(x, y)) = 255;
			}
		}
	}
	return mat_start;

}

cv::Mat dilateGOA(cv::Mat& img, int k)
{
	cv::Mat mat_start(img.size(), CV_8UC1, Scalar(0));
	int cols = img.cols;
	int rows = img.rows;


	for (int x = 1; x < cols - 1; x++)
	{
		for (int y = 1; y < rows - 1; y++)
		{
			Point curr_point(x, y);
			float dx = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y))) / 2;
			float dy = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x, y + 1))) / 2;

			float sobol = (sqrt(pow(dx, 2) + pow(dy, 2))) - 250;

			mat_start.at<uint8_t>(curr_point) = static_cast<uint8_t>(sobol - 6) * 10;



			if (25 < mat_start.at<uint8_t>(curr_point) && mat_start.at<uint8_t>(curr_point) < 255)

			{
				mat_start.at<uint8_t>(curr_point) = 255;
			}
			else
			{
				mat_start.at<uint8_t>(curr_point) = 0;
			}

			if (mat_start.at<uint8_t>(curr_point) == 255)
			{
				for (int l = x - k; l <= x + k; l++)
				{
					/*int remainingk = k - abs(x - l);*/
					for (int m = y - k; m <= y + k; m++) 
					{
						
						if (l >= 0 && 
							m >= 0 && 
							l < mat_start.cols && 
							m < mat_start.rows && 
							mat_start.at<uint8_t>(Point(l,m)) == 0)

							mat_start.at<uint8_t>(Point(l,m)) = 2;
						
					}
				}
			}
		}
	}
	for (int x = 0; x < cols; x++)
	{
		for (int y = 0; y < rows; y++)
		{
			if (mat_start.at<uint8_t>(Point(x, y)) == 2)
			{
				mat_start.at<uint8_t>(Point(x, y)) = 255;
			}
		}
	}
	return mat_start;
}

cv::Mat dilateAndErozia(cv::Mat& img, int k, int z)
{
	cv::Mat mat_start(img.size(), CV_8UC1, Scalar(0));
	int cols = img.cols;
	int rows = img.rows;

	for (int x = 1; x < cols - 1; x++)
	{
		for (int y = 1; y < rows - 1; y++)
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
			

			if (mat_start.at<uint8_t>(curr_point) == 255)
			{
				for (int l = x - k; l <= x + k; l++)
				{
					/*int remainingk = k - abs(x - l);*/
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


	for (int x = 0; x < cols; x++)
	{
		for (int y = 0; y < rows; y++)
		{
			Point curr_point(x, y);
			
			if (mat_start.at<uint8_t>(Point(x, y)) == 2)
			{
				mat_start.at<uint8_t>(Point(x, y)) = 255;
			}

			if (mat_start.at<uint8_t>(curr_point) == 0)
			{
				for (int l = x - z; l <= x + z; l++)
				{

					for (int m = y - z; m <= y + z; m++)
					{

						if (l >= 0 &&
							m >= 0 &&
							l < mat_start.cols &&
							m < mat_start.rows &&
							mat_start.at<uint8_t>(Point(l, m)) == 255)

							mat_start.at<uint8_t>(Point(l, m)) = 2;

					}
				}
			}
		}
	}
	for (int x = 0; x < cols; x++)
	{
		for (int y = 0; y < rows; y++)
		{
			
			if (mat_start.at<uint8_t>(Point(x, y)) == 2)
			{
				mat_start.at<uint8_t>(Point(x, y)) = 0;
			}
		}
	}
	
	return mat_start;
	
}

cv::Mat dilateEroziaLevel(cv::Mat& img, int k)
{
	cv::Mat mat_start(img.size(), CV_8UC1, Scalar(0));
	int cols = img.cols;
	int rows = img.rows;

	
	for (int x = 1; x < cols - 1; x++)
	{
		for (int y = 1; y < rows - 1; y++)
		{
			Point curr_point(x, y);
			float dx = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y))) / 2;
			float dy = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x, y + 1))) / 2;

			float sobol = (sqrt(pow(dx, 2) + pow(dy, 2))) - 250;

			mat_start.at<uint8_t>(curr_point) = static_cast<uint8_t>(sobol - 6) * 10;


			if (25 < mat_start.at<uint8_t>(curr_point) && mat_start.at<uint8_t>(curr_point) < 255)

			{
				mat_start.at<uint8_t>(curr_point) = 255;
			}
			else
			{
				mat_start.at<uint8_t>(curr_point) = 0;
			}
			
			




			if (mat_start.at<uint8_t>(curr_point) == 0)
			{
				for (int l = x - k; l <= x + k; l++)
				{
					
					for (int m = y - k; m <= y + k; m++)
					{

						if (l >= 0 &&
							m >= 0 &&
							l < mat_start.cols &&
							m < mat_start.rows &&
							mat_start.at<uint8_t>(Point(l, m)) == 255)

							mat_start.at<uint8_t>(Point(l, m)) = 2;

					}
				}
			}
		}
	}
	for (int x = 0; x < cols; x++)
	{
		for (int y = 0; y < rows; y++)
		{
			if (mat_start.at<uint8_t>(Point(x, y)) == 2)
			{
				mat_start.at<uint8_t>(Point(x, y)) = 0;
			}
		}
	}
	return mat_start;
}

cv::Mat EroziaAndDilate(cv::Mat& img, int k, int z)
{
	cv::Mat mat_start(img.size(), CV_8UC1, Scalar(0));
	int cols = img.cols;
	int rows = img.rows;


	for (int x = 1; x < cols - 1; x++)
	{
		for (int y = 1; y < rows - 1; y++)
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
			

			if (mat_start.at<uint8_t>(curr_point) == 0)
			{
				for (int l = x - k; l <= x + k; l++)
				{
					/*int remainingk = k - abs(x - l);*/
					for (int m = y - k; m <= y + k; m++)
					{
						if (l >= 0 &&
							m >= 0 &&
							l < mat_start.cols &&
							m < mat_start.rows &&
							mat_start.at<uint8_t>(Point(l, m)) == 255)
							mat_start.at<uint8_t>(Point(l, m)) = 2;
						
					}
				}

			}
		}
	}


	for (int x = 0; x < cols; x++)
	{
		for (int y = 0; y < rows; y++)
		{
			Point curr_point(x, y);


			if (mat_start.at<uint8_t>(Point(x, y)) == 2)
			{
				mat_start.at<uint8_t>(Point(x, y)) = 0;
			}

			if (mat_start.at<uint8_t>(curr_point) == 255)
			{
				for (int l = x - z; l <= x + z; l++)
				{

					for (int m = y - z; m <= y + z; m++)
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
	for (int x = 0; x < cols; x++)
	{
		for (int y = 0; y < rows; y++)
		{

			if (mat_start.at<uint8_t>(Point(x, y)) == 2)
			{
				mat_start.at<uint8_t>(Point(x, y)) = 255;
			}
		}
	}
	return mat_start;

}

cv::Mat Bin(cv::Mat& img)
{
	cv::Mat mat_start(img.size(), CV_8UC1, Scalar(0));
	int cols = img.cols;
	int rows = img.rows;

	double GX1[3][3] = { { 1, 2, 1 },{ 0, 0, 0 },{ -1, -2, -1 } };
	double GY1[3][3] = { { 1, 0, -1 },{ 2, 0, -2 },{ 1, 0, -1 } };

	double GX[3][3] = { { -3, -10, -3 },{ 0, 0, 0 },{ 3, 10, 3 } };
	double GY[3][3] = { { -3, 0, 3 },{ -10, 0, 10 },{ -3, 0, 3 } };


	for (int x = 1; x < cols - 1; x++)
	{
		for (int y = 1; y < rows - 1; y++)
		{

			Point curr_point(x, y);
			float dx = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y))) / 2;
			float dy = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x, y + 1))) / 2;

			float sobol = (sqrt(pow(dx, 2) + pow(dy, 2))) - 250;

			mat_start.at<uint8_t>(curr_point) = static_cast<uint8_t>(sobol - 6) * 10;

			if (20 < mat_start.at<uint8_t>(curr_point) && mat_start.at<uint8_t>(curr_point) < 255)

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

cv::Mat dilateMXN(cv::Mat& img, int k, int z)
{

	cv::Mat mat_start(img.size(), CV_8UC1, Scalar(0));
	int cols = img.cols;
	int rows = img.rows;

	for (int x = 1; x < cols - 1; x++)
	{
		for (int y = 1; y < rows - 1; y++)
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


			if (mat_start.at<uint8_t>(curr_point) == 255)
			{

				for (int l = abs(x - k); l <= abs(x + k); l++)
				{
					/*int remainingk = k - abs(x - l);*/
					for (int m = (y - k); m <= (y + k); m++)
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


	for (int x = 0; x < cols; x++)
	{
		for (int y = 0; y < rows; y++)
		{
			Point curr_point(x, y);

			if (mat_start.at<uint8_t>(Point(x, y)) == 2)
			{
				mat_start.at<uint8_t>(Point(x, y)) = 255;
			}

			if (mat_start.at<uint8_t>(curr_point) == 0)
			{
				for (int l = (x - z); l <= (x + z); l++)
				{

					for (int m = (y - z); m <= (y + z); m++)
					{

						if (l >= 0 &&
							m >= 0 &&
							l < mat_start.cols &&
							m < mat_start.rows &&
							mat_start.at<uint8_t>(Point(l, m)) == 255)

							mat_start.at<uint8_t>(Point(l, m)) = 2;

					}
				}
			}
		}
	}
	for (int x = 0; x < cols; x++)
	{
		for (int y = 0; y < rows; y++)
		{

			if (mat_start.at<uint8_t>(Point(x, y)) == 2)
			{
				mat_start.at<uint8_t>(Point(x, y)) = 0;
			}
		}
	}

	return mat_start;
}

cv::Mat NewSobol(cv::Mat& img)
{
	cv::Mat mat_start(img.size(), CV_8UC1, Scalar(0));
	int cols = img.cols;
	int rows = img.rows;

	double GX1[3][3] = { { 1, 2, 1 },{ 0, 0, 0 },{ -1, -2, -1 } };
	double GY1[3][3] = { { 1, 0, -1 },{ 2, 0, -2 },{ 1, 0, -1 } };

	double GX[3][3] = { { -3, -10, -3 },{ 0, 0, 0 },{ 3, 10, 3 } };
	double GY[3][3] = { { -3, 0, 3 },{ -10, 0, 10 },{ -3, 0, 3 } };


	for (int x = 1; x < cols - 1; x++)
	{
		for (int y = 1; y < rows - 1; y++)
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

	cv::Mat mat_start(img.size(), CV_8UC1, Scalar(0));
	int cols = img.cols;
	int rows = img.rows;

	double GX[3][3] = { { -3, -10, -3 },{ 0, 0, 0 },{ 3, 10, 3 } };
	double GY[3][3] = { { -3, 0, 3 },{ -10, 0, 10 },{ -3, 0, 3 } };


	for (int x = 1; x < cols - 1; x++)
	{
		for (int y = 1; y < rows - 1; y++)
		{
			Point curr_point(x, y);
			float dx = ((img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y))) *  GX[3][3] )/2;
			float dy = ((img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x, y + 1)))* GY[3][3])/2;
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
	cv::Mat mat_mod(img.size(), CV_8UC1, Scalar(0));


	int cols = img.cols;
	int rows = img.rows;

	double E[3][3] = {
	{ -3, -3, 5 },
	{ -3, 5, 5 },
	{ -3, -3, 5 } };
	double NE[3][3] = { 
	{ -3, 5, 5 },
	{ -3, 0, 5 },
	{ -3, -3, -3 } };
	double N[3][3] = { 
	{ 5, 5, 5 },
	{ -3, 0, -3 },
	{ -3, -3, -3 } };
	double NW[3][3] = { 
	{ 5, 5, -3 },
	{ 5, 0, -3 },
	{ -3, -3, -3 } };
	double W[3][3] = { 
	{ 5, -3, -3 },
	{ 5, 0, -3 },
	{ 5, -3, -3 } };
	double SW[3][3] = { 
	{ -3, -3, -3 },
	{ 5, 0, -3 },
	{ 5, 5, -3 } };
	double S[3][3] = { 
	{ -3, -3, -3 },
	{ -3, 0, -3 },
	{ 5, 5, 5 } };
	double SE[3][3] = { 
	{ -3, -3, 5 },
	{ -3, 0, 5 },
	{ -3, 5, 5 } };


	for (int x = 1; x < cols - 1; x++)
	{
		for (int y = 1; y < rows - 1; y++)
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

cv::Mat NewLoG(cv::Mat& img)
{
	double GX[3][3] = { { 0, -1, 0 },{ -1, 4, -1 },{ 0, -1, 0 } };
	double GY[3][3] = { { -1, -1, -1 },{ -1, 8, -1 },{ -1, -1, -1 } };

	cv::Mat sub_mat(img.size(), CV_8UC1, Scalar(0));
	for (int x = 1; x < img.cols - 1; x++)
	{
		for (int y = 1; y < img.rows - 1; y++)
		{
			Point curr_point(x, y);

			
			float ddx = img.at<uint8_t>(Point(x + 1, y)) - 2 * img.at<uint8_t>(Point(x, y)) + img.at<uint8_t>(Point(x - 1, y));
			float ddy = img.at<uint8_t>(Point(x, y + 1)) - 2 * img.at<uint8_t>(Point(x, y)) + img.at<uint8_t>(Point(x, y - 1));

				sub_mat.at<uint8_t>(curr_point) = static_cast<uint8_t>(ddx *  ddy );


		}
	}

	return sub_mat;
}

cv::Mat BinandDeleteOnlyPixels(cv::Mat& img)
{


	cv::Mat mat_start(img.size(), CV_8UC1, Scalar(0));
	int cols = img.cols;
	int rows = img.rows;

	double GX1[3][3] = { { 1, 2, 1 },{ 0, 0, 0 },{ -1, -2, -1 } };
	double GY1[3][3] = { { 1, 0, -1 },{ 2, 0, -2 },{ 1, 0, -1 } };

	double GX[3][3] = { { -3, -10, -3 },{ 0, 0, 0 },{ 3, 10, 3 } };
	double GY[3][3] = { { -3, 0, 3 },{ -10, 0, 10 },{ -3, 0, 3 } };


	for (int x = 1; x < cols - 1; x++)
	{
		for (int y = 1; y < rows - 1; y++)
		{
			Point curr_point(x, y);
			float dx = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y))) * GX1[3][3];
			float dy = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x, y + 1))) * GY1[3][3];
			 
			float sobol = (sqrt(pow(dx, 2) + pow(dy, 2)));

			mat_start.at<uint8_t>(curr_point) = static_cast<uint8_t>(sobol) * 10;


			if (25 < mat_start.at<uint8_t>(curr_point) && mat_start.at<uint8_t>(curr_point) < 255)

			{
				mat_start.at<uint8_t>(curr_point) = 255;
			}
			else
			{
				mat_start.at<uint8_t>(curr_point) = 0;
			}

			if (mat_start.at<uint8_t>(curr_point) == 255)
			{

				if (x > 0 &&
					mat_start.at<uint8_t>(Point(x - 1, y)) == 0)
					mat_start.at<uint8_t>(Point(x - 1, y)) = 2;

				if (y > 0 &&
					mat_start.at<uint8_t>(Point(x, y - 1)) == 0)
					mat_start.at<uint8_t>(Point(x, y - 1)) = 2;

				if (x + 1 < mat_start.cols &&
					mat_start.at<uint8_t>(Point(x + 1, y)) == 0)
					mat_start.at<uint8_t>(Point(x + 1, y)) = 2;

				if (y + 1 < mat_start.rows &&
					mat_start.at<uint8_t>(Point(x, y + 1)) == 0)
					mat_start.at<uint8_t>(Point(x, y + 1)) = 2;
			}
		}
	}
	for (int x = 0; x < cols; x++)
	{
		for (int y = 0; y < rows; y++)
		{
			if (mat_start.at<uint8_t>(Point(x, y)) == 2)
			{
				mat_start.at<uint8_t>(Point(x, y)) = 0;
			}
		}
	}
	return mat_start;
}

cv::Mat lagrange2(cv::Mat& img)
{

	cv::Mat mat_start(img.size(), CV_8UC1, Scalar(0));
	int cols = img.cols;
	int rows = img.rows;

	for (int x = 2; x < cols - 2; x++)
	{
		for (int y = 2; y < rows - 2; y++)
		{
			Point curr_point(x, y);

			double h = img.at<uint8_t>(Point(x + 1, y)) - (img.at<uint8_t>(curr_point));
			float dx = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y))) / 2;
			float dy = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x, y + 1))) / 2;

			float sobol = (sqrt(pow(dx, 2) + pow(dy, 2))) - 250;

			mat_start.at<uint8_t>(curr_point) = static_cast<uint8_t>(sobol - 6) * 10;

			if (50 < mat_start.at<uint8_t>(curr_point) && mat_start.at<uint8_t>(curr_point) < 255)

			{
				mat_start.at<uint8_t>(curr_point) = 255;
			}

			else

			{
				mat_start.at<uint8_t>(curr_point) = 0;
			}
			
 
			
			float t = 3.8;

				mat_start.at<uint8_t>(curr_point) += ((
				(1 - t)*mat_start.at<uint8_t>(Point(x - 1, y - 1)) +
				3 * t*mat_start.at<uint8_t>(Point(x + 1, y - 1))
				*(1 - t) + 3 * t*t*mat_start.at<uint8_t>(Point(x + 1, y + 1))*
				(1 - t) + t * t*t*mat_start.at<uint8_t>(Point(x  - 1 , y + 1))
				));


		}

	}
	return mat_start;
}

cv::Mat Catmull_Rom(cv::Mat& img)
{

	cv::Mat mat_start(img.size(), CV_8UC1, Scalar(0));
	int cols = img.cols;
	int rows = img.rows;

	for (int x = 1; x < cols - 1; x++)
	{
		for (int y = 1; y < rows - 1; y++)
		{
			Point curr_point(x, y);
			float dx = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y))) / 2;
			float dy = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x, y + 1))) / 2;

			float sobol = (sqrt(pow(dx, 2) + pow(dy, 2))) - 250;

			mat_start.at<uint8_t>(curr_point) = static_cast<uint8_t>(sobol - 6) * 10;
			double h = img.at<uint8_t>(Point(x + 1, y)) - (img.at<uint8_t>(curr_point));


			if (50 < mat_start.at<uint8_t>(curr_point) && mat_start.at<uint8_t>(curr_point) < 255)

			{
				mat_start.at<uint8_t>(curr_point) = 255;
			}
			else
			{
				mat_start.at<uint8_t>(curr_point) = 0;
			}



			float t = 1.5;

				mat_start.at<uint8_t>(curr_point) +=  1/2*(-t*(1-t)*(1 - t))*
				mat_start.at<uint8_t>(Point(x - 1, y - 1)) + (2 - 5*t*t + 3*t*t*t )*
				mat_start.at<uint8_t>(Point(x + 1, y - 1)) + t*(1+4*t-3*t*t)*
				mat_start.at<uint8_t>(Point(x + 1, y + 1)) - t*t*(1-t)*
				mat_start.at<uint8_t>(Point(x - 1, y + 1));


		}

	}
	return mat_start;
}

cv::Mat B_Spline(cv::Mat& img)
{

	cv::Mat mat_start(img.size(), CV_8UC1, Scalar(0));
	int cols = img.cols;
	int rows = img.rows;

	for (int x = 2; x < cols - 2; x++)
	{
		for (int y = 2; y < rows - 2; y++)
		{
			Point curr_point(x, y);
			float dx = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y))) / 2;
			float dy = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x, y + 1))) / 2;

			float sobol = (sqrt(pow(dx, 2) + pow(dy, 2))) - 250;

			mat_start.at<uint8_t>(curr_point) = static_cast<uint8_t>(sobol - 6) * 10;

			double h = img.at<uint8_t>(Point(x + 1, y)) - (img.at<uint8_t>(curr_point));



			if (50 < mat_start.at<uint8_t>(curr_point) && mat_start.at<uint8_t>(curr_point) < 255)

			{
				mat_start.at<uint8_t>(curr_point) = 255;
			}
			else
			{
				mat_start.at<uint8_t>(curr_point) = 0;
			}

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

cv::Mat dilateBiz(cv::Mat& img, int k)
{
	cv::Mat mat_start(img.size(), CV_8UC1, Scalar(0));
	int cols = img.cols;
	int rows = img.rows;


	for (int x = 1; x < cols - 1; x++)
	{
		for (int y = 1; y < rows - 1; y++)
		{
			Point curr_point(x, y);
			

			double h = img.at<uint8_t>(Point(x + 1, y)) - (img.at<uint8_t>(curr_point));
			float dx = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y))) / 2;
			float dy = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x, y + 1))) / 2;

			float sobol = (sqrt(pow(dx, 2) + pow(dy, 2))) - 250;

			mat_start.at<uint8_t>(curr_point) = static_cast<uint8_t>(sobol - 6) * 10;

			if (45 < mat_start.at<uint8_t>(curr_point) && mat_start.at<uint8_t>(curr_point) < 255)

			{
				mat_start.at<uint8_t>(curr_point) = 255;
			}
			else
			{
				mat_start.at<uint8_t>(curr_point) = 0;
			}

			if (mat_start.at<uint8_t>(curr_point) == 255)
			{
				for (int l = x - k; l <= x + k; l++)
				{
					/*int remainingk = k - abs(x - l);*/
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

			float t = 0.9;
				mat_start.at<uint8_t>(curr_point) += 1 / 2 * pow((-t * (1 - t)), 2)*
				mat_start.at<uint8_t>(Point(x - 1, y - 1)) + (2 - 5 * t*t + 3 * t*t*t)*
				mat_start.at<uint8_t>(Point(x + 1, y - 1)) + t * (1 + 4 * t - 3 * t*t)*
				mat_start.at<uint8_t>(Point(x + 1, y + 1)) - t * t*(1 - t)*
				mat_start.at<uint8_t>(Point(x - 10, y + 1));

		}
	}
	for (int x = 0; x < cols; x++)
	{
		for (int y = 0; y < rows; y++)
		{
			if (mat_start.at<uint8_t>(Point(x, y)) == 2)
			{
				mat_start.at<uint8_t>(Point(x, y)) = 255;
			}
		}
	}


	return mat_start;
}

cv::Mat Splines(cv::Mat& img)
{
	cv::Mat mat_start(img.size(), CV_8UC1, Scalar(0));
	int cols = img.cols;
	int rows = img.rows;


	for (int x = 1; x < cols - 1; x++)
	{
		for (int y = 1; y < rows - 1; y++)
		{
			Point curr_point(x, y);

			double h = img.at<uint8_t>(Point(x + 1, y)) - (img.at<uint8_t>(curr_point));

			float dx = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y))) / 2;
			float dy = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x, y + 1))) / 2;

			float sobol = (sqrt(pow(dx, 2) + pow(dy, 2))) - 250;

			mat_start.at<uint8_t>(curr_point) = static_cast<uint8_t>(sobol - 6) * 10;

			if (45 < mat_start.at<uint8_t>(curr_point) && mat_start.at<uint8_t>(curr_point) < 255)

			{
				mat_start.at<uint8_t>(curr_point) = 255;
			}
			else
			{
				mat_start.at<uint8_t>(curr_point) = 0;
			}

			float t = 2.9;

				mat_start.at<uint8_t>(curr_point) += 1 / 2 * (-t * (1 - t)*(1 - t))*
				mat_start.at<uint8_t>(Point(x - 1, y - 1)) + (2 - 5 * t*t + 3 * t*t*t)*
				mat_start.at<uint8_t>(Point(x + 1, y - 1)) + t * (1 + 4 * t - 3 * t*t)*
				mat_start.at<uint8_t>(Point(x + 1, y + 1)) - t * t*(1 - t)*
				mat_start.at<uint8_t>(Point(x - 1, y + 1));
		
		}

	}

	return mat_start;
}

cv::Mat LOGLith(cv::Mat& img)
{
	cv::Mat mat_start(img.size(), CV_8UC1, Scalar(0));
	cv::Mat mat_f(img.size(), CV_8UC1, Scalar(0));
	int cols = img.cols;
	int rows = img.rows;


	for (int x = 1; x < cols - 1; x++)
	{
		for (int y = 1; y < rows - 1; y++)
		{
			Point curr_point(x, y);


			double h = img.at<uint8_t>(Point(x + 1, y)) - (img.at<uint8_t>(curr_point));
			float dx = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y))) / 2;
			float dy = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x, y + 1))) / 2;

			float sobol = (sqrt(pow(dx, 2) + pow(dy, 2))) - 250;

			mat_start.at<uint8_t>(curr_point) = static_cast<uint8_t>(sobol - 6) * 10;
			/*float D = sqrt(pow(img.at<uint8_t>(Point(x + 1)) - img.at<uint8_t>(Point(x)), 2) +
				pow(img.at<uint8_t>(Point(y + 1)) - img.at<uint8_t>(Point(y)), 2));*/

			if (52 < mat_start.at<uint8_t>(curr_point) && mat_start.at<uint8_t>(curr_point) < 255)

			{
				mat_start.at<uint8_t>(curr_point) = 255;
			}
			else
			{
				mat_start.at<uint8_t>(curr_point) = 0;
			}

			mat_start.at<uint8_t>(curr_point) += 255 * log10f((img.at<uint8_t>(curr_point)) + img.at<uint8_t>(Point(x + 1, y + 1)));

		}
	}
		
	return mat_start;
}

cv::Mat calcKircsha(cv::Mat& img, int k, int z) //operator Kircsha
{
	cv::Mat mat_start(img.size(), CV_8UC1, Scalar(0));
	int cols = img.cols;
	int rows = img.rows;
	
	double E[3][3] = {
	{ -3, -3, 5 },
	{ -3, 5, 5 },
	{ -3, -3, 5 } };
	double NE[3][3] = {
	{ -3, 5, 5 },
	{ -3, 0, 5 },
	{ -3, -3, -3 } };
	double N[3][3] = {
	{ 5, 5, 5 },
	{ -3, 0, -3 },
	{ -3, -3, -3 } };
	double NW[3][3] = {
	{ 5, 5, -3 },
	{ 5, 0, -3 },
	{ -3, -3, -3 } };
	double W[3][3] = {
	{ 5, -3, -3 },
	{ 5, 0, -3 },
	{ 5, -3, -3 } };
	double SW[3][3] = {
	{ -3, -3, -3 },
	{ 5, 0, -3 },
	{ 5, 5, -3 } };
	double S[3][3] = {
	{ -3, -3, -3 },
	{ -3, 0, -3 },
	{ 5, 5, 5 } };
	double SE[3][3] = {
	{ -3, -3, 5 },
	{ -3, 0, 5 },
	{ -3, 5, 5 } };

	for (int x = 1; x < cols - 1; x++)
	{
		for (int y = 1; y < rows - 1; y++)
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
					/*int remainingk = k - abs(x - l);*/
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
	for (int x = 0; x < cols; x++)
	{
		for (int y = 0; y < rows; y++)
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
	cv::Mat mat_start(img.size(), CV_8UC1, Scalar(0));
	int cols = img.cols;
	int rows = img.rows;
	double E[3][3] = {
	{ -1, 0, 1 },
	{ -2, 0, 2 },
	{ -1, 0, 1 } };
	double NE[3][3] = {
	{ 0, 1, 2 },
	{ -1, 0, 1 },
	{ -2, -1, 0 } };
	double N[3][3] = {
	{ 1, 2, 1 },
	{ 0, 0, 0 },
	{ -1, -2, -1 } };
	double NW[3][3] = {
	{ 2, 1, 0 },
	{ 1, 0, -1 },
	{ -0, -1, -2 } };
	double W[3][3] = {
	{ 1, 0, -1 },
	{ 2, 0, -2 },
	{ 1, 0, -1 } };
	double SW[3][3] = {
	{ 0, -1, -1 },
	{ 1, 0, -1 },
	{ 2, 1, 0 } };
	double S[3][3] = {
	{ -1, -2, -1 },
	{ 0, 0, 0 },
	{ 1, 2, 1 } };
	double SE[3][3] = {
	{ -2, -1, 0 },
	{ -1, 0, 1 },
	{ 0, 1, 2 } };

	for (int x = 1; x < cols - 1; x++)
	{
		for (int y = 1; y < rows - 1; y++)
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
					/*int remainingk = k - abs(x - l);*/
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
	for (int x = 0; x < cols; x++)
	{
		for (int y = 0; y < rows; y++)
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
	cv::Mat mat_start(img.size(), CV_8UC1, Scalar(0));
	int cols = img.cols;
	int rows = img.rows;

	GaussianBlur(img, img, Size(3, 3), 25);
	//TODO create filter GaussianBlur(Size(3,3), 25);

	for (int x = 1; x < cols - 1; x++)
	{
		for (int y = 1; y < rows - 1; y++)
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

cv::Mat MarrHildrethNew(cv::Mat& img, float sigm)
{
	cv::Mat mat_start(img.size(), CV_8UC1, Scalar(0));
	int cols = img.cols;
	int rows = img.rows;
	
	GaussianBlur(img, img, Size(3, 3), 25);

	for (int x = 1; x < cols - 1; x++)
	{
		for (int y = 1; y < rows - 1; y++)
		{

			Point curr_point(x, y);

			float marr_hidd = ((img.at<uint8_t>(Point(x + 1, y)) +
				img.at<uint8_t>(Point(x - 1, y)) +
				img.at<uint8_t>(Point(x, y + 1)) +
				img.at<uint8_t>(Point(x, y - 1)) -
				4 * img.at<uint8_t>(Point(x, y)))/2);


			mat_start.at<uint8_t>(curr_point) = static_cast<uint8_t>(marr_hidd * sigm);

		}
	}
	return mat_start;
}

cv::Mat NewFilter(cv::Mat& img) //used filter Laplassian and Gaussian
{
	cv::Mat mat_start(img.size(), CV_8UC1, Scalar(0));
	int cols = img.cols;
	int rows = img.rows;

	int Gx[3][3] = {
		{0,-1,0},
		{-1,4,-1},
		{0,-1,0} };
	int Gy[3][3] = {
		{-1,-1,-1},
		{-1,8,-1},
		{-1,-1,-1} };


	for (int x = 1; x < cols - 1; x++)
	{
		for (int y = 1; y < rows - 1; y++)
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

std::pair<cv::Mat, cv::Mat> calc5x5Gradient(cv::Mat& img)
{
	cv::Mat mat_mod(img.size(), CV_8UC1, Scalar(0));
	cv::Mat mat_angle(img.size(), CV_8UC1, Scalar(0));

	int cols = img.cols;
	int rows = img.rows;

	float max_angle = numeric_limits<double>::min();
	float min_angle = numeric_limits<double>::max();

	for (int x = 2; x < cols - 2; x++)
	{
		for (int y = 2; y < rows - 2; y++)
		{
			vector<uint8_t> arr;
			for (int i = -2; i <= 2; i++)
			{
				for (int j = -2; j <= 2; j++)
				{
					auto cur = img.at<uint8_t>(Point(j + x, i + y));
					arr.push_back(cur);
				}
			}

			uint8_t const& cnt = arr[12];
			float sum = 0.;

			for (auto const it : arr)
			{
				sum += pow(it - cnt, 2);
			}
			float mod = sqrt(sum);

			Point curr_point(x, y);

			mat_mod.at<uint8_t>(curr_point) = 255 - static_cast<uint8_t>(mod);

			float dx = sqrt(pow(arr[0] - cnt, 2) + pow(arr[2] - cnt, 2) + pow(arr[4] - cnt, 2) + pow(arr[10] - cnt, 2) + pow(arr[14] - cnt, 2) + pow(arr[24] - cnt, 2));
			float dy = sqrt(pow(arr[0] - cnt, 2) + pow(arr[2] - cnt, 2) + pow(arr[4] - cnt, 2) + pow(arr[20] - cnt, 2) + pow(arr[22] - cnt, 2) + pow(arr[24] - cnt, 2));

			//float dx = sqrt(pow(arr[0] - cnt, 2) + pow(arr[3] - cnt, 2) + pow(arr[6] - cnt, 2) + pow(arr[3] - cnt, 2) + pow(arr[5] - cnt, 2) + pow(arr[8] - cnt, 2));
			//float dy = sqrt(pow(arr[0] - cnt, 2) + pow(arr[1] - cnt, 2) + pow(arr[2] - cnt, 2) + pow(arr[6] - cnt, 2) + pow(arr[7] - cnt, 2) + pow(arr[8] - cnt, 2));

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

cv::Mat newfil(Mat &img)
{
	cv::Mat mat_start(img.size(), CV_8UC1, Scalar(0));
	cv::Mat mat_sub(img.size(), CV_8UC1, Scalar(0));
	
	int cols = img.cols;
	int rows = img.rows;
	GaussianBlur(img, img, Size(5, 5), 25);

	for (int x = 1; x < cols - 1; x++)
	{
		for (int y = 1; y < rows - 1; y++)
		{
			Point curr_point(x, y);


			//float dx = (img.at<uint8_t>(Point(x+1,y-1)) + 2* img.at<uint8_t>(Point(x + 1, y)) + img.at<uint8_t>(Point(x+1,y+1)) - img.at<uint8_t>(Point(x-1,y-1)) - 2* img.at<uint8_t>(Point(x-1,y))-
			//	img.at<uint8_t>(Point(x-1,y+1)))/4 ;
			//
			//float dy = (img.at<uint8_t>(Point(x - 1, y + 1)) + 2 * img.at<uint8_t>(Point(x, y + 1)) + img.at<uint8_t>(Point(x + 1, y + 1)) - img.at<uint8_t>(Point(x - 1, y - 1)) - 2 * img.at<uint8_t>(Point(x, y - 1)) -
			//	img.at<uint8_t>(Point(x + 1, y - 1))) / 4;
			//float sobol = (sqrt(pow(dx, 2) + pow(dy, 2))) - 250;
			//mat_sub.at<uint8_t>(curr_point) = static_cast<uint8_t>(sobol)*2;


			//float ddx = img.at<uint8_t>(Point(x + 1, y)) - 2 * img.at<uint8_t>(Point(x, y)) + img.at<uint8_t>(Point(x - 1, y));
			//float ddy = img.at<uint8_t>(Point(x, y + 1)) - 2 * img.at<uint8_t>(Point(x, y)) + img.at<uint8_t>(Point(x, y - 1));

			//float sobol = ddx + ddy;

			//mat_sub.at<uint8_t>(curr_point) = static_cast<uint8_t>(sobol);

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





void form_an_image(std::ostream & st) {
	srand((unsigned)time(NULL));
	for (int i = 1; i <= N * 3; i++) {
		st << 0 + rand() % 256 << " ";
		if (i % 3 == 0) st << std::endl;
	}
}

void K_means::identify_centers()
{
	srand((unsigned)time(NULL));
	rgb temp;
	rgb *mas = new rgb[q_klaster];
	for (int i = 0; i < q_klaster; i++) {
		temp = pixcel[0 + rand() % k_pixcel];
		for (int j = i; j < q_klaster; j++) {
			if (temp.r != mas[j].r && temp.g != mas[j].g && temp.b != mas[j].b) {
				mas[j] = temp;
			}
			else {
				i--;
				break;
			}
		}
	}
	for (int i = 0; i < q_klaster; i++) {
		centr.push_back(mas[i]);
	}
	delete[]mas;
}

K_means::K_means(int n, rgb * mas, int n_klaster)
{
	for (int i = 0; i < n; i++) {
		pixcel.push_back(*(mas + i));
	}
	q_klaster = n_klaster;
	k_pixcel = n;
	identify_centers();
}

K_means::K_means(int n_klaster, std::istream & os) : q_klaster(n_klaster)
{
	rgb temp;
	while (os >> temp.r && os >> temp.g && os >> temp.b) {
		pixcel.push_back(temp);
	}
	k_pixcel = pixcel.size();
	identify_centers();
}

void K_means::clustering(std::ostream & os)
{
	os << "\n\n������ �������������:" << std::endl;

	std::vector<int> check_1(k_pixcel, -1);
	std::vector<int> check_2(k_pixcel, -2);
	int iter = 0;

	while (true)
	{
		os << "\n\n---------------- �������� �"
			<< iter << " ----------------\n\n";
		{
			for (int j = 0; j < k_pixcel; j++) {
				double *mas = new double[q_klaster];

				for (int i = 0; i < q_klaster; i++) {
					*(mas + i) = compute(pixcel[j], centr[i]);
					os << "���������� �� ������� " << j << " � ������ #"
						<< i << ": " << *(mas + i) << std::endl;
				}

				double min_dist = *mas;
				int m_k = 0;
				for (int i = 0; i < q_klaster; i++) {
					if (min_dist > *(mas + i)) {
						min_dist = *(mas + i);
						m_k = i;
					}
				}
				os << "����������� ���������� � ������ #" << m_k << std::endl;
				os << "������������� ����� #" << m_k << ": ";
				centr[m_k].r = compute_s(pixcel[j].r, centr[m_k].r);
				centr[m_k].g = compute_s(pixcel[j].g, centr[m_k].g);
				centr[m_k].b = compute_s(pixcel[j].b, centr[m_k].b);
				os << centr[m_k].r << " " << centr[m_k].g
					<< " " << centr[m_k].b << std::endl;
				delete[] mas;
			}

			int *mass = new int[k_pixcel];
			os << "\n�������� ������������� ��������: " << std::endl;
			for (int k = 0; k < k_pixcel; k++) {
				double *mas = new double[q_klaster];

				for (int i = 0; i < q_klaster; i++) {
					*(mas + i) = compute(pixcel[k], centr[i]);
					os << "���������� �� ������� �" << k << " � ������ #"
						<< i << ": " << *(mas + i) << std::endl;
				}

				double min_dist = *mas;
				int m_k = 0;
				for (int i = 0; i < q_klaster; i++) {
					if (min_dist > *(mas + i)) {
						min_dist = *(mas + i);
						m_k = i;
					}
				}
				mass[k] = m_k;
				os << "������� �" << k << " ����� ����� � ������ #" << m_k << std::endl;
			}

			os << "\n������ ������������ �������� � �������: \n";
			for (int i = 0; i < k_pixcel; i++) {
				os << mass[i] << " ";
				check_1[i] = *(mass + i);
			}
			os << std::endl << std::endl;

			os << "��������� �������������: " << std::endl;
			int itr = KK + 1;
			for (int i = 0; i < q_klaster; i++) {
				os << "������� #" << i << std::endl;
				for (int j = 0; j < k_pixcel; j++) {
					if (mass[j] == i) {
						os << pixcel[j].r << " " << pixcel[j].g
							<< " " << pixcel[j].b << std::endl;
						mass[j] = ++itr;
					}
				}
			}

			delete[] mass;

			os << "����� ������: \n";
			for (int i = 0; i < q_klaster; i++) {
				os << centr[i].r << " " << centr[i].g
					<< " " << centr[i].b << " - #" << i << std::endl;
			}
		}
		iter++;
		if (check_1 == check_2 || iter >= max_iterations) {
			break;
		}
		check_2 = check_1;
	}
	os << "\n\n����� �������������." << std::endl;
}

void K_means::print() const
{
	std::cout << "��������� �������: " << std::endl;
	for (int i = 0; i < k_pixcel; i++) {
		std::cout << pixcel[i].r << " " << pixcel[i].g
			<< " " << pixcel[i].b << " - �" << i << std::endl;
	}
	std::cout << std::endl << "��������� ��������� ������ �������������: " << std::endl;
	for (int i = 0; i < q_klaster; i++) {
		std::cout << centr[i].r << " " << centr[i].g << " "
			<< centr[i].b << " - #" << i << std::endl;
	}
	std::cout << "\n���������� ���������: " << q_klaster << std::endl;
	std::cout << "���������� ��������: " << k_pixcel << std::endl;
}

std::ostream & operator<<(std::ostream & os, const K_means & k)
{
	os << "��������� �������: " << std::endl;
	for (int i = 0; i < k.k_pixcel; i++) {
		os << k.pixcel[i].r << " " << k.pixcel[i].g
			<< " " << k.pixcel[i].b << " - �" << i << std::endl;
	}
	os << std::endl << "��������� ��������� ������ �������������: " << std::endl;
	for (int i = 0; i < k.q_klaster; i++) {
		os << k.centr[i].r << " " << k.centr[i].g << " "
			<< k.centr[i].b << " - #" << i << std::endl;
	}
	os << "\n���������� ���������: " << k.q_klaster << std::endl;
	os << "���������� ��������: " << k.k_pixcel << std::endl;
	return os;
}

K_means::~K_means()
{
}


cv::Mat MatrixGrad(cv::Mat& img , int h)
{
	cv::Mat sub_mat(img.size(), CV_8UC1, Scalar(0));
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
	
	int cols = src.cols;
	int rows = src.rows;

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





//cv::Mat superPix(cv::Mat& img)
//{
//	cv::Mat matrix_pix(img.size(), CV_8UC1, Scalar(0));
//	int rows = img.rows;
//	int cols = img.cols;
//
//	for (int i = 0; i < rows; i++)
//	{
//		for (int j = 0; j < cols; j++)
//		{
//			Point curr_point(i, j);
//
//			
//
//			float dx1 = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(i + 1, j))) / 2;
//			float dy1 = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(i, j + 1))) / 2;
//
//			float sobol = (sqrt(pow(dx1, 2) + pow(dy1, 2))) - 250;
//			matrix_pix.at<uint8_t>(curr_point) = static_cast<uint8_t>(sobol - 6) * 10;
//
//			int nx, ny;
//			int m;
//			nx = 100;
//			ny = 100;
//			m = 20;
//
//		// Scale to [0,1] and l*a*b colorspace
//			matrix_pix.convertTo(img, CV_32F, 1 / 255.);
//		Mat imlab;
//		cvtColor(matrix_pix, imlab, cv::COLOR_BGR2Lab); //cv::COLOR_BGR2Lab);
//
//		int h = matrix_pix.rows;
//		int w = matrix_pix.cols;
//		int n = nx * ny;
//
//		float dx = w / float(nx);
//		float dy = h / float(ny);
//		int S = (dx + dy + 1) / 2; // window width
//
//		// Initialize centers
//		vector<Point> centers;
//		for (int i = 0; i < ny; i++) 
//		{
//			for (int j = 0; j < nx; j++) 
//			{
//				centers.push_back(Point(j * dx + dx / 2, i * dy + dy / 2));
//			}
//		}
//
//		// Initialize labels and distance maps
//		vector<int> label_vec(n);
//		for (int i = 0; i < n; i++)
//			label_vec[i] = i * 255 * 255 / n;
//
//		Mat labels = -1 * Mat::ones(imlab.size(), CV_32S);
//		Mat dists = -1 * Mat::ones(imlab.size(), CV_32F);
//		Mat window;
//		Point p1, p2;
//		Vec3f p1_lab, p2_lab;
//
//		// Iterate 10 times. In practice more than enough to converge
//		for (int i = 0; i < 10; i++) 
//		{
//			// For each center...
//			for (int c = 0; c < n; c++)
//			{
//				int label = label_vec[c];
//				p1 = centers[c];
//				p1_lab = imlab.at<Vec3f>(p1);
//				int xmin = max(p1.x - S, 0);
//				int ymin = max(p1.y - S, 0);
//				int xmax = min(p1.x + S, w - 1);
//				int ymax = min(p1.y + S, h - 1);
//
//				// Search in a window around the center
//				window = matrix_pix(Range(ymin, ymax), Range(xmin, xmax));
//
//				// Reassign pixels to nearest center
//				for (int i = 0; i < window.rows; i++)
//				{
//					for (int j = 0; j < window.cols; j++)
//					{
//						p2 = Point(xmin + j, ymin + i);
//						p2_lab = imlab.at<Vec3f>(p2);
//						float d = dist(p1, p2, p1_lab, p2_lab, m, S);
//						float last_d = dists.at<float>(p2);
//						if (d < last_d || last_d == -1)
//						{
//							dists.at<float>(p2) = d;
//							labels.at<int>(p2) = label;
//						}
//					}
//				}
//			}
//		}
//
//		// Calculate superpixel boundaries
//		labels.convertTo(labels, CV_32F);
//		Mat gx, gy, grad;
//		filter2D(labels, gx, -1, sobel1);
//		filter2D(labels, gy, -1, sobel1.t());
//		magnitude(gx, gy, grad);
//		grad = (grad > 1e-4) / 255;
//		Mat show = 1 - grad;
//		show.convertTo(show, CV_32F);
//
//		// Draw boundaries on original image
//		vector<Mat> rgb(3);
//		split(matrix_pix, rgb);
//		for (int i = 0; i < 3; i++)
//			rgb[i] = rgb[i].mul(show);
//		merge(rgb, matrix_pix);
//		imshow("result", matrix_pix);
//		}
//
//	}
//
//	return matrix_pix;
//	
//}


Mat myfilter2d(Mat img, Mat filter) {
	Mat dst = img.clone();
	cout << " filter data successfully found.  Rows:" << filter.rows << " cols:" << filter.cols << " channels:" << filter.channels() << "\n";
	cout << " input data successfully found.  Rows:" << img.rows << " cols:" << img.cols << " channels:" << img.channels() << "\n";

	for (int i = 0 - (filter.rows / 2); i < img.rows - (filter.rows / 2); i++) {
		for (int j = 0 - (filter.cols / 2); j < img.cols - (filter.cols / 2); j++) {  //adding k and l to i and j will make up the difference and allow us to process the whole image
			float filtertotal = 0;
			for (int k = 0; k < filter.rows; k++) {
				for (int l = 0; l < filter.rows; l++) {
					if (i + k >= 0 && i + k < img.rows && j + l >= 0 && j + l < img.cols) {  //don't try to process pixels off the endge of the map
						float a = img.at<uint8_t>(i + k, j + l);
						float b = filter.at<float>(k, l);
						float product = a * b;
						filtertotal += product;
					}
				}
			}
			//filter all proccessed for this pixel, write it to dst
			dst.at<uint8_t>(i + (filter.rows / 2), j + (filter.cols / 2)) = filtertotal;

		}
	}
	return dst;
}


Mat convolute2D(Mat& img, double** kernel, int W) {
	Mat filtered_image = img.clone();
	// find center position of kernel (half of kernel size)
	int kCenterX = W / 2;
	int kCenterY = W / 2;
	int xx = 0;
	int yy = 0;
	cout << endl << "Performing convolution .." << endl;
	cout << "Image Size : " << img.rows << ", " << img.cols << endl;
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			for (int x = 0; x < W; ++x) {
				xx = W - 1 - x;
				for (int y = 0; y < W; ++y) {
					yy = W - 1 - y;
					int ii = i + (x - kCenterX);
					int jj = j + (y - kCenterY);
					if (ii >= 0 && ii < img.rows && jj >= 0 && jj < img.cols) {
						filtered_image.at<uint8_t>(Point(j, i)) +=static_cast<uint>( img.at<uint8_t>(Point(jj, ii)) * kernel[xx][yy]);
					}
				}
			}
		}
	}
	return filtered_image;
}