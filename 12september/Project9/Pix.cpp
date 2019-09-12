
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
	
	float c,b, dh, dv, dd;
	int width = img.cols;
	int height = img.rows;
	for (int k = 0; k < NIter; k++)
	{
		for (int y = 0; y < (width >> (k + 1)); y++)
		{
			for (int x = 0; x < (width /*>> (k + 1)*/); x++)
			{
				c = (img.at<uint8_t>( y, x) + img.at<uint8_t>( y,  x + 1) + img.at<uint8_t>( y + 1,  x) + img.at<uint8_t>( y + 1,  x + 1));
				dst.at<uint8_t>(y, x) = c;
			
				/*dst.at<uint8_t>(y + (height >> (k + 1)), x) = c;*/

			

				/*dh = (img.at<uint8_t>( y,  x) + img.at<uint8_t>( y + 1,  x) - img.at<uint8_t>( y,  x + 1) - img.at<uint8_t>( y + 1, 2 x + 1));
				dst.at<uint8_t>(y, x + (width >> (k + 1))) = dh;*/

				/*
				dv = (img.at<uint8_t>(2 * y, 2 * x) + img.at<uint8_t>(2 * y, 2 * x + 1) - img.at<uint8_t>(2 * y + 1, 2 * x) - img.at<uint8_t>(2 * y + 1, 2 * x + 1));
				dst.at<uint8_t>(y + (height >> (k + 1)), x) = dv;

				dd = (img.at<uint8_t>(2 * y, 2 * x) - img.at<uint8_t>(2 * y, 2 * x + 1) - img.at<uint8_t>(2 * y + 1, 2 * x) + img.at<uint8_t>(2 * y + 1, 2 * x + 1));
				dst.at<uint8_t>(y + (height >> (k + 1)), x + (width >> (k + 1))) = dd;*/
			
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


cv::Mat calc3x3GradientSOBOL(cv::Mat& img)
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




cv::Mat calc3x3GradientPrevitta(cv::Mat& img)
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

			float dx1 = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y))) / 2;
			float dy1 = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x, y + 1))) / 2;
			float dx2 = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y + 1))) / 2;
			float dy2 = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y - 1))) / 2;
			float dx3 = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x , y - 1))) / 2;
			float dy3 = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x - 1, y  - 1))) / 2;
			float dx4 = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x - 1, y))) / 2;
			float dy4 = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x - 1, y + 1))) / 2;

			float sobol = sqrt(pow(dx1, 2) + pow(dy1, 2) + pow(dx2, 2) + pow(dy2, 2) + pow(dx3, 2) + pow(dy3, 2) + pow(dx4, 2) + pow(dy4, 2)) - 250;


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
	cv::Mat sub_mat(img.size(), CV_8UC1, Scalar(0));
	for (int x = 1; x < img.cols - 1; x++)
	{
		for (int y = 1; y < img.rows - 1; y++)
		{
			Point curr_point(x, y);

			sub_mat.at<uint8_t>(curr_point) = (img.at<uint8_t>(Point(x + 1, y)) +
				img.at<uint8_t>(Point(x - 1, y)) +
				img.at<uint8_t>(Point(x , y + 1)) + 
				img.at<uint8_t>(Point(x, y - 1)) - 
				4 * img.at<uint8_t>(curr_point))/2;


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


void transpose(int matrix[3][3]) 
{
    int t;
    for(int i = 0; i < 3; ++i)
    {
        for(int j = i; j < 3; ++j)
        {
            t = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = t;
        }
    }
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




Mat dilateY(Mat& img) {


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




Mat dilateX(Mat& img) {


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



Mat dilateXY(Mat& img) {
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

Mat dilate3X3Y(Mat& img) {


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


Mat dilateGOA(Mat& img , int k)
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
					int remainingk = k - abs(x - l);
					for (int m = y - remainingk; m <= y + remainingk; m++) 
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






Mat dilateYEroze(Mat& img) {


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




			if (mat_start.at<uint8_t>(curr_point) == 0)
			{

				if (x > 0 &&
					mat_start.at<uint8_t>(Point(x - 1, y)) == 255)
					mat_start.at<uint8_t>(Point(x - 1, y)) = 2;

				if (y > 0 &&
					mat_start.at<uint8_t>(Point(x, y - 1)) == 255)
					mat_start.at<uint8_t>(Point(x, y - 1)) = 2;

				if (x + 1 < mat_start.cols &&
					mat_start.at<uint8_t>(Point(x + 1, y)) == 255)
					mat_start.at<uint8_t>(Point(x + 1, y)) = 2;

				if (y + 1 < mat_start.rows &&
					mat_start.at<uint8_t>(Point(x, y + 1)) == 255)
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

Mat dilateEroziaLevel(Mat& img, int k)
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
					int remainingk = k - abs(x - l);
					for (int m = y - remainingk; m <= y + remainingk; m++)
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