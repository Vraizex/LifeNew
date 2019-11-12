#include "Pix.h"
#include "image_filters.h"
#include "convolution.h"
#include "morphology.h"

#include <iostream>
#include <random>
#include <time.h>

Mat sobel = (Mat_<float>(3, 3) << -1 / 16., -2 / 16., -1 / 16., 0, 0, 0, 1 / 16., 2 / 16., 1 / 16.);

float dist(Point p1, Point p2, Vec3f p1_lab, Vec3f p2_lab, float compactness, float S);



int main()
{
	try 
	{
		setlocale(LC_ALL, "Russian");
		
		/*string file_path = "C:\\Users\\User\\Downloads\\Lenna.png";*/
		/*std::string file_path = "C:\\Users\\User\\Downloads\\Telegram Desktop\\DataSet_V\\DataSet_V\\img0_.png";*/
		std::string file_path = "C:\\Users\\User\\Downloads\\im0_.png";
		std::string file_path1 = "C:\\Users\\User\\Downloads\\im0_.png";
		cv::Mat labels;
		cv::Mat img = imread(file_path, 0);
		cv::Mat img1 = imread(file_path, 0);
		cv::Mat dst; /*= imread(file_path, 1);*/
		cv::Mat_<Vec3b> src1 = imread(file_path, 1);
		cv::Mat src = imread(file_path, 0);
		
		if (!img.data) // no image loading;
		{
			throw std::system_error(errno, std::system_category(), file_path);
		}
		

		//
		//morphology::Kernel delatation_kernel{ 15, 3};
		//img = morphology::dilate(img, delatation_kernel);// delatation_kernel);
		////Example
		////img = morphology::dilate(img, morphology::Kernel{ 5, 5});
		//namedWindow("Dilate");
		//imshow("Dilate", img);
		
		morphology::Kernel sizes{ 1, 1};
		img1 = IP::sobol(img);

		//img1 = morphology::eroze(img1, sizes);// 10, 1);
		namedWindow("Eroze");
		imshow("Erozia",img1);

		//namedWindow("Sobel");
		//imshow("Sobel", dst = img - img1);

		float Kernel[3][3] = 
		{					
			{ 1, 1, 1},
			{ 1,-12, 1},
			{ 1, 1, 1},
		};

		dst = img.clone();
		for (int y = 0; y < img.rows; y++)
		{
			for (int x = 0; x < img.cols; x++)
			{
				dst.at<uchar>(y, x) = 0.0;
			}
		}
		convolution::circularIndexing(img, dst, Kernel);

		namedWindow("Kernal for image");
		imshow("Kernal for image", dst);

		//namedWindow("Just image");
		//imshow("Just image", img);

		// K_MEANS Good Workkks
		//k_means(src);

		Window::wait();
	}
	catch (std::exception const& ex)
	{
		std::cout << "fail: " << ex.what() << std::endl;
	}
	system("pause");
}

float dist(Point p1, Point p2, Vec3f p1_lab, Vec3f p2_lab, float compactness, float S)
{
	float dl = p1_lab[0] - p2_lab[0];
	float da = p1_lab[1] - p2_lab[1];
	float db = p1_lab[2] - p2_lab[2];
	float d_lab = sqrtf(dl*dl + da * da + db * db);
	float dx = p1.x - p2.x;
	float dy = p1.y - p2.y;
	float d_xy = sqrtf(dx*dx + dy * dy);

	return d_lab + compactness / S * d_xy;
}

double magGrad(Point p1, Point p2)
{
	float dx = p1.x - p2.x;
	float dy = p1.y - p2.y;

	return sqrtf((dx*dx) + (dy*dy));
}

