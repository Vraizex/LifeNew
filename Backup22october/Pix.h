#pragma once
#define _USE_MATH_DEFINES
#define HARD 1  // Жесткий
#define SOFT 2  // Мягкий
#define GARROT 3  // Фильтр Гаррота

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/video/tracking.hpp"
#include <opencv2/highgui.hpp>

#include <iostream>
#include <cmath>
#include <limits>
#include <stdio.h>
#include <vector>
#include <fstream> 
#include <sstream>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include "morphology.h"

using namespace std;
using namespace cv;

class Window
{
public:
	Window(std::string const& title) : _title(title), _is_opened(false) {
		namedWindow(_title, cv::WINDOW_NORMAL);
	}
	void show(cv::Mat const& m, bool wait_flag = false) {
		imshow(_title, m);
		_is_opened = true;
		if (wait_flag) {
			waitKey(0);
		}
	}
	static void wait(int timeout = 0) {
		waitKey(timeout);
	}
	~Window() {
		if (_is_opened) {
			destroyWindow(_title);
		}
	}
protected:
	std::string _title;
	bool _is_opened;
};

class Sobla_kernal
{
public:
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
		for (int y = 1; y < src.rows - 1; y++) {
			for (int x = 1; x < src.cols - 1; x++) {
				sum = 0.0;
				for (int k = -1; k <= 1; k++) {
					for (int j = -1; j <= 1; j++) {
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
		for (int y = 0; y < src.rows; y++) {
			for (int x = 0; x < src.cols; x++) {
				sum = 0.0;
				for (int k = -1; k <= 1; k++) {
					for (int j = -1; j <= 1; j++) {
						x1 = reflect(src.cols, x - j);
						y1 = reflect(src.rows, y - k);
						sum +=  Kernel[j + 1][k + 1] * src.at<uchar>(y1, x1);
					}
				}
				dst.at<uchar>(y, x) = sum;
			}
		}
	}

	void circularIndexing(Mat & src, Mat  & dst, float Kernel[][3]) //координаты которые выходят за границы изображения переходят на противоположную сторону
	{
		float sum, x1, y1;
		for (int y = 0; y < src.rows; y++) {
			for (int x = 0; x < src.cols; x++) {
				sum = 0.0;
				for (int k = -1; k <= 1; k++) {
					for (int j = -1; j <= 1; j++) {
						x1 = circular(src.cols, x - j);
						y1 = circular(src.rows, y - k);
						sum += Kernel[j + 1][k + 1] * src.at<uchar>(y1, x1);
					}
				}
				dst.at<uchar>(y, x) = sum;
			}
		}
	}

	int xGradient(Mat & img, int x, int y) //Gradient dx
	{
		return img.at<uchar>(y - 1, x - 1) +
			2 * img.at<uchar>(y, x - 1) +
			img.at<uchar>(y + 1, x - 1) -
			img.at<uchar>(y - 1, x + 1) -
			2 * img.at<uchar>(y, x + 1) -
			img.at<uchar>(y + 1, x + 1);
	}

	int yGradient(Mat & img, int x, int y) //Gradient dy
	{
		return img.at<uchar>(y - 1, x - 1) +
			2 * img.at<uchar>(y - 1, x) +
			img.at<uchar>(y - 1, x + 1) -
			img.at<uchar>(y + 1, x - 1) -
			2 * img.at<uchar>(y + 1, x) -
			img.at<uchar>(y + 1, x + 1);
	}
	
	
	//void Dilation(Mat& img, bool* mask[], Mat& dst)
	//{
	//	// W, H – размеры исходного и результирующего изображений
	//	// MW, MH – размеры структурного множества
	//	for (int y = MH / 2; y < img.cols – MH / 2; y++)
	//	{
	//		for (int x = MW / 2; x < img.rows – MW / 2; x++)
	//		{
	//			Mat max = 0;
	//			for (int j = -MH / 2; j <= MH / 2; j++)
	//			{
	//				for (int i = -MW / 2; i <= MW / 2; i++)
	//					if ((mask[i][j]) && (img[x + i][y + j] > max))
	//					{
	//						max = img[x + i][y + j];
	//					}
	//			}
	//			dst[x][y] = max;
	//		}
	//	}
	//}


	//void Erosion(Mat& img, bool* mask[], Mat& dst)
	//{
	//	cv::Mat sub_mat(img.size(), CV_8UC1, Scalar(0));
	//	// W, H – размеры исходного и результирующего изображений
	//	// MW, MH – размеры структурного множества
	//	for (int y = MH / 2; y < img.cols – MH / 2; y++)
	//	{
	//		for (int x = MW / 2; x < img.rows – MW / 2; x++)
	//		{
	//			Mat min = MAXBIT;
	//			for (int j = -MH / 2; j <= MH / 2; j++)
	//			{
	//				for (int i = -MW / 2; i <= MW / 2; i++)
	//					if ((mask[i][j]) && (img[x + i][y + j] < min))
	//					{
	//						min = img[x + i][y + j];
	//					}
	//			}
	//			dst[x][y] = min;
	//		}
	//	}
	//}




	//Sobla_kernal();
	//~Sobla_kernal();

private:

};

//Sobla_kernal::Sobla_kernal()
//{
//}
//
//Sobla_kernal::~Sobla_kernal()
//{
//}


//cv::Mat dilateXY(cv::Mat& img);
//cv::Mat dilate3X3Y(cv::Mat& img);
//cv::Mat dilateGOA(cv::Mat& img, int k);
//cv::Mat dilateAndErozia(cv::Mat& img, int k, int z);
//cv::Mat EroziaAndDilate(cv::Mat& img, int k, int z);
//cv::Mat dilateY(cv::Mat& img);
//cv::Mat dilateX(cv::Mat& img);
//cv::Mat dilateMXN(cv::Mat& img, int k, int z);
//cv::Mat dilateEroziaLevel(cv::Mat& img, int k);

//#include "svertki.hpp"
//cv::Mat calcLoGDiskret(cv::Mat& img);
//cv::Mat calcLoGDiskretWeights(cv::Mat& img);
//cv::Mat calcLoGDiskretWeightsProg(cv::Mat& img);
//cv::Mat calc3x3GradientSOBOLBinAndMatrix(cv::Mat& img);
//cv::Mat cvHaarWavelet(cv::Mat &img, cv::Mat &dst, int NIter);
//pair<cv::Mat, cv::Mat> calc3x3Gradient(cv::Mat& img);
//cv::Mat cvInvHaarWavelet(Mat &src, Mat &dst, int NIter, int SHRINKAGE_TYPE = 0, float SHRINKAGE_T = 50); // Вейвлет-преобразование
//cv::Mat calcVGradient(cv::Mat& img);
//cv::Mat BinandDeleteOnlyPixels(cv::Mat& img);
//cv::Mat lagrange2(cv::Mat& img);
//cv::Mat Catmull_Rom(cv::Mat& img);
//cv::Mat B_Spline(cv::Mat& img);
//cv::Mat Splines(cv::Mat& img);
//cv::Mat LOGLith(cv::Mat& img);
//cv::Mat calcKircsha(cv::Mat& img, int k, int z);
//cv::Mat calcRobinsone(cv::Mat& img, int k);
//cv::Mat MarrHildeth(cv::Mat& img, float sigm);
//cv::Mat MarrHildrethNew(cv::Mat& img, float sigm);
//cv::Mat NewFilter(cv::Mat& img);



//#include "image_filters.hpp"
//cv::Mat calcSOBOL(cv::Mat& img);
//cv::Mat calcLoG(cv::Mat& img);
//cv::Mat calcPrevitta(cv::Mat& img);
//cv::Mat calcRobertsa(cv::Mat& img);
//cv::Mat calcHGradient(cv::Mat& img);
//cv::Mat NewSobol(cv::Mat& img);
//cv::Mat NewShar(cv::Mat& img);
//cv::Mat NewGradientPrevitta(cv::Mat& img);
//cv::Mat NewLoG(cv::Mat& img);
//cv::Mat dilateBiz(cv::Mat& img, int k);
//std::pair<cv::Mat, cv::Mat> calc5x5Gradient(cv::Mat& img);
//cv::Mat newfil(cv::Mat& img);
//cv::Mat MatrixGrad(cv::Mat& img, int h);
//cv::Mat calcHough(cv::Mat& img);




//cv::Mat dilate(cv::Mat const& img, Kernel const &kernel);
//
//auto result = delate(erozia(img, { 3, 3 }), { 3, 3 });
//
//cv::Mat dilateAndErozia(mat const& m, {}) {
//	delate(erozia(img, { 3, 3 }), { 3, 3 });
//}




cv::Mat Bin(cv::Mat& img);





const int N = 50; //количество пикселей для случайной генерации данных
const int KK = 10; //количество кластеров
const int max_iterations = 100; //максимальное количество итераций

typedef struct {            //пиксель
	double r;
	double g;
	double b;
} rgb;

//void form_an_image(std::ostream & st); //функция записи в файл 
//									   //каждого пикселя

class K_means
{
private:
	std::vector<rgb> pixcel; //вектор пикселей
	int q_klaster;           //количество кластеров
	int k_pixcel;            //количество пикселей
	std::vector<rgb> centr;  //центры кластеризации
	void identify_centers(); //метод случайного выбора начальных центров
	inline double compute(rgb k1, rgb k2)
	{
		return sqrt(pow((k1.r - k2.r), 2) + pow((k1.g - k2.g), 2) + pow((k1.b - k2.b), 2));
	}
	inline double compute_s(double a, double b) {
		return (a + b) / 2;
	};
public:
	K_means() : q_klaster(0), k_pixcel(0) {}; //конструкторы
	K_means(int n, rgb *mas, int n_klaster);
	K_means(int n_klaster, std::istream & os);
	void clustering(std::ostream & os); //метод кластерезации
	void print()const; //метод вывода
	~K_means();
	friend std::ostream & operator<<(std::ostream & os, const K_means & k); //перегружений оператор << 
};



Mat kMeans(Mat& img);

//Methods release
Mat_<Vec3b> k_means(Mat_<Vec3b> & src);
