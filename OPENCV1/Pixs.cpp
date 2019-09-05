#include <QtCore/QCoreApplication>

#define _USE_MATH_DEFINES

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui.hpp>


#include <iostream>
#include <cmath>
#include <limits>
#include "Pixs.h"

using namespace std;
using namespace cv;



int main(int argc, char *argv[])
{
	QCoreApplication a(argc, argv);
	
	try {
		setlocale(LC_ALL, "Russian");
		std::string file_path = "C:\\Users\\User\\Downloads\\im0_.png";
		Mat img = imread(file_path, 0);
		if (!img.data) 
		{
			throw std::system_error(errno, std::system_category(), file_path);
		}

		Mat mat_horizontal = calcHGradient(img);
		Window win_horizontal("Gradiend Field horizontal");
		win_horizontal.show(mat_horizontal);


		Mat mat_all = allGradient(img);
		Window win_all("Gradiend Field All");
		win_all.show(mat_all);

		auto mat3x3 = calc3x3Gradient(img);
		Window win_mod_3x3("Gradiend Field 3x3 module");
		Mat modules;
		mat3x3.first.convertTo(modules, -1, 0.8, 30.);
		win_mod_3x3.show(modules);


		Window win_angle_3x3("Gradiend Field 3x3 angle");
		win_mod_3x3.show(modules);
		win_angle_3x3.show(mat3x3.second);

		Window foobar("Foobar");
		foobar.show(mat3x3.first / 2 + mat3x3.second / 2);
		Window::wait();

	}
	catch (std::exception const& ex)
	{
		std::cout << "fail: " << ex.what() << std::endl;
	}

	system("pause");
}


