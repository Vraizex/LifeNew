
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

class Window
{
public:
	Window(std::string const& title): _title(title), _is_opened(false) {
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

//void waitImage(string const& title, Mat const& img)
//{
//	namedWindow(title, WINDOW_NORMAL);
//	imshow(title, img);
//	waitKey();
//	destroyWindow(title);
//}

cv::Mat calcHGradient(cv::Mat& img);
cv::Mat calcVGradient(cv::Mat& img);
cv::Mat allGradient(cv::Mat& img);
std::pair<cv::Mat, cv::Mat> calc3x3Gradient(cv::Mat& img);

int main()
{
	try {
		setlocale(LC_ALL, "Russian");
		/*std::string file_path = "C:\\Users\\User\\Downloads\\Lenna.png"*/;
		/*std::string file_path = "C:\\Users\\User\\Downloads\\Telegram Desktop\\DataSet_V\\DataSet_V\\img0_.png";*/
		std::string file_path = "C:\\Users\\User\\Downloads\\im0_.png";
		Mat img = imread(file_path, 0);
		if (!img.data) // no image loading;
		{
			throw std::system_error(errno, std::system_category(), file_path);
		}


		//GaussianBlur(img, img, { 25, 25 }, 5);


		Mat mat_horizontal = calcHGradient(img);
		Window win_horizontal("Gradiend Field horizontal");
		
		win_horizontal.show(mat_horizontal);

		//Mat mat_vertical = calcVGradient(img);
		//Window win_vertical("Gradiend Field vertical");
		//win_vertical.show(mat_vertical);

		Mat mat_all = allGradient(img);
		Window win_all("Gradiend Field All");

		win_all.show(mat_all);

		auto mat3x3 = calc3x3Gradient(img);

		Window win_mod_3x3("Gradiend Field 3x3 module");
		
		

		Window win_angle_3x3("Gradiend Field 3x3 angle");

		Mat modules;
		mat3x3.first.convertTo(modules,-100, 1, 2);

		win_mod_3x3.show(modules);// mat3x3.first);
		imwrite("C:\\Users\\User\\Downloads\\angle.png", mat3x3.second);
		win_angle_3x3.show(mat3x3.second);

		Window foobar("Foobar");

		foobar.show(mat3x3.first/2 + mat3x3.second/2);

		Window::wait();

/*
		//GaussianBlur(sub_mat, sub_mat, Size(25, 25), 2.5);
		

		erode(sub_mat, sub_mat, getStructuringElement(MORPH_RECT, Size(5, 5)));
		waitImage("Angle", sub_mat);

		//Canny(sub_mat, sub_mat, 100, 200);
		//waitImage("New_pic", sub_mat);

		double max_angle = numeric_limits<double>::min();
		double min_angle = numeric_limits<double>::max();


		for (int x = 1; x < sub_mat.cols - 1; x++)
		{
			for (int y = 1; y < sub_mat.rows - 1; y++)
			{
				//for (int N = -1; N < 2; N++)


				Point curr_point(x, y);

				uint8_t neighPixel1 = sub_mat.at<uint8_t>(x, y);
				uint8_t neighPixel2 = sub_mat.at<uint8_t>(x + 1, y + 1);

				double b = pow((pow(neighPixel1, 2) - pow(neighPixel2, 2)), 1 / 2);

				cout << " b = " << b << endl;


				double angle = atan2(sub_mat.at<uint8_t>(x) - sub_mat.at<uint8_t>(x + 1), sub_mat.at<uint8_t>(y) - sub_mat.at<uint8_t>(y + 1));
				double angles = angle * 180 / M_PI;

				max_angle = max(angle, max_angle);
				min_angle = min(angle, min_angle);

				uint8_t val = static_cast<uint8_t>((angle + 3.2) * 35);

				sub_mat.at<uint8_t>(curr_point) = val;

			}
		}

		cout << "max angle: " << min_angle << " : " << max_angle << endl;
		waitImage("Used matrix 3X3", sub_mat);
*/
	}
	catch (std::exception const& ex) 
	{
		std::cout << "fail: " << ex.what() << std::endl;
	}

	system("pause");

}

cv::Mat calcHGradient(cv::Mat& img)
{
	cv::Mat sub_mat(img.size(), CV_8UC1, Scalar(0));
	for (int x = 1; x < img.cols - 1; x++)
	{
		for (int y = 1; y < img.rows - 1; y++)
		{
			Point curr_point(x, y);
			sub_mat.at<uint8_t>(curr_point) = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y)) + 255) / 2 ;
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
			sub_mat.at<uint8_t>(curr_point) = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x, y + 1 )) + 255) / 2;
		}
	}

	return sub_mat;
}

cv::Mat allGradient(cv::Mat& img)
{
	cv::Mat sub_mat(img.size(), CV_8UC1, Scalar(0));
	for (int x = 1; x < img.cols - 1; x++)
	{
		for (int y = 1; y < img.rows - 1; y++)
		{
			Point curr_point(x, y);
			uint8_t dx = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x + 1, y)) + 255) / 2;
			uint8_t dy = (img.at<uint8_t>(curr_point) - img.at<uint8_t>(Point(x, y + 1)) + 255) / 2;

			sub_mat.at<uint8_t>(curr_point) =  sqrt(pow(dx, 2) + pow(dy, 2));
		}
	}

	return sub_mat;
}


//double bearing(double a1, double a2, double b1, double b2)
//{
//	static const double TWOPI = M_PI * 2;
//	static const double RAD2DEG = 57.2957795130823209;
//	 if (a1 = b1 and a2 = b2) throw an error 
//	double theta = atan2(b1 - a1, a2 - b2);
//	if (theta < 0.0)
//		theta += TWOPI;
//	return RAD2DEG * theta;
//}

//cv::Mat segment_colorfully(cv::Mat const& image, float sigma, int k, int min)
//{
//	
//	std::cout << "Starting color segmentation..." << std::endl;
//	segm::image<segm::rgb> image_raw(image);
//	int num_ccs;
//
//	segm::image<segm::rgb>* seg = segm::segment_image(&image_raw, sigma, k, min, &num_ccs);
//
//	std::cout << "segmentation: got " << num_ccs << " components" << std::endl;
//
//	cv::Mat image_segm;
//	// filling the mat with the segmented image
//	for (int i = 0; i < seg->width() * seg->height(); i++)
//	{
//		image_segm.push_back(cv::Vec3b(seg->data[i].r, seg->data[i].g, seg->data[i].b));
//	}
//
//	image_segm = image_segm.reshape(0, image.rows);
//
//	//delete seg;
//
//	return image_segm;
//}

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

					
					//cout << (int) cur << endl;
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
			float angle_rad = atan2(dx , dy);

			float angle_grad = ((angle_rad * 180. / M_PI) + 3.2);

			max_angle = max(angle_grad, max_angle);
			min_angle = min(angle_grad, min_angle);

			mat_angle.at<uint8_t>(curr_point) = static_cast<uint8_t>(angle_grad);
			
/*
			vector<uint8_t> arr1;
			for (int i = -1; i <= 1; i++)
			{
				for (int j = -1; j <= 1; j++)
				{
				
					auto cur1 = img.at<uint8_t>(Point(j + x));
					auto cur2 = img.at<uint8_t>(Point(i + y));
					arr1.push_back(cur1);
					arr1.push_back(cur2);
					//cout << (int) cur << endl;
				}
			}
			
			uint8_t const& cnt1 = arr1[4];
			float angle = 0.;
			for (auto const it : arr1)
			{
				angle = atan2(it, mod);
			}
			float angles = ((angle * 180 / M_PI) + 3.2) * 35;
					
					

			mat_angle.at<uint8_t>(curr_point) = 255 - static_cast<uint8_t>(angles);
			
			//auto ch = getchar();
*/

/*			for (int N = -1; N <= 1; N++)
			{
				Point curr_point(x, y);	
								
				uint8_t startPixel = img.at<uint8_t>(x, y);
				uint8_t neighPixel = img.at<uint8_t>(x + N, y + N);
				
				mat_mod.at<uint8_t>(curr_point) = (pow(pow(neighPixel, 2) - pow(startPixel, 2), 1 / 2) + 255) / 2;
				cout << "mod: " << static_cast<uint32_t>(mat_mod.at<uint8_t>(curr_point)) << endl;

				uint8_t angle = atan2(img.at<uint8_t>(Point(x + N)) - img.at<uint8_t>(Point(x)), img.at<uint8_t>(Point(y + N)) - img.at<uint8_t>(Point(y)));
				uint8_t angles = angle * 180 / M_PI;
				uint8_t val = static_cast<uint8_t>((angle + 3.2) * 35);
				
				mat_angle.at<uint8_t>(curr_point) = val;
				cout << "ang: " << static_cast<uint32_t>(val) << endl;
			
			}
*/		}
	}
	

	cout << "angle: " << min_angle << ", " << max_angle << endl;

	return {mat_mod, mat_angle};
}
