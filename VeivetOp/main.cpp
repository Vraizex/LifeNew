#include "Pix.h"


void main()
{
	try {
		setlocale(LC_ALL, "Russian");

		/*string file_path = "C:\\Users\\User\\Downloads\\Lenna.png";*/
		/*std::string file_path = "C:\\Users\\User\\Downloads\\Telegram Desktop\\DataSet_V\\DataSet_V\\img0_.png";*/
		std::string file_path = "C:\\Users\\User\\Downloads\\im0_.png";
		Mat img = imread(file_path, 0);
		Mat dst = imread(file_path, 0);
		if (!img.data) // no image loading;
		{
			throw std::system_error(errno, std::system_category(), file_path);
		}

		Mat mat_wave = cvHaarWavelet(img, dst ,1000);
		Window win_wave("Veilvet");
		win_wave.show(mat_wave);


		Mat mat_wave_inv = cvInvHaarWavelet(img, dst, 10,  0, 20);
		Window win_wave_inv("InVeilvet");
		win_wave_inv.show(mat_wave_inv);
		
		//Mat mat_wave_inv1 = cvInvHaarWavelet(img, dst, 10, 1, 20);
		//Window win_wave_inv1("InVeilvet1");
		//win_wave_inv1.show(mat_wave_inv1);

		//Mat mat_wave_inv2 = cvInvHaarWavelet(img, dst, 10, 2, 20);
		//Window win_wave_inv2("InVeilvet2");
		//win_wave_inv2.show(mat_wave_inv2);
		//
		//Mat mat_wave_inv3 = cvInvHaarWavelet(img, dst, 10, 3, 20);
		//Window win_wave_inv3("InVeilvet3");
		//win_wave_inv3.show(mat_wave_inv3);

		/*Mat mat_horizontal = calcHGradient(img);
		Window win_horizontal("Gradiend Field horizontal");
		win_horizontal.show(mat_horizontal);

		auto mat3x3 = calc3x3Gradient(img);
		
		Window win_mod_3x3("Gradiend Field 3x3 module");
		Window win_angle_3x3("Gradiend Field 3x3 angle");
		
		win_angle_3x3.show(mat3x3.first);
		win_angle_3x3.show(mat3x3.second);*/

		
		/*cvHaarWavelet(Mat &img, Mat &sub_mat, int NIter)*/



		Window::wait();




		
	}
	catch (std::exception const& ex)
	{
		std::cout << "fail: " << ex.what() << std::endl;
	}

	system("pause");

}

