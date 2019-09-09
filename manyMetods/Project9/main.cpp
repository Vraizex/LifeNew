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

		

		//Mat mat_horizontal = calcHGradient(img);
		//Window win_horizontal("Gradiend Field horizontal");
		//win_horizontal.show(mat_horizontal);

		//auto mat3x3 = calc3x3Gradient(img);
		//Window win_mod_3x3("Gradiend Field 3x3 module");
		//Mat modules;
		//mat3x3.first.convertTo(modules, -1, 0.8, 30.);
		//win_mod_3x3.show(modules);


		//Window win_angle_3x3("Gradiend Field 3x3 angle");
		//win_mod_3x3.show(modules);
		//win_angle_3x3.show(mat3x3.second);

		//Window foobar("Foobar");
		//foobar.show(mat3x3.first / 2 + mat3x3.second / 2);
		//Window::wait();
		
	
	/*	Mat mat_Robertsa = calcRobertsa(img);
		Window win_Robertsa("Robertsa");
		win_Robertsa.show(mat_Robertsa);*/

		//Mat mat_LoG = calcLoG(img);
		//Window win_LoG("LoG");
		//win_LoG.show(mat_LoG);
	
		Mat mat_DISKRetLOG = calcLoGDiskret(img);	
		Window win_DISKRetLOG("DISKRetLOG");
		win_DISKRetLOG.show(mat_DISKRetLOG);


		//Mat mat_DISKRetLOGWegth = calcLoGDiskretWeights(img);
		//Window win_DISKRetLOGWegth("DISKRetLOGWegth");
		//win_DISKRetLOGWegth.show(mat_DISKRetLOGWegth);

		//Mat mat_DISKRetLOGWegthProg = calcLoGDiskretWeightsProg(img);
		//Window win_DISKRetLOGWegthProg("DISKRetLOGWegthProg");
		//win_DISKRetLOGWegthProg.show(mat_DISKRetLOGWegthProg);

		Mat mat_sobel = calc3x3GradientSOBOL(img);
		Window win_sobol("Sobol");
		win_sobol.show(mat_sobel);


	/*	Mat mat_Pr = calc3x3GradientPrevitta(img);
		Window win_Previtta("Previtta");
		win_Previtta.show(mat_Pr);
*/

		//Mat mat_wave = cvHaarWavelet(img, dst, 30);
		//Window win_wave("Veilvet");
		//win_wave.show(mat_wave);
		
		//
		//Mat imgPanel(100, 250, CV_8UC1, Scalar(0));


		//Mat imgPanelRoi(img, Rect(0, 0, 50, 50));
		//img.copyTo(imgPanelRoi);

		//imshow("imgPanel", img);
		//waitKey();

		//Mat mat_wave_inv = cvInvHaarWavelet(img, dst, 10, 1, 10);
		//Window win_wave_inv("InVeilvet");
		//win_wave_inv.show(mat_wave_inv);
		

		Window::wait();

		
	}
	catch (std::exception const& ex)
	{
		std::cout << "fail: " << ex.what() << std::endl;
	}

	system("pause");

}

