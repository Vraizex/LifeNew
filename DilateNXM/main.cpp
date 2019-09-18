#include "Pix.h"



void main()
{
	try {
		setlocale(LC_ALL, "Russian");

		/*string file_path = "C:\\Users\\User\\Downloads\\Lenna.png";*/
		/*std::string file_path = "C:\\Users\\User\\Downloads\\Telegram Desktop\\DataSet_V\\DataSet_V\\img0_.png";*/
		std::string file_path = "C:\\Users\\User\\Downloads\\im0_.png";
		std::string file_path1 = "C:\\Users\\User\\Downloads\\im0_.png";
		
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



		
	
		Mat mat_Robertsa = calcRobertsa(img);
		Window win_Robertsa("Robertsa");
		win_Robertsa.show(mat_Robertsa);

		Mat mat_LoG = calcLoG(img);
		Window win_LoG("LoG");
		win_LoG.show(mat_LoG);
	


		Mat mat_DISKRetLOGWegth = calcLoGDiskretWeights(img) ;
		Window win_DISKRetLOGWegth("DISKRetLOGWegth");
		win_DISKRetLOGWegth.show(mat_DISKRetLOGWegth);

		Mat mat_DISKRetLOGWegthProg = calcLoGDiskretWeightsProg(img) ;
		Window win_DISKRetLOGWegthProg("DISKRetLOGWegthProg");
		win_DISKRetLOGWegthProg.show(mat_DISKRetLOGWegthProg);

		Mat mat_sobel = calc3x3GradientSOBOL(img);
		Window win_sobol("Sobol");
		win_sobol.show(mat_sobel);


		Mat mat_Pr = calc3x3GradientPrevitta(img);
		Window win_Previtta("Previtta");
		win_Previtta.show(mat_Pr);



		Mat mat_BinX = dilateX(img);
		Window win_BinX("BinX");
		win_BinX.show(mat_BinX);



		Mat mat_BinY = dilateY(img);
		Window win_BinY("BinY");
		win_BinY.show(mat_BinY);


		Mat mat_SUM =  dilateXY(img);
		Window win_BinSum("SUMBin");
		win_BinSum.show(mat_SUM);


		Mat mat_SUMa = dilateAndErozia(img, 10, 8) ;
		Window win_BinSuma("10 Dilate and 8 Erozia");
		win_BinSuma.show(mat_SUMa);
		
		Mat mat_erozanddite = EroziaAndDilate(img, 1, 15);
		Window win_erozdit("FIRST Erozia and Ditale");
		win_erozdit.show(mat_erozanddite);

		Mat mat_dilateMXN = dilateMXN(img, 10, 0) - dilateMXN(img, 8,2);
		Window win_dilateMXN("dilateMXN");
		win_dilateMXN.show(mat_dilateMXN  );
		

	/*	Mat mat_dila = (img,12,1);
		Window win_dilateMXN11("dilateMXN11");
		win_dilateMXN11.show(mat_dilateMXN11);*/

		//Mat mat_SUMa2 = dilateGOA(img, 1) ;
		//Window win_BinSuma2("SUMBin+ X+Y2");
		//win_BinSuma2.show(mat_SUMa2);

		//Mat mat_SUMa3 = dilateGOA(img, 2);
		//Window win_BinSuma3("SUMBin+ X+Y3");
		//win_BinSuma3.show(mat_SUMa3);


		//Mat mat_SUMa4 =  dilateGOA(img,5);
		//Window win_BinSuma4("SUMBin+ X+Y4");
		//win_BinSuma4.show(mat_SUMa4);


		//Mat mat_SUMa5 = dilateGOA(img, 10);
		//Window win_BinSuma5("SUMBin+ X+Y 5");
		//win_BinSuma5.show(mat_SUMa5);


	

		Window::wait();

		
	}
	catch (std::exception const& ex)
	{
		std::cout << "fail: " << ex.what() << std::endl;
	}

	system("pause");

}

