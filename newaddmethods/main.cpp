#include "Pix.h"



void main()
{
	try {
		setlocale(LC_ALL, "Russian");

		/*string file_path = "C:\\Users\\User\\Downloads\\Lenna.png";*/
		/*std::string file_path = "C:\\Users\\User\\Downloads\\Telegram Desktop\\DataSet_V\\DataSet_V\\img0_.png";*/
		std::string file_path = "C:\\Users\\User\\Downloads\\img\\123.png";
		std::string file_path1 = "C:\\Users\\User\\Downloads\\im0_.png";
		
		cv::Mat img = imread(file_path, 0);
		cv::Mat dst = imread(file_path, 0);
		
		if (!img.data) // no image loading;
		{
			throw std::system_error(errno, std::system_category(), file_path);
		}


		//Mat mat_horizontal = calcHGradient(img);
		//Window win_horizontal("Gradiend Field horizontal");
		//win_horizontal.show(mat_horizontal);

		/*auto mat3x3 = calc3x3Gradient(img);
		Window win_mod_3x3("Gradiend Field 3x3 module");
		Mat modules;
		mat3x3.first.convertTo(modules, -1, 0.8, 30.);
		mat3x3.second.convertTo(modules, -1, 0.8, 30.);
		win_mod_3x3.show(modules);
		*/

		//Mat mat_lagrange2 = dilateAndErozia(img, 10, 8) + calc3x3GradientPrevitta(img) + LOGLith(img) + calc3x3GradientPrevitta(img);
		//Window win_lagrange2("lagrange2");
		//win_lagrange2.show(mat_lagrange2);

		cv::Mat mat_experians = dilateAndEroziaProbMatrix(img, 3, 0) + dilateAndEroziaRobinsone(img, 3);
		Window win_experians("Mat");
		win_experians.show(mat_experians);

		cv::Mat mat_MarrHildet = MarrHildeth(img, 0.9)+ dilateAndEroziaProbMatrix(img, 3, 0) + dilateAndEroziaRobinsone(img, 3);
		Window win_MarrHildeth("Mat MarrHildeth(Mat& img, float sigm)");
		win_MarrHildeth.show(mat_MarrHildet);

		cv::Mat mat_MarrHildethNew = MarrHildethNew(img, 0.9) + dilateAndEroziaProbMatrix(img, 3, 0) + dilateAndEroziaRobinsone(img, 3);
		Window win_MarrHildethNew("Mat MarrHildethNew");
		win_MarrHildethNew.show(mat_MarrHildethNew);
	
		cv::Mat mat_newFilterforapplication = NewFilter(img) * 50;
		Window win_newFilterForapplication("This is new Filter LOG Methods for image and Computer vision");
		win_newFilterForapplication.show(mat_newFilterforapplication);

		cv::Mat mat_example = calcHGradient(img) * 0.5 + MarrHildeth(img, 0.9)+ dilateAndEroziaProbMatrix(img, 3, 0) + dilateAndEroziaRobinsone(img, 3) ;
		Window win_exap("Example probably");
		win_exap.show(mat_example);

		//Mat mat_experiansRobinsone = dilateAndEroziaRobinsone(img, 3);
		//Window win_experiansRobinsone("Mat_Robinsone");
		//win_experiansRobinsone.show(mat_experiansRobinsone);
	
		//Mat mat_spilens =  Splines(img);
		//Window win_spl("Spline");
		//win_spl.show(mat_spilens);

		//Mat mat_LOGLith = LOGLith(img) + calc3x3GradientPrevitta(img);
		//Window win_Lith("Mat LOGLith(Mat& img)");
		//win_Lith.show(mat_LOGLith);

		//Mat mat_Catmull = Catmull_Rom(img);
		//Window win_Catmull("Catmull");
		//win_Catmull.show(mat_Catmull);

		//Mat mat_B_spline = B_Spline(img) + EroziaAndDilate(img, 1, 3);
		//Window win_B_spline("B_spline");
		//win_B_spline.show(mat_B_spline);

		//Mat mat_B_dilateBiz = dilateBiz(img, 4);
		//Window win_B_splineBiz("B_spline Biz");
		//win_B_splineBiz.show(mat_B_dilateBiz);

		//Mat mat_Sobela = NewSobol(img);
		//Window win_soNew("newSobel");
		//win_soNew.show(mat_Sobela);

		//Mat mat_Char = NewShar(img);
		//Window win_NewShar("mat_Char");
		//win_NewShar.show(mat_Char);

		//Mat mat_newPrev = NewGradientPrevitta(img);
		//Window win_newPrev("NewPrett");
		//win_newPrev.show(mat_newPrev);
	
		//Mat mat_NewLoG = NewLoG(img);
		//Window win_NewLoG("NewLoG");
		//win_NewLoG.show(mat_NewLoG);

		//Mat mat_Robertsa = calcRobertsa(img);
		//Window win_Robertsa("Robertsa");
		//win_Robertsa.show(mat_Robertsa);

		//Mat mat_LoG = calcLoG(img);
		//Window win_LoG("LoG");
		//win_LoG.show(mat_LoG);

		//Mat mat_DISKRetLOGWegth = calcLoGDiskretWeights(img) ;
		//Window win_DISKRetLOGWegth("DISKRetLOGWegth");
		//win_DISKRetLOGWegth.show(mat_DISKRetLOGWegth);

		//Mat mat_DISKRetLOGWegthProg = calcLoGDiskretWeightsProg(img) ;
		//Window win_DISKRetLOGWegthProg("DISKRetLOGWegthProg");
		//win_DISKRetLOGWegthProg.show(mat_DISKRetLOGWegthProg);

		//Mat mat_sobel = calc3x3GradientSOBOL(img);
		//Window win_sobol("SobolUSED");
		//win_sobol.show(mat_sobel);

		//Mat mat_Pr = calc3x3GradientPrevitta(img);
		//Window win_Previtta("PrevittaUSED");
		//win_Previtta.show(mat_Pr);

		//Mat mat_BinX = dilateX(img);
		//Window win_BinX("BinX");
		//win_BinX.show(mat_BinX);

		//Mat mat_BinY = dilateY(img);
		//Window win_BinY("BinY");
		//win_BinY.show(mat_BinY);

		//Mat mat_SUM =  dilateXY(img);
		//Window win_BinSum("SUMBin");
		//win_BinSum.show(mat_SUM);

		//Mat mat_Bin =BinandDeleteOnlyPixels(img);
		//Window win_Bin("Bin");
		//win_Bin.show(mat_Bin);

		//Mat mat_SUMa = dilateAndErozia(img, 0, 8) ;
		//Window win_BinSuma("10 Dilate and 8 Erozia");
		//win_BinSuma.show(mat_SUMa);
		//
		//Mat mat_erozanddite = EroziaAndDilate(img, 1, 3);
		//Window win_erozdit("FIRST Erozia and Ditale");
		//win_erozdit.show(mat_erozanddite);

		//Mat mat_dilateMXN = Bin(img) - EroziaAndDilate(img,1,0);
		//Window win_dilateMXN("dilateMXN");
		//win_dilateMXN.show(mat_dilateMXN  );
		//
		//Mat mat_dilateMXN1 = dilateAndErozia(img, 3, 1) - Bin(img);
		//Window win_dilateMXN1("dilateMXN1");
		//win_dilateMXN1.show(mat_dilateMXN1);
		//
		//Mat mat_dilateMXN2 = dilateAndErozia(img, 12, 8) - (Bin(img) - EroziaAndDilate(img, 3, 0)) + (dilateAndErozia(img,2,0) - Bin(img));
		//Window win_dilateMXN2("dilateMXN2");
		//win_dilateMXN2.show(mat_dilateMXN2);

		////Mat mat_dila = (img,12,1);
		////Window win_dilateMXN11("dilateMXN11");
		////win_dilateMXN11.show(mat_dilateMXN11);

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

		//Mat mat_SUMa14 = dilateGOA(img, 14);
		//Window win_BinSuma14("SUMBin+ X+Y 14");
		//win_BinSuma14.show(mat_SUMa14);
		Window::wait();

		
	}
	catch (std::exception const& ex)
	{
		std::cout << "fail: " << ex.what() << std::endl;
	}

	system("pause");

}

