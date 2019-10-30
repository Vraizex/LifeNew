#include "Pix.h"
#include <iostream>
#include <random>
#include <time.h>

RNG rng(12345);

Mat sobel = (Mat_<float>(3, 3) << -1 / 16., -2 / 16., -1 / 16., 0, 0, 0, 1 / 16., 2 / 16., 1 / 16.);



float dist(Point p1, Point p2, Vec3f p1_lab, Vec3f p2_lab, float compactness, float S);


//double sqr(double value) 
//{
//	return value * value;
//}
//
//double sqr_distance(Point first, Point second) 
//{
//	return sqr(first.x - second.x) + sqr(first.y - second.y);
//}



//Histograma
void showHistogram(const string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}


int main()
{
	try {
		setlocale(LC_ALL, "Russian");

		/*string file_path = "C:\\Users\\User\\Downloads\\Lenna.png";*/
		/*std::string file_path = "C:\\Users\\User\\Downloads\\Telegram Desktop\\DataSet_V\\DataSet_V\\img0_.png";*/
		std::string file_path = "C:\\Users\\User\\Downloads\\im0_.png";
		std::string file_path1 = "C:\\Users\\User\\Downloads\\im0_.png";
		cv::Mat labels;
		cv::Mat img = imread(file_path, 1);
		cv::Mat dst = imread(file_path, 1);
		cv::Mat_<Vec3b> src = imread(file_path, 1);
		
		if (!img.data) // no image loading;
		{
			throw std::system_error(errno, std::system_category(), file_path);
		}

		// K_MEANS Good Workkks
		//k_means(src);

		//kMeans(src);
		//kMeans(src);


		//Mat samples(src.rows * src.cols, 3, CV_32F);
		//for (int y = 0; y < src.rows; y++)
		//	for (int x = 0; x < src.cols; x++)
		//		for (int z = 0; z < 3; z++)
		//			samples.at<float>(y + x * src.rows, z) = src.at<Vec3b>(y, x)[z];


		//int clusterCount = 3;
		//Mat labels1;
		//int attempts = 2;
		//Mat centers;
		//kmeans(samples, clusterCount, labels1, TermCriteria(cv::TermCriteria::MAX_ITER +	cv::TermCriteria::EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);


		//Mat new_image(src.size(), src.type());
		//for (int y = 0; y < src.rows; y++)
		//	for (int x = 0; x < src.cols; x++)
		//	{
		//		int cluster_idx = labels1.at<int>(y + x * src.rows, 0);
		//		new_image.at<Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
		//		new_image.at<Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
		//		new_image.at<Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
		//	}
		//imshow("clustered image", new_image);
		//waitKey(0);
		
		/*cv::Mat mat_mod(img.size(), CV_8UC1, Scalar(0));

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

				}
			}*/

		//TODO Algorithm of SuperPixel
		//SLIC methods
		//Edit Filter2D and Magnitude Split Function
		int nx, ny;
		int m;

		// Default values
		nx = 80;
		ny = 80;
		m = 20;


		// Scale to [0,1] and l*a*b colorspace
		img.convertTo(img, CV_32F, 1 / 255.);
		Mat imlab;
		cvtColor(img, imlab, COLOR_BGR2Lab);

		int h = img.rows;
		int w = img.cols;
		int n = nx * ny;

		float dx = w / float(nx);
		float dy = h / float(ny);
		int S = (dx + dy + 1) / 2; // window width

		// Initialize centers
		vector<Point> centers;
		for (int i = 0; i < ny; i++) {
			for (int j = 0; j < nx; j++) {
				centers.push_back(Point(j*dx + dx / 2, i*dy + dy / 2));
			}
		}

		// Initialize labels and distance maps
		vector<int> label_vec(n);
		for (int i = 0; i < n; i++)
			label_vec[i] = i * 255 * 255 / n;

		Mat labelsX = -1 * Mat::ones(imlab.size(), CV_32S);
		Mat dists = -1 * Mat::ones(imlab.size(), CV_32F);
		Mat window;
		Point p1, p2;
		Vec3f p1_lab, p2_lab;

		// Iterate 10 times. In practice more than enough to converge
		for (int i = 0; i < 10; i++) {
			// For each center...
			for (int c = 0; c < n; c++)
			{
				int label = label_vec[c];
				p1 = centers[c];
				p1_lab = imlab.at<Vec3f>(p1);
				int xmin = max(p1.x - S, 0);
				int ymin = max(p1.y - S, 0);
				int xmax = min(p1.x + S, w - 1);
				int ymax = min(p1.y + S, h - 1);

				// Search in a window around the center
				window = img(Range(ymin, ymax), Range(xmin, xmax));

				// Reassign pixels to nearest center
				for (int i = 0; i < window.rows; i++) {
					for (int j = 0; j < window.cols; j++) {
						p2 = Point(xmin + j, ymin + i);
						p2_lab = imlab.at<Vec3f>(p2);
						float d = dist(p1, p2, p1_lab, p2_lab, m, S);
						float last_d = dists.at<float>(p2);
						if (d < last_d || last_d == -1) {
							dists.at<float>(p2) = d;
							labelsX.at<int>(p2) = label;
						}
					}
				}
			}
		}

		// Calculate superpixel boundaries
		labelsX.convertTo(labelsX, CV_32F);
		Mat gx, gy, grad;

		/** @brief Создает изображение с ядром.

			Функция применяет произвольный линейный фильтр к изображению.Операция на месте поддерживается.когда
			апертура частично находится вне изображения, функция интерполирует значения пикселей посторонних
			в соответствии с указанным режимом границы.

			Функция действительно вычисляет корреляцию, а не свертку :

		f[\ texttt{ dst } (x, y) = \ sum _{ \ stackrel {0 \ leq x '<\ texttt {kernel.cols},} {0 \ leq y' < \ texttt {kernel.rows} } } \ texttt{ kernel } (x ', y') * \ texttt{ src } (x + x'- \ texttt {anchor.x}, y + y' - \ texttt{ anchor.y }) \ f]

			То есть ядро ​​не отражается вокруг точки привязки.Если вам нужна настоящая свертка, переверните
			ядро с помощью #flip и установите новый якорь на `(kernel.cols - anchor.x - 1, kernel.rows -
			anchor.y - 1) `.

			Функция использует алгоритм на основе DFT в случае достаточно больших ядер(~`11 x 11` или
				больше) и прямой алгоритм для небольших ядер.

			@param src входное изображение.
			@param dst выводит изображение того же размера и того же количества каналов, что и src.
			@param ddepth желаемая глубина целевого изображения, см. @ref filter_depths "комбинации"
			ядро свертки @param (или, скорее, корреляционное ядро), одноканальная плавающая точка
			матрица; если вы хотите применить разные ядра к разным каналам, разделите изображение на
			Разделяйте цветовые плоскости с помощью разделения и обрабатывайте их индивидуально.
			@param anchor якорь ядра, который указывает относительную позицию отфильтрованной точки в
			ядро; якорь должен находиться внутри ядра; значение по умолчанию(-1, -1) означает, что якорь
			находится в центре ядра.
			@param delta необязательное значение, добавляемое к отфильтрованным пикселям перед сохранением их в dst.
			@param borderType метод экстраполяции пикселей, см. #BorderTypes
			@sa sepFilter2D, dft, matchTemplate
			*/
		filter2D(labelsX, gx, -1, sobel); // operator sobel DX
		filter2D(labelsX, gy, -1, sobel.t()); // operator sobel Dy
		magnitude(gx, gy, grad);
		grad = (grad > 1e-4) / 255;
		Mat show = 1 - grad;
		show.convertTo(show, CV_32F);

		// Draw boundaries on original image
		vector<Mat> rgb(3);
		split(img, rgb);
		for (int i = 0; i < 3; i++)
			rgb[i] = rgb[i].mul(show);

		merge(rgb, img);

		imshow("EndImage",img);

		//auto mat_text = calc3x3Gradient(img);
		//Window win_text("Texture");
		//win_text.show(mat_text.first);

		//Mat mat_H = calcHGradient(img);
		//Window win_H("summa H Gradient");
		//win_H.show(mat_H);
		//
		//Mat mat_V = calcVGradient(img) ;
		//Window win_V("summa V Gradient");
		//win_V.show(mat_V);

		//Mat mat_weigths = mat_text.first + 0.4 * mat_H;
		//Window win_X("Win_X");
		//win_X.show(mat_weigths);

		//Mat mat_wei = mat_text.first + 0.4 * mat_V;
		//Window win_Y("Win_Y");
		//win_Y.show(mat_wei);

		//
		//Mat mat_all1 = mat_weigths + calcSOBOL(img);
		//Window win_all1("Win_all X");
		//win_all1.show(mat_all1);

		//Mat mat_all2 = mat_wei + calcSOBOL(img);
		//Window win_all2("Win_all Y");
		//win_all2.show(mat_all2);


		//Mat mat_final = mat_all1 + mat_all2;
		//Window win_final("SuperPix:");
		//win_final.show(mat_final);

		//Mat new_sobel = MatrixGrad(dst, 1);
		//Window win_sobel("Sobel ABS");
		//win_sobel.show(new_sobel);

		//Mat new_sobel1 = MatrixGrad(dst, 2);
		//Window win_sobel1("Sobel ABS2");
		//win_sobel1.show(new_sobel1);

		//Mat new_sobel2 = MatrixGrad(dst, 5);
		//Window win_sobel2("Sobel ABS3");
		//win_sobel2.show(new_sobel2);

		//Mat new_sobel3 = MatrixGrad(dst, 7);
		//Window win_sobel3("Sobel ABS4");
		//win_sobel3.show(new_sobel3);

		//Mat new_sobel4 = MatrixGrad(dst, 10);
		//Window win_sobel4("Sobel ABS5");
		//win_sobel4.show(new_sobel4);



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

	//cv::TermCriteria::MAX_ITER +
	//	cv::TermCriteria::EPS

		//Mat src = imread("C:\\Users\\User\\Downloads\\im0_.png", 1);
		//Mat samples(src.rows * src.cols, 3, CV_32F);
		//for (int y = 0; y < src.rows; y++)
		//	for (int x = 0; x < src.cols; x++)
		//		for (int z = 0; z < 3; z++)
		//			samples.at<float>(y + x * src.rows, z) = src.at<Vec3b>(y, x)[z];


		//int clusterCount = 5;
		//Mat labels;
		//int attempts = 2;
		//Mat centers;
		//kmeans(samples, clusterCount, labels, TermCriteria(cv::TermCriteria::MAX_ITER +	cv::TermCriteria::EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);


		//Mat new_image(src.size(), src.type());
		//for (int y = 0; y < src.rows; y++)
		//	for (int x = 0; x < src.cols; x++)
		//	{
		//		int cluster_idx = labels.at<int>(y + x * src.rows, 0);
		//		new_image.at<Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
		//		new_image.at<Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
		//		new_image.at<Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
		//	}
		//imshow("clustered image", new_image);
		//waitKey(0);
		
		//cv::Mat mat_hocj = calcSOBOL(img) + 
		//	(dilateAndErozia(img, 1, 0) - dilateAndErozia(img, 0, 1)) +
		//	NewFilter(img) +
		//	MarrHildrethNew(img, 0.9) +
		//	NewGradientPrevitta(img) +
		//	calcHGradient(img) +
		//	MarrHildeth(img, 0.9);
		//Window win_hocj("New methods Used Sobol and Moprphology gradient");
		//win_hocj.show(mat_hocj);

		//cv::Mat blank(mat_hocj.size(), CV_8U, cv::Scalar(0xFF));
		//cv::Mat dest;
		//imshow("originalimage", mat_hocj);

		// Create markers image
		//cv::Mat markers(mat_hocj.size(), CV_8U, cv::Scalar(-1));
		////Rect(topleftcornerX, topleftcornerY, width, height);
		////top rectangle
		//markers(Rect(0, 0, mat_hocj.cols, 5)) = Scalar::all(1);
		////bottom rectangle
		//markers(Rect(0, mat_hocj.rows - 5, mat_hocj.cols, 5)) = Scalar::all(1);
		////left rectangle
		//markers(Rect(0, 0, 5, mat_hocj.rows)) = Scalar::all(1);
		////right rectangle
		//markers(Rect(mat_hocj.cols - 5, 0, 5, mat_hocj.rows)) = Scalar::all(1);
		////centre rectangle
		//int centreW = mat_hocj.cols / 4;
		//int centreH = mat_hocj.rows / 4;
		//markers(Rect((mat_hocj.cols / 2) - (centreW / 2), (mat_hocj.rows / 2) - (centreH / 2), centreW, centreH)) = Scalar::all(2);
		//markers.convertTo(markers, cv::COLOR_BGR2GRAY);
		//imshow("markers", markers);

		////Create watershed segmentation object
		//WatershedSegmenter segmenter;
		//segmenter.setMarkers(markers);
		//cv::Mat wshedMask = segmenter.process(img);
		//cv::Mat mask;
		//convertScaleAbs(wshedMask, mask, 1, 0);
		//double thresh = threshold(mask, mask, 1, 255, THRESH_BINARY);
		//bitwise_and(mat_hocj, mat_hocj, dest, mask);
		//dest.convertTo(dest, CV_8U);

		//imshow("final_result", dest);
		/*cv::waitKey(0);*/





		//Mat mat_lagrange2 = dilateAndErozia(img, 10, 8) + calc3x3GradientPrevitta(img) + LOGLith(img) + calc3x3GradientPrevitta(img);
		//Window win_lagrange2("lagrange2");
		//win_lagrange2.show(mat_lagrange2);

		//cv::Mat mat_experians = calcKircsha(img, 3, 0) + calcRobinsone(img, 3);
		//Window win_experians("Mat");
		//win_experians.show(mat_experians);

		//cv::Mat mat_MarrHildet = MarrHildeth(img, 0.9)+ calcKircsha(img, 3, 0) + calcRobinsone(img, 3);
		//Window win_MarrHildeth("Mat MarrHildeth(Mat& img, float sigm)");
		//win_MarrHildeth.show(mat_MarrHildet);

		//cv::Mat mat_MarrHildethNew = MarrHildrethNew(img, 0.9) + calcKircsha(img, 3, 0) + calcRobinsone(img, 3);
		//Window win_MarrHildethNew("Mat MarrHildethNew");
		//win_MarrHildethNew.show(mat_MarrHildethNew);
	
		//cv::Mat mat_newFilterforapplication = NewFilter(img) * 50;
		//Window win_newFilterForapplication("This is new Filter LOG Methods for image and Computer vision");
		//win_newFilterForapplication.show(mat_newFilterforapplication);

		//auto mat_example = calc5x5Gradient(img);
		//Window win_exap("Example probably");
		//win_exap.show(mat_example.first);
		//Window win_exap2("Example probably 2");
		//win_exap2.show(mat_example.second);

		//cv::Mat mat_new_Format = calcHGradient(img) - calcKircsha(img, 12, 0) +
		//MarrHildrethNew(img, 0.7) + calc3x3GradientSOBOLBinAndMatrix(img) +
		//calcPrevitta(img) + calcRobertsa(img) + calcRobinsone(img,4) +
		//dilateAndErozia(img,10,8) + NewFilter(img);
		//Window win_prob("Make example for used all filters");
		//win_prob.show(mat_new_Format);

		//cv::Mat mat_made_privat = calcPrevitta(img) + (dilateAndErozia(img, 1, 0) - dilateAndErozia(img, 0, 1));
		//Window win_prirt("Privatrte");
		//win_prirt.show(mat_made_privat);

		//cv::Mat mat_status = calcVGradient(img);
		//Window win_stat("VGradient");
		//win_stat.show(mat_status);

		//cv::Mat mat_gradientMprph = dilateAndErozia(img, 1, 0) - dilateAndErozia(img, 0, 1);
		//Window win_grad("Morphology gradient Close mines Open");
		//win_grad.show(mat_gradientMprph);

		//cv:Mat mat_news = calcRobinsone(img, 10) + (dilateAndErozia(img, 1, 0) - dilateAndErozia(img, 0, 1));
		//Window win_news("Analysis used Opertor Robinsone and Morphology Gradient");
		//win_news.show(mat_news);

		//cv::Mat mat_hocj = calcSOBOL(img) + 
		//	(dilateAndErozia(img, 1, 0) - dilateAndErozia(img, 0, 1)) +
		//	NewFilter(img) +
		//	MarrHildrethNew(img, 0.9) +
		//	NewGradientPrevitta(img) +
		//	calcHGradient(img) +
		//	MarrHildeth(img, 0.9);
		//Window win_hocj("New methods Used Sobol and Moprphology gradient");
		//win_hocj.show(mat_hocj);

		//cv::Mat mat_great = calcSOBOL(img) +
		//	(dilateAndErozia(img, 1, 0) - dilateAndErozia(img, 0, 1)) +
		//	NewFilter(img) +
		//	MarrHildrethNew(img, 0.9) +
		//	NewGradientPrevitta(img) +
		//	calcHGradient(img) +
		//	MarrHildeth(img, 0.9) +
		//	calcKircsha(img, 5, 0) +
		//	dilateEroziaLevel(img, 3);
		//Window win_great("New Window");
		//win_great.show(mat_great);

		//cv::Mat mat_newfil = newfil(img);
		//	Window win_newfil("newfil");
		//	win_newfil.show(mat_newfil);
		





		//Mat mat_experiansRobinsone = calcRobinsone(img, 3);
		//Window win_experiansRobinsone("Mat_Robinsone");
		//win_experiansRobinsone.show(mat_experiansRobinsone);
	
		//Mat mat_spilens =  Splines(img);
		//Window win_spl("Spline");
		//win_spl.show(mat_spilens);

		//Mat mat_LOGLith = LOGLith(img) + calcPrevitta(img);
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

		//Mat mat_sobel = calcSOBOL(img);
		//Window win_sobol("SobolUSED");
		//win_sobol.show(mat_sobel);

		//Mat mat_Pr = calcPrevitta(img);
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

		//Mat mat_SUMa = dilateAndErozia(img, 12, 8) ;
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


		//Mat mat_H = calcHough(img);
		//Window win("Hough");
		//win.show(mat_H);


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

cv::Mat methodsSobol(Mat&img)
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

			float dx = (abs(img.at<uint8_t>(Point(x + 1, y))) - abs(img.at<uint8_t>(Point(x - 1, y)))) / 2;
			float dy = (abs(img.at<uint8_t>(Point(x, y + 1))) - abs(img.at<uint8_t>(Point(x, y - 1)))) / 2;

			float sobol = (sqrt(pow(dx, 2) + pow(dy, 2))) - 250;


			mat_mod.at<uint8_t>(curr_point) = static_cast<uint8_t>(sobol - 6) * 10;

		}
	}

	return  mat_mod;

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

	float grad = sqrtf((dx*dx) + (dy*dy));
}