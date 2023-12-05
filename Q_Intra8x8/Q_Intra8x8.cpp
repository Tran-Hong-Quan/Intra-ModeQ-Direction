// C++ program for the above approach 
#include <iostream> 
#include <opencv2/opencv.hpp> 
using namespace cv;
using namespace std;

void GenIntra(Mat img);
void GenDecodeIntra(Mat img);

//  Z  A  B  C  D  E  F  G  H  I  J  K  L  M   N  O  P
//  Q  a1 b1 c1 d1 e1 f1 g1 h1
//  R  a2 b2 c2 d2 e2 f2 g2 h2
//  S  a3 b3 c3 d3 e3 f3 g3 h3
//  T  a4 b4 c4 d4 e4 f4 g4 h4
//  U  a5 b5 c5 d5 e5 f5 g5 h5
//  V  a6 b6 c6 d6 e6 f6 g6 h6
//  W  a7 b7 c7 d7 e7 f7 g7 h7
//  X  a8 b8 c8 d8 e8 f8 g8 h8

#define P_Z prePixel[0]

// Driver code 
int main(int argc, char** argv)
{
	// Read the image file as 
	// imread("default.jpg"); 
	Mat image = imread("myImg.png");

	// Error Handling 
	if (image.empty()) {
		cout << "Image File "
			<< "Not Found" << endl;

		// wait for any key press 
		cin.get();
		return -1;
	}

	// Show Image inside a window with 
	// the name provided 
	imshow("Original image", image);
	GenIntra(image);

	// Wait for any keystroke 
	waitKey(0);
	return 0;
}

void GetIntraBlock(Mat src, Vec3b prePixle[], Mat& difBlock, Mat& copyBlock) {
	//Start from D, end at K
	int prePixelIndex = 4;

	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			copyBlock.at<Vec3b>(j, i) = prePixle[prePixelIndex];
			difBlock.at<Vec3b>(j, i) = prePixle[prePixelIndex] - src.at<Vec3b>(j, i);
		}
		prePixelIndex++;
	}
}

void GenIntra(Mat img) {

	int  rows = img.rows;
	int  cols = img.cols;
	auto type = img.type();

	Mat dif = cv::Mat(rows, cols, type);
	Mat copy = cv::Mat(rows, cols, type);

	Vec3b prePixel[25];
	//Get Z
	prePixel[0] = img.at<Vec3b>(0, 0);

	for (int x = 1; x < rows + 8; x += 8)
	{
		// Get Q R S T U V W X
		for (int i = 17, t = 0; i < 25; i++)
		{
			if (x + t >= rows)
				prePixel[i] = Vec3b(0, 0, 0);
			else
				prePixel[i] = img.at<Vec3b>(x + t, 0);
			t++;
		}
		for (int y = 1; y < cols + 8; y += 8)
		{
			// Get  A  B  C  D  E  F  G  H  I  J  K  L  M   N  O  P
			for (int i = 1, t = 0; i < 17; i++)
			{
				if (y + t >= cols)
					prePixel[i] = Vec3b(0, 0, 0);
				else
					prePixel[i] = img.at<Vec3b>(0, y + t);
				t++;
			}

			Mat block = cv::Mat(8, 8, type);
			Mat copyBlock = cv::Mat(8, 8, type);
			Mat difBlock = cv::Mat(8, 8, type);
			Mat scr = cv::Mat(8, 8, type);

			//Get source block
			for (int i = 0; i < 8; i++)
			{
				for (int j = 0; j < 8; j++)
				{
					if (x + i >= rows || y + j >= cols)
						scr.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
					else
						scr.at<Vec3b>(i, j) = img.at<Vec3b>(x + i, y + j);
				}
			}

			GetIntraBlock(scr, prePixel, difBlock, copyBlock);

			//Paste to dif and copy image from block
			for (int i = x, a = 0; i < x + 8; i++)
			{
				for (int j = y, b = 0; j < y + 8; j++)
				{
					if (i < rows && j < cols)
					{
						copy.at<Vec3b>(i, j) = copyBlock.at<Vec3b>(a, b);
						dif.at<Vec3b>(i, j) = difBlock.at<Vec3b>(a, b);
					}
					b++;
				}
				a++;
			}
		}
	}

	//Copy border from source img to dif img
	for (int i = 0; i < cols; i++)
	{
		dif.at<Vec3b>(0, i) = img.at<Vec3b>(0, i);
	}
	for (int i = 0; i < rows; i++)
	{
		dif.at<Vec3b>(i, 0) = img.at<Vec3b>(i, 0);
	}

	imshow("Dif image", dif);
	imshow("Copy image", copy);

	GenDecodeIntra(dif);
}

void GetDecodeIntraBlock(Mat& res, Mat encodeBlock, Vec3b prePixle[]) {
	//Start from D, end at K
	int prePixelIndex = 4;

	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			res.at<Vec3b>(j, i) = prePixle[prePixelIndex] - encodeBlock.at<Vec3b>(j, i);
		}
		prePixelIndex++;
	}
}

void GenDecodeIntra(Mat img) {
	int  rows = img.rows;
	int  cols = img.cols;
	auto type = img.type();

	Mat res = cv::Mat(rows, cols, type);

	Vec3b prePixel[25];
	//Get Z
	prePixel[0] = img.at<Vec3b>(0, 0);

	for (int x = 1; x < rows + 8; x += 8)
	{
		// Get Q R S T U V W X
		for (int i = 17, t = 0; i < 25; i++)
		{
			if (x + t >= rows)
				prePixel[i] = Vec3b(0, 0, 0);
			else
				prePixel[i] = img.at<Vec3b>(x + t, 0);
			t++;
		}
		for (int y = 1; y < cols + 8; y += 8)
		{
			// Get  A  B  C  D  E  F  G  H  I  J  K  L  M   N  O  P
			for (int i = 1, t = 0; i < 17; i++)
			{
				if (y + t >= cols)
					prePixel[i] = Vec3b(0, 0, 0);
				else
					prePixel[i] = img.at<Vec3b>(0, y + t);
				t++;
			}

			Mat resBlock = cv::Mat(8, 8, type);
			Mat scr = cv::Mat(8, 8, type);

			//Get source block
			for (int i = 0; i < 8; i++)
			{
				for (int j = 0; j < 8; j++)
				{
					if (x + i >= rows || y + j >= cols)
						scr.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
					else
						scr.at<Vec3b>(i, j) = img.at<Vec3b>(x + i, y + j);
				}
			}

			GetDecodeIntraBlock(resBlock, scr, prePixel);

			//Paste to res from res block
			for (int i = x, a = 0; i < x + 8; i++)
			{
				for (int j = y, b = 0; j < y + 8; j++)
				{
					if (i < rows && j < cols)
					{
						res.at<Vec3b>(i, j) = resBlock.at<Vec3b>(a, b);
					}
					b++;
				}
				a++;
			}
		}
	}

	imshow("Decode img", res);
}