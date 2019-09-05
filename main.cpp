#pragma comment(lib,"opencv_world410.lib")
#pragma comment(lib,"libmxnet.lib")


#include "Retinaface.h"
using namespace std;
void Draw(Mat& src, vector<Face> faces, bool use_landmarks, double fontsize = 0.5, int thickness = 1)
{
	int offset = 40.0 * fontsize;
	//cout << "find " << faces.size() << " faces fps " << fps / times << endl;
	for (size_t i = 0; i < faces.size(); i++)
	{
		rectangle(src, faces[i].boundingbox, Scalar(0, 255, 0), 2);
		Point p(faces[i].boundingbox.br().x + 2, faces[i].boundingbox.y + offset);
		putText(src, to_string(faces[i].score), p, FONT_HERSHEY_COMPLEX, fontsize, Scalar(255, 0, 0), thickness);

		if (use_landmarks)
		{
			for (size_t j = 0; j < faces[i].landmarks.size(); j++)
			{
				circle(src, faces[i].landmarks[j], 1, Scalar(255, 0, 0), 2);
			}
		}
	}
}



int main(int argc, char* args[])
{
	bool use_gpu=false;
	RetinaFace m_facedetector(use_gpu);
	m_facedetector.Loadmodel("models", "mnet.25");
	string path="/1.jpg";
	cout << "image_path :" << path << endl;
	src = imread(path);
	vector<Face> faces = m_facedetector.detect(src, score, vector<mx_float> {scale});
	cout << "find " << faces.size() << " faces" << endl;

	Draw(src, faces, use_lankmarks);
	if (width_disp&&height_disp)
	{
		resize(src, src, Size(width_disp, height_disp));
	}
	imshow(path, src);
	
	waitKey(0);
}
