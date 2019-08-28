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
	if (argc <= 1)
	{
		cout << "\\\> retinaface.exe " << endl;
		cout << "argv:" << endl;
		cout << "             -v/-i video/image path           (type:string,-v videopath(0?webcam:videofile),-i imagepath,default=0)" << endl;
		cout << "             -s    scale                      (type:float,default=1.0)" << endl;
		cout << "             -t    score_threshold            (type:float,default=0.8)" << endl;
		cout << "             -l    use_landmarks              (type:int,default=1)" << endl;
		cout << "             -g    use_gpu                    (type:int,default=1)" << endl;
		cout << "             -h         resize height         (type:int,default=src.rows)" << endl;
		cout << "             -w         resize width          (type:int,default=src.cols)" << endl;
		cout << "Example:" << endl << endl;
		cout << ">>>>>>>>>>>>> detect in image <<<<<<<<<<<<<" << endl;
		cout << "\\\> .\\retinaface.exe -i .\\people.jpeg -s 1.0 -t 0.8 -l 1 -g 1" << endl << endl;
		cout << ">>>>>>>>>>>>> detect in video <<<<<<<<<<<<<" << endl;
		cout << "\\\> .\\retinaface.exe -v .\\multiple_faces.avi -s 0.5 -t 0.8 -l 0 -g 1" << endl << endl;
		cout << ">>>>>>>>>>>>> detect in webcam <<<<<<<<<<<<<" << endl;
		cout << "\\\> .\\retinaface.exe -v 0 -s 0.5 -t 0.8 -l 1 -g 1" << endl << endl << endl;
		return 0;
	}
	//dlog(argc);
	bool isvideo = true;
	bool use_lankmarks = 1;
	bool use_gpu = 1;
	string path = "0";
	mx_float scale = 1.0;
	mx_float score = 0.8;
	int height_disp = 0;
	int width_disp = 0;
	for (size_t i = 1; i < argc; i++)
	{
		if (string(args[i]) == "-v")
		{
			isvideo = true;
			path = args[i + 1];
		}
		else if (string(args[i]) == "-i")
		{
			isvideo = false;
			path = args[i + 1];
		}
		else if (string(args[i]) == "-s")
		{
			scale = atof(args[i + 1]);
		}
		else if (string(args[i]) == "-t")
		{
			score = atof(args[i + 1]);
		}
		else if (string(args[i]) == "-l")
		{
			use_lankmarks = atoi(args[i + 1]);
		}
		else if (string(args[i]) == "-g")
		{
			use_gpu = atoi(args[i + 1]);
		}
		else if (string(args[i]) == "-h")
		{
			height_disp = atoi(args[i + 1]);
		}
		else if (string(args[i]) == "-w")
		{
			width_disp = atoi(args[i + 1]);
		}
	}
	cout << "scale :" << scale << endl;
	cout << "score :" << score << endl;
	cout << "use_landmarks? :" << use_lankmarks << endl;
	cout << "use_gpu? :" << use_gpu << endl;
	cout << "width :" << width_disp << endl;
	cout << "height :" << height_disp << endl;
	RetinaFace m_facedetector(use_gpu);
	m_facedetector.Loadmodel("E:/PyCode/insightface/models", "mnet.25");
	m_facedetector.use_landmarks = use_lankmarks;

	Mat src;

	if (isvideo)
	{
		cout << "video_path :" << path << endl;
		VideoCapture *video = nullptr;
		if (path == "0")
		{
			video = new VideoCapture(0);
		}
		else
		{
			video = new VideoCapture(path);
		}
		double totalframecount = video->get(CAP_PROP_FRAME_COUNT);
		time_t start, end;

		double fcount = 0;
		MultiTrack vtracker;

		//FaceUtil::Init("E:\\Code\\opencv_pos\\x64\\Release\\shape_predictor_68_face_landmarks.dat");
		while (true)
		{
			*video >> src;
			if (!src.data)
			{
				cout << "video is end" << endl;
				if (path == "0")
				{
					break;
				}
				video->set(CAP_PROP_POS_FRAMES, 0);
				fcount = 0;
				continue;
			}
			start = clock();
			vector<Face> faces = m_facedetector.detect(src, score, vector<mx_float> {scale});
			
			Draw(src, faces, use_lankmarks);
			end = clock();
			double times = double(end - start) / 1000.0;
			fcount++;
			printf("\r");
			printf("find %d faces fps %f ,remainder %.0f playpercent %.2f%% "
				, faces.size(), 1.0 / times, totalframecount - fcount, (fcount * 100) / totalframecount);
			if (width_disp&&height_disp)
			{
				resize(src, src, Size(width_disp, height_disp));
			}

			imshow(path, src);

			faces.swap(vector<Face>());
			if (waitKey(1) == 27)
			{
				break;
			}
		}
		video->release();
		delete video;
		video = nullptr;

	}
	else
	{
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
	}
	waitKey(0);
	return 0;
}
