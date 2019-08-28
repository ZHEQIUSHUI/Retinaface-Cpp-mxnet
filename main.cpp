#pragma comment(lib,"opencv_world410.lib")
#pragma comment(lib,"libmxnet.lib")
#pragma comment(lib,"dlib19.17.0_release_64bit_msvc1900.lib")


#include "Retinaface.h"
#include "GenderAgePredict.h"
#include "facenet.h"
#include <opencv2\tracking.hpp>
#include "Tracker.h"
//#include "facepose.h"
//using namespace std;
void DrawAgeGenderScore(Mat& src, vector<Face> faces, bool use_landmarks, double fontsize = 0.5, int thickness = 1)
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
			putText(src, faces[i].gender, Point(p.x, p.y + offset), FONT_HERSHEY_COMPLEX, fontsize, Scalar(255, 0, 0), thickness);
			putText(src, "Age:" + to_string((int)faces[i].age), Point(p.x, p.y + 2 * offset), FONT_HERSHEY_COMPLEX, fontsize, Scalar(255, 0, 0), thickness);

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
		cout << "             -gender    use_gender            (type:int,default=1)" << endl;
		cout << "             -age       use_age               (type:int,default=1)" << endl;
		cout << "             -h         resize height         (type:int,default=src.rows)" << endl;
		cout << "             -w         resize width          (type:int,default=src.cols)" << endl;
		cout << "Example:" << endl << endl;
		cout << ">>>>>>>>>>>>> detect in image <<<<<<<<<<<<<" << endl;
		cout << "\\\> .\\retinaface.exe -i .\\people.jpeg -s 1.0 -t 0.8 -l 1 -g 1 -age 1 -gender 1" << endl << endl;
		cout << ">>>>>>>>>>>>> detect in video <<<<<<<<<<<<<" << endl;
		cout << "\\\> .\\retinaface.exe -v .\\multiple_faces.avi -s 0.5 -t 0.8 -l 0 -g 1 -age 1 -gender 1" << endl << endl;
		cout << ">>>>>>>>>>>>> detect in webcam <<<<<<<<<<<<<" << endl;
		cout << "\\\> .\\retinaface.exe -v 0 -s 0.5 -t 0.8 -l 1 -g 1 -age 1 -gender 1" << endl << endl << endl;
		return 0;
	}
	//dlog(argc);
	bool isvideo = true;
	bool use_lankmarks = 1;
	bool use_gpu = 1;
	bool use_gender = 1;
	bool use_age = 1;
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
		else if (string(args[i]) == "-gender")
		{
			use_gender = atoi(args[i + 1]);
		}
		else if (string(args[i]) == "-age")
		{
			use_age = atoi(args[i + 1]);
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
	cout << "use_gender? :" << use_gender << endl;
	cout << "use_age? :" << use_age << endl;
	cout << "width :" << width_disp << endl;
	cout << "height :" << height_disp << endl;
	RetinaFace m_facedetector(use_gpu);
	m_facedetector.Loadmodel("E:/PyCode/insightface/models", "mnet.25");
	m_facedetector.use_landmarks = use_lankmarks;

	GenderAgeDetect m_genderage_detector(use_gpu);

#ifdef Amethod
	m_genderage_detector.Loadmodel("models", "age", "gender");
#elif defined Bmethod
	m_genderage_detector.Loadmodel("models", "model");
#endif // Amethod

	FaceRecognition FaceId(use_gpu);
	FaceId.Loadmodel("E:/Code/Receiver/Receiver/bin/Debug/model", "F_net");
	
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
			/*if (faces.size())
			{
				FaceUtil::DetectPose(src, faces[0].boundingbox);
				FaceUtil::DrawRecord(src, Scalar(0, 0, 125), Scalar(0, 111, 0), 2, 2);
			}*/
			if (faces.size() && use_lankmarks)
			{
				m_genderage_detector.detect(src, faces, use_age, use_gender);
			}
			//¸ú×Ù
			int trackingObj = 0;
			trackingObj = vtracker.update(src, faces);
			/*for (size_t i = 0; i < objs.size(); i++)
			{
				rectangle(src, objs[i].lastbox, Scalar(238, 92, 128),2);
			}*/

			DrawAgeGenderScore(src, faces, use_lankmarks);
			end = clock();
			double times = double(end - start) / 1000.0;
			fcount++;
			printf("\r");
			printf("find %d faces fps %f ,remainder %.0f playpercent %.2f%% ,tracking object %d"
				, faces.size(), 1.0 / times, totalframecount - fcount, (fcount * 100) / totalframecount, trackingObj);
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

		if (faces.size() && use_lankmarks)
		{
			m_genderage_detector.detect(src, faces, use_age, use_gender);
		}
		DrawAgeGenderScore(src, faces, use_lankmarks);
		imshow(path, src);
	}
	waitKey(0);


	//RetinaFace m_facedetector(1);
	//m_facedetector.Loadmodel("E:/PyCode/insightface/models", "mnet.25");
	//m_facedetector.use_landmarks = 1;
	//VideoCapture video(0);
	//Mat src;
	//while (true)
	//{
	//	video >> src;
	//	vector<Face> faces = m_facedetector.detect(src, 0.5, vector<mx_float> {1.0});
	//	Point2f target[5] = {	   Point2f(30.2946, 51.6963),
	//							   Point2f(65.5318, 51.5014),
	//							   Point2f(48.0252, 71.7366),
	//						       Point2f(33.5493, 92.3655),
	//							   Point2f(62.7299, 92.2041) };

	//	Mat _dst(5, 2, CV_32F);
	//	for (size_t i = 0; i < _dst.rows; i++)
	//	{
	//		_dst.at<float>(i, 0) = target[i].x + 8.0;
	//		_dst.at<float>(i, 1) = target[i].y;
	//	}
	//	Mat warp = alignFace(src, faces[0], _dst);
	//	/*Mat _src(5, 2, CV_32F);
	//	for (size_t i = 0; i < _src.rows; i++)
	//	{
	//		_src.at<float>(i, 0) = faces[0].landmarks[i].x;
	//		_src.at<float>(i, 1) = faces[0].landmarks[i].y;
	//	}
	//	Mat m = FacePreprocess::similarTransform(_src, _dst);
	//	Mat _m = m.rowRange(0, 2);
	//
	//	Mat warp;
	//	warpAffine(src, warp, _m, Size(112, 112));*/
	//	
	//	imshow("align", warp);
	//	rectangle(src,faces[0].boundingbox,Scalar(0,0,255));
	//	imshow("src", src);
	//	if (waitKey(1)==27)
	//	{
	//		break;
	//	}
	//}

	return 0;
}
