#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <mxnet-cpp/MxNetCpp.h>
#include <math.h>
#include <fstream>
#include "Face.h"

using namespace mxnet::cpp;
using namespace std;
using namespace cv;

struct Face
{
	Rect boundingbox;
	float score;
	vector<Point2f> landmarks;
};
class RetinaFace
{

	Context *Ctx = nullptr;
	Symbol sym_net;
	std::map<std::string, mxnet::cpp::NDArray> args;
	std::map<std::string, mxnet::cpp::NDArray> aux;

	vector<int> _feat_stride_fpn = { 32, 16, 8 };

	map<int, map<string, vector<mx_float>>> anchor_cfg;

	map<int, vector<vector<mx_float>>> _anchors_fpn;
	map<int, int> _num_anchors;


private:

	vector<vector<mx_float>> _mkanchors(vector<mx_float> ws, vector<mx_float> hs, mx_float x_ctr, mx_float y_ctr)
	{
		vector<vector<mx_float>> anchors;

		for (int i = 0; i < ws.size(); i++)
		{
			vector<mx_float> anchor =
			{
				x_ctr - 0.5f * (ws[i] - 1),
				y_ctr - 0.5f * (hs[i] - 1),
				x_ctr + 0.5f * (ws[i] - 1),
				y_ctr + 0.5f * (hs[i] - 1)
			};
			anchors.push_back(anchor);
		}

		return anchors;
	}

	void _whctrs(vector<mx_float> anchor, int& w, int& h, mx_float& x_ctr, mx_float& y_ctr)
	{
		w = anchor[2] - anchor[0] + 1;
		h = anchor[3] - anchor[1] + 1;
		x_ctr = anchor[0] + 0.5f * mx_float(w - 1);
		y_ctr = anchor[1] + 0.5f * mx_float(h - 1);
	}

	vector<vector<mx_float>>  _scale_enum(vector<mx_float> base_anchor, vector<mx_float> scales)
	{
		int w, h;
		mx_float x_ctr, y_ctr;
		_whctrs(base_anchor, w, h, x_ctr, y_ctr);
		vector<mx_float> ws;
		vector<mx_float> hs;
		for (int i = 0; i < scales.size(); i++)
		{
			ws.push_back(w * scales[i]);
		}
		for (int i = 0; i < scales.size(); i++)
		{
			hs.push_back(h * scales[i]);
		}
		vector<vector<mx_float>> anchors = _mkanchors(ws, hs, x_ctr, y_ctr);
		ws.swap(vector<mx_float>());
		hs.swap(vector<mx_float>());
		return anchors;
	}

	vector<vector<mx_float>> _ratio_enum(vector<mx_float> base_anchor, vector<mx_float> _ratios)
	{
		int w, h;
		mx_float x_ctr, y_ctr;
		_whctrs(base_anchor, w, h, x_ctr, y_ctr);
		mx_float _size = w * h;
		//vector<mx_float> size_ratios ;
		/* for(int i = 0; i < _ratios.size(); i++)
		{
		size_ratios.push_back(_size / _ratios[i]);
		}*/
		vector<mx_float> ws;
		vector<mx_float> hs;
		for (int i = 0; i < _ratios.size(); i++)
		{
			ws.push_back(round(sqrt(_size / _ratios[i])));
		}

		for (int i = 0; i < ws.size(); i++)
		{
			hs.push_back(round(ws[i] * _ratios[i]));
		}
		vector<vector<mx_float>> anchors = _mkanchors(ws, hs, x_ctr, y_ctr);
		ws.swap(vector<mx_float>());
		hs.swap(vector<mx_float>());
		return anchors;
	}

	vector<vector<mx_float>> generate_anchors(vector<mx_float> bs, vector<mx_float> ratios, vector<mx_float> scales, int stride, bool dense_anchor)
	{
		vector<mx_float> base_anchor{ 0, 0, bs[0] - 1, bs[0] - 1 };
		vector<vector<mx_float>> ratio_anchors = _ratio_enum(base_anchor, ratios);

		vector<vector<mx_float>> anchors;
		for (int i = 0; i < ratio_anchors.size(); i++)
		{
			vector<vector<mx_float>> anchors_temp = _scale_enum(ratio_anchors[i], scales);
			anchors.insert(anchors.end(), anchors_temp.begin(), anchors_temp.end());
			/*for (int j = 0; j < anchors_temp.size(); j++)
			{
			anchors.push_back(anchors_temp[j]);
			}*/
		}

		if (dense_anchor)
		{
			if (stride % 2 != 0)
			{
				cout << ("stride%2!=0") << endl;
				vector<vector<mx_float>> anchors_error;
				return anchors_error;
			}
			vector<vector<mx_float>> anchors2(anchors);
			for (int i = 0; i < anchors2.size(); i++)
			{
				for (int j = 0; j < anchors2[i].size(); j++)
				{
					anchors2[i][j] += (stride / 2);
				}
			}
			anchors.insert(anchors.end(), anchors2.begin(), anchors2.end());
		}
		return anchors;
	}

	vector<vector<vector<mx_float>>> generate_anchors_fpn(bool dense_anchor = false)
	{
		vector<vector<vector<mx_float>>> anchors;
		for (int i = 0; i < _feat_stride_fpn.size(); i++)
		{
			int stride = _feat_stride_fpn[i];
			map<string, vector<mx_float>> v = anchor_cfg[stride];
			vector<mx_float> bs = v["BASE_SIZE"];
			vector<mx_float> _ratios = v["RATIOS"];
			vector<mx_float> _scales = v["SCALES"];
			vector<vector<mx_float>> anchors_temp = generate_anchors(bs, _ratios, _scales, stride, dense_anchor);
			anchors.push_back(anchors_temp);
		}
		return anchors;
	}


	void Init()
	{
		vector<string> fpn_keys;
		for (int i = 0; i < _feat_stride_fpn.size(); i++)
		{
			fpn_keys.push_back("stride" + to_string(_feat_stride_fpn[i]));
		}
		vector<mx_float> a11{ 32, 16 };
		vector<mx_float> a12{ 16 };
		vector<mx_float> a13{ 1.0 };
		vector<mx_float> a14{ 9999 };

		vector<mx_float> a21{ 8, 4 };
		vector<mx_float> a22{ 16 };
		vector<mx_float> a23{ 1.0 };
		vector<mx_float> a24{ 9999 };

		vector<mx_float> a31{ 2, 1 };
		vector<mx_float> a32{ 16 };
		vector<mx_float> a33{ 1.0 };
		vector<mx_float> a34{ 9999 };
		map <string, vector<mx_float>> a1;
		a1["SCALES"] = a11;
		a1["BASE_SIZE"] = a12;
		a1["RATIOS"] = a13;
		a1["ALLOWED_BORDER"] = a14;
		map < string, vector<mx_float>> a2;
		a2["SCALES"] = a21;
		a2["BASE_SIZE"] = a22;
		a2["RATIOS"] = a23;
		a2["ALLOWED_BORDER"] = a24;
		map < string, vector<mx_float>> a3;
		a3["SCALES"] = a31;
		a3["BASE_SIZE"] = a32;
		a3["RATIOS"] = a33;
		a3["ALLOWED_BORDER"] = a34;
		anchor_cfg[32] = a1;
		anchor_cfg[16] = a2;
		anchor_cfg[8] = a3;
		bool dense_anchor = false;
		vector<vector<vector<mx_float>>> anchors = generate_anchors_fpn(dense_anchor);

		for (int i = 0; i < _feat_stride_fpn.size(); i++)
		{
			_anchors_fpn[_feat_stride_fpn[i]] = anchors[i];
		}

		for (int i = 0; i < _feat_stride_fpn.size(); i++)
		{
			_num_anchors[_feat_stride_fpn[i]] = anchors[i].size();
		}
	}

	vector<vector<vector<vector<mx_float>>>> anchors_plane(int height, int width, int stride, vector<vector<mx_float>> anchors_fpn)
	{
		int A = anchors_fpn.size();
		vector<vector<vector<vector<mx_float>>>> all_anchors(height, vector<vector<vector<mx_float>>>(width, vector<vector<mx_float>>(A, vector<mx_float>(4)))); //(height, width, A, 4)

		int	k = 0;
		int	sh = 0;
		int	sw = 0;

		for (size_t iw = 0; iw < width; iw++)
		{
			sw = iw * stride;
			for (size_t ih = 0; ih < height; ih++)
			{
				sh = ih * stride;
				for (size_t k = 0; k < A; k++)
				{
					all_anchors[ih][iw][k][0] = anchors_fpn[k][0] + sw;
					all_anchors[ih][iw][k][1] = anchors_fpn[k][1] + sh;
					all_anchors[ih][iw][k][2] = anchors_fpn[k][2] + sw;
					all_anchors[ih][iw][k][3] = anchors_fpn[k][3] + sh;
				}
			}
		}

		return all_anchors;
	}

	void _clip_pad(std::vector<mx_float> tensor, int height, int width)
	{

	}

	vector<vector<mx_float>> bbox_pred_and_clip_boxes(vector<vector<mx_float>> boxes, vector<vector<mx_float>> bbox_deltas, int width, int height)
	{
		vector<vector<mx_float>> pred_boxes;
		if (boxes.size() == 0)
		{
			return pred_boxes;
		}
		/*vector<mx_float> widths;
		vector<mx_float> heights;
		vector<mx_float> ctr_xs;
		vector<mx_float> ctr_ys;
		for (size_t bi = 0; bi < boxes.size(); bi++)
		{
		vector<mx_float> box = boxes[bi];
		mx_float width = box[2] - box[0] + 1.0f;
		mx_float height = box[3] - box[1] + 1.0f;
		mx_float ctr_x = box[0] + 0.5f*(width - 1.0f);
		mx_float ctr_y = box[1] + 0.5f*(height - 1.0f);

		widths.push_back(width);
		heights.push_back(height);
		ctr_xs.push_back(ctr_x);
		ctr_ys.push_back(ctr_y);
		}
		vector<mx_float> dx;
		vector<mx_float> dy;
		vector<mx_float> dw;
		vector<mx_float> dh;
		for (size_t bi = 0; bi < bbox_deltas.size(); bi++)
		{
		vector<mx_float> box = bbox_deltas[bi];
		dx.push_back(box[0]);
		dy.push_back(box[1]);
		dw.push_back(box[2]);
		dh.push_back(box[3]);
		}
		vector<mx_float> pred_ctr_xs;
		vector<mx_float> pred_ctr_ys;
		vector<mx_float> pred_ws;
		vector<mx_float> pred_hs;
		for (size_t xi = 0; xi < boxes.size(); xi++)
		{
		vector<mx_float> box = boxes[xi];
		mx_float width = box[2] - box[0] + 1.0f;
		mx_float height = box[3] - box[1] + 1.0f;
		mx_float ctr_x = box[0] + 0.5f*(width - 1.0f);
		mx_float ctr_y = box[1] + 0.5f*(height - 1.0f);

		vector<mx_float> box_delta = bbox_deltas[xi];
		mx_float pred_ctr_x = box_delta[0] * width + ctr_x;
		mx_float pred_ctr_y = box_delta[1] * height + ctr_y;
		mx_float pred_w = exp(box_delta[2])* width;
		mx_float pred_h = exp(box_delta[3]) * height;
		pred_ctr_xs.push_back(pred_ctr_x);
		pred_ctr_ys.push_back(pred_ctr_y);
		pred_ws.push_back(pred_w);
		pred_hs.push_back(pred_h);
		}*/
		for (size_t i = 0; i < boxes.size(); i++)
		{
			vector<mx_float> box = boxes[i];
			mx_float w = box[2] - box[0] + 1.0f;
			mx_float h = box[3] - box[1] + 1.0f;
			mx_float ctr_x = box[0] + 0.5f*(w - 1.0f);
			mx_float ctr_y = box[1] + 0.5f*(h - 1.0f);

			vector<mx_float> box_delta = bbox_deltas[i];
			mx_float pred_ctr_x = box_delta[0] * w + ctr_x;
			mx_float pred_ctr_y = box_delta[1] * h + ctr_y;
			mx_float pred_w = exp(box_delta[2]) * w;
			mx_float pred_h = exp(box_delta[3]) * h;

			vector<mx_float> box_result;
			mx_float x1 = pred_ctr_x - 0.5f*(pred_w - 1.0f);
			mx_float y1 = pred_ctr_y - 0.5f*(pred_h - 1.0f);
			mx_float x2 = pred_ctr_x + 0.5f*(pred_w - 1.0f);
			mx_float y2 = pred_ctr_y + 0.5f*(pred_h - 1.0f);
			box_result.push_back(max(min(x1, mx_float(width - 1)), 0.0f));  // width > x > 0
			box_result.push_back(max(min(y1, mx_float(height - 1)), 0.0f)); // height > y > 0
			box_result.push_back(max(min(x2, mx_float(width - 1)), 0.0f));
			box_result.push_back(max(min(y2, mx_float(height - 1)), 0.0f));
			pred_boxes.push_back(box_result);
		}
		return pred_boxes;
	}


	mxnet::cpp::NDArray data2ndarray(mxnet::cpp::Context ctx, float * data, int batch_size, int num_channels, int height, int width)
	{
		mxnet::cpp::NDArray ret(mxnet::cpp::Shape(batch_size, num_channels, height, width), ctx, false);

		ret.SyncCopyFromCPU(data, batch_size * num_channels * height * width);

		ret.WaitToRead();  //mxnet::cpp::NDArray::WaitAll();

		return ret;
	}

	//select element by indexs in vector
	template <typename T>
	vector<T> selectByindex(vector<T> vec, vector<size_t> idxs)
	{
		vector<T> result;
		for (size_t i = 0; i < idxs.size(); i++)
		{
			result.push_back(vec[idxs[i]]);
		}
		return result;
	}

	//true 从大到小 获取从大到小顺序的索引
	//v={100,300,200} false->return {0,2,1}  true->return {1,2,0}
	template <typename T>
	vector<size_t> sort_indexes_e(vector<T> &v, bool reverse = false)
	{
		vector<size_t> idx(v.size());
		iota(idx.begin(), idx.end(), 0);
		sort(idx.begin(), idx.end(),
			[&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });
		if (reverse)
		{
			std::reverse(idx.begin(), idx.end());
		}
		return idx;
	}

	bool cmp(size_t a, size_t b)
	{
		return a > b;
	}

	vector<vector<Point2f>> landmark_pred(vector<vector<mx_float>> boxes, vector<vector<Point2f>> landmark_deltas)
	{
		vector<vector<Point2f>> pred;
		if (boxes.size() == 0)
		{
			return pred;
		}
		/*	vector<mx_float> widths;
		vector<mx_float> heights;
		vector<mx_float> ctr_xs;
		vector<mx_float> ctr_ys;
		for (size_t i = 0; i < boxes.size(); i++)
		{
		vector<mx_float> box = boxes[i];
		mx_float width = box[2] - box[0] + 1.0f;
		mx_float height = box[3] - box[1] + 1.0f;
		mx_float ctr_x = box[0] + 0.5f*(width - 1.0f);
		mx_float ctr_y = box[1] + 0.5f*(height - 1.0f);
		widths.push_back(width);
		heights.push_back(height);
		ctr_xs.push_back(ctr_x);
		ctr_ys.push_back(ctr_y);
		}
		*/
		for (size_t i = 0; i < boxes.size(); i++)
		{
			vector<mx_float> box = boxes[i];
			vector<Point2f> landmarks = landmark_deltas[i];
			for (size_t j = 0; j < 5; j++)
			{
				landmarks[j].x = landmarks[j].x*(box[2] - box[0] + 1.0f) + (box[0] + 0.5f*((box[2] - box[0] + 1.0f) - 1.0f));
				landmarks[j].y = landmarks[j].y*(box[3] - box[1] + 1.0f) + box[1] + 0.5f*((box[3] - box[1] + 1.0f) - 1.0f);
				/*	landmarks[j].x = landmarks[j].x*widths[i] + ctr_xs[i];
				landmarks[j].y = landmarks[j].y*heights[i] + ctr_ys[i];*/
			}
			pred.push_back(landmarks);
		}
		return pred;
	}

	vector<size_t> nms(vector<vector<mx_float>> proposals_list, vector<mx_float> scores_list, mx_float thresh)
	{
		//减少装拆箱 reduce unboxing and packing times
		/*vector<mx_float> x1s;
		vector<mx_float> y1s;
		vector<mx_float> x2s;
		vector<mx_float> y2s;
		for (size_t i = 0; i < proposals_list.size(); i++)
		{
		vector<mx_float> box = proposals_list[i];
		x1s.push_back(box[0]);
		y1s.push_back(box[1]);
		x2s.push_back(box[2]);
		y2s.push_back(box[3]);
		}*/
		vector<mx_float> areas;
		for (size_t i = 0; i < proposals_list.size(); i++)
		{
			vector<mx_float> box = proposals_list[i];
			areas.push_back((box[2] - box[0] + 1)*(box[3] - box[1] + 1));
		}
		//int ndets = proposals_list.size();
		vector<int> suppressed(proposals_list.size());
		int i = 0;
		int j = 0;
		mx_float ix1 = 0.0, iy1 = 0.0, ix2 = 0.0, iy2 = 0.0, iarea = 0.0;
		mx_float xx1 = 0.0, yy1 = 0.0, xx2 = 0.0, yy2 = 0.0;
		mx_float w = 0.0, h = 0.0;
		mx_float inter = 0.0, ovr = 0.0;
		vector<size_t> order = sort_indexes_e(scores_list, true);
		vector<size_t> keep;
		for (size_t _i = 0; _i < proposals_list.size(); _i++)
		{
			i = order[_i];
			if (suppressed[i] == 1)
				continue;

			keep.push_back(i);
			vector<mx_float> box = proposals_list[i];
			mx_float ix1 = box[0];
			mx_float iy1 = box[1];
			mx_float ix2 = box[2];
			mx_float iy2 = box[3];
			mx_float iarea = areas[i];
			for (size_t _j = _i + 1; _j < proposals_list.size(); _j++)
			{
				j = order[_j];
				if (suppressed[j] == 1)
					continue;
				vector<mx_float> _box = proposals_list[j];
				mx_float xx1 = max(box[0], _box[0]);
				mx_float yy1 = max(box[1], _box[1]);
				mx_float xx2 = min(box[2], _box[2]);
				mx_float yy2 = min(box[3], _box[3]);
				mx_float w = max(0.0f, xx2 - xx1 + 1);
				mx_float h = max(0.0f, yy2 - yy1 + 1);
				mx_float inter = w * h;
				mx_float ovr = inter / (iarea + areas[j] - inter);
				if (ovr >= thresh)
					suppressed[j] = 1;
			}
		}
		areas.swap(vector<mx_float>());
		return keep;
	}

	/*void bbox_vote(vector<vector<mx_float>>& proposals_list, vector<mx_float>& scores_list)
	{
	vector<mx_float> x1s;
	vector<mx_float> y1s;
	vector<mx_float> x2s;
	vector<mx_float> y2s;
	vector<mx_float> inter;
	vector<size_t> merge_index;
	vector<vector<mx_float>> proposal_accu_sum;
	vector<mx_float> proposal_accu_sum_score;
	vector<vector<mx_float>> proposals;
	vector<mx_float> scores;
	for (size_t i = 0; i < proposals_list.size(); i++)
	{
	vector<mx_float> box = proposals_list[i];
	x1s.push_back(box[0]);
	y1s.push_back(box[1]);
	x2s.push_back(box[2]);
	y2s.push_back(box[3]);
	}
	vector<mx_float> areas;
	for (size_t i = 0; i < x1s.size(); i++)
	{
	areas.push_back((x2s[i] - x1s[i] + 1)*(y2s[i] - y1s[i] + 1));
	}
	vector<mx_float> box_0 = proposals_list[0];
	for (size_t i = 0; i < proposals_list.size(); i++)
	{
	vector<mx_float> box = proposals_list[i];
	x1s[i] = max(box_0[0], box[0]);
	y1s[i] = max(box_0[1], box[1]);
	x2s[i] = min(box_0[2], box[2]);
	y2s[i] = min(box_0[3], box[3]);
	}

	for (size_t i = 0; i < proposals_list.size(); i++)
	{
	inter.push_back(max(0.0f, x2s[i] - x1s[i] + 1)*max(0.0f, y2s[i] - y1s[i] + 1));
	}

	mx_float area0 = areas[0];

	for (size_t i = 0; i < proposals_list.size(); i++)
	{
	if ((inter[i] / (area0 + areas[i] - inter[i])) > nms_threshold)
	{
	merge_index.push_back(i);
	}
	}
	vector<vector<mx_float>> proposals_accu = selectByindex(proposals_list, merge_index);
	vector<mx_float> score_accu = selectByindex(scores_list, merge_index);
	std::sort(merge_index.begin(), merge_index.end(), this->cmp);

	std::vector<vector<mx_float>>::iterator it_proposal = proposals_list.begin();
	std::vector<mx_float>::iterator it_score = scores_list.begin();
	for (size_t i = 0; i < merge_index.size(); i++)
	{
	proposals_list.erase(it_proposal + merge_index[i]);
	scores_list.erase(it_score + merge_index[i]);
	}

	if (merge_index.size() <= 1 && proposals_list.size() == 0)
	{
	scores = score_accu;
	proposals = proposals_accu;
	}

	for (size_t i = 0; i < proposals_accu.size(); i++)
	{
	proposals_accu[i][0] = proposals_accu[i][0] * score_accu[i];
	proposals_accu[i][1] = proposals_accu[i][1] * score_accu[i];
	proposals_accu[i][2] = proposals_accu[i][2] * score_accu[i];
	proposals_accu[i][3] = proposals_accu[i][3] * score_accu[i];
	}
	std::vector<mx_float>::iterator max_score = std::max_element(score_accu.begin(), score_accu.end());

	for (size_t i = 0; i < proposals_accu.size(); i++)
	{
	vector<mx_float> box;
	mx_float sum = proposals_accu[i][0] + proposals_accu[i][1] + proposals_accu[i][2] + proposals_accu[i][3];
	box.push_back(sum / (*max_score));
	box.push_back(sum / (*max_score));
	box.push_back(sum / (*max_score));
	box.push_back(sum / (*max_score));
	proposal_accu_sum.push_back(box);
	proposal_accu_sum_score.push_back(*max_score);
	}
	scores.insert(scores.end(), proposal_accu_sum_score.begin(), proposal_accu_sum_score.end());
	proposals.insert(proposals.end(), proposal_accu_sum.begin(), proposal_accu_sum.end());
	}*/

public:

	mx_float nms_threshold = 0.4;
	mx_float decay4 = 0.5;
	bool use_landmarks = true;
	bool vote = false;
	RetinaFace(bool use_gpu)
	{
		Ctx = use_gpu ? new Context(kGPU, 0) : new Context(kCPU, 0);

		int fmc = 3;

		Init();
	}

	~RetinaFace()
	{
		delete Ctx;
	}

	void Loadmodel(String floder, String prefix)
	{
		sym_net = Symbol::Load(floder + "/" + prefix + "-symbol.json");
		std::map<std::string, mxnet::cpp::NDArray> params;
		NDArray::Load(floder + "/" + prefix + "-0000.params", nullptr, &params);
		for (const auto &k : params)
		{
			if (k.first.substr(0, 4) == "aux:")
			{
				auto name = k.first.substr(4, k.first.size() - 4);
				aux[name] = k.second.Copy(*Ctx);
			}
			if (k.first.substr(0, 4) == "arg:")
			{
				auto name = k.first.substr(4, k.first.size() - 4);
				args[name] = k.second.Copy(*Ctx);
			}
		}
		// WaitAll is need when we copy data between GPU and the main memory
		mxnet::cpp::NDArray::WaitAll();

	}

	vector<Face> detect(Mat img, float thresh = 0.8, vector<float> scales = { 1.0 }, bool do_flip = false)
	{
		vector<vector<mx_float>> proposals_list;
		vector<mx_float> scores_list;
		vector<vector<Point2f>> landmarks_list;
		vector<int> flips = { 0 };
		if (do_flip)
		{
			flips = { 0, 1 };
		}
		for (int i = 0; i < scales.size(); i++)
		{
			float scale = scales[i];
			for (int j = 0; j < flips.size(); j++)
			{
				int _flip = flips[j];
				Mat temp;
				if (scale != 1.0)
				{
					resize(img, temp, Size(), scale, scale);
				}
				else
				{
					img.copyTo(temp);
				}

				if (_flip)
				{
					flip(temp, temp, _flip);
				}

				temp.convertTo(temp, CV_32FC3);
				Mat bgr[3];
				split(temp, bgr);

				Mat b_img = bgr[0];
				Mat g_img = bgr[1];
				Mat r_img = bgr[2];

				int len_img = temp.cols * temp.rows;
				float* data_img = new float[temp.channels()*len_img]; //rrrrr...ggggg...bbbbb...
				memcpy(data_img, r_img.data, len_img * sizeof(*data_img));
				memcpy(data_img + len_img, g_img.data, len_img * sizeof(*data_img));
				memcpy(data_img + len_img + len_img, b_img.data, len_img * sizeof(*data_img));

				NDArray data = data2ndarray(*Ctx, data_img, 1, 3, temp.rows, temp.cols);

				args["data"] = data;
				Executor *exec = sym_net.SimpleBind(*Ctx, args, map<string, NDArray>(), map<string, OpReqType>(), aux);
				exec->Forward(false);

				//process outputs
				for (int _idx = 0; _idx < _feat_stride_fpn.size(); _idx++)
				{
					int stride = _feat_stride_fpn[_idx];
					int A = _num_anchors[stride];
					int idx = 0;
					/*if (use_landmarks)
					idx = _idx * 3;
					else
					idx = _idx * 2;*/
					idx = _idx * 3;
					//score
					std::vector<mx_float> score_data;
					vector<mx_uint> score_data_shape = exec->outputs[idx].GetShape();
					int size_dim = score_data_shape[2] * score_data_shape[3];
					exec->outputs[idx].SyncCopyToCPU(&score_data, exec->outputs[idx].Size());
					int offset = (A * size_dim);
					std::vector<mx_float> scores_temp(score_data.begin() + offset, score_data.end());


					//bbox
					idx++;
					std::vector<mx_float> bbox_data;
					vector<vector<mx_float>> bbox_deltas;
					vector<mx_uint> bbox_data_shape = exec->outputs[idx].GetShape();
					int data_height = bbox_data_shape[2];
					int data_width = bbox_data_shape[3];
					exec->outputs[idx].SyncCopyToCPU(&bbox_data, exec->outputs[idx].Size());
					int K = data_height*data_width;
					for (size_t b = 0; b < bbox_data.size() / 8; b++)
					{
						std::vector<mx_float> box1(4);
						std::vector<mx_float> box2(4);
						box1[0] = bbox_data[b + 0 * K];
						box1[1] = bbox_data[b + 1 * K];
						box1[2] = bbox_data[b + 2 * K];
						box1[3] = bbox_data[b + 3 * K];
						box2[0] = bbox_data[b + 4 * K];
						box2[1] = bbox_data[b + 5 * K];
						box2[2] = bbox_data[b + 6 * K];
						box2[3] = bbox_data[b + 7 * K];
						bbox_deltas.push_back(box1);
						bbox_deltas.push_back(box2);
					}


					vector<vector<mx_float>> anchors_fpn = _anchors_fpn[stride];

					vector<vector<vector<vector<mx_float>>>> all_anchors = anchors_plane(data_height, data_width, stride, anchors_fpn);
					vector<vector<mx_float>> anchors;
					for (size_t h = 0; h < all_anchors.size(); h++)
					{
						vector<vector<vector<mx_float>>> hh = all_anchors[h];
						for (size_t w = 0; w < hh.size(); w++)
						{
							vector<vector<mx_float>> ww = hh[w];
							for (size_t a = 0; a < ww.size(); a++)
							{
								vector<mx_float> anchor = ww[a];
								anchors.push_back(anchor);
							}
						}
					}
					//√
					vector<vector<mx_float>> proposals = bbox_pred_and_clip_boxes(anchors, bbox_deltas, temp.cols, temp.rows);

					vector<mx_float> scores;
					for (size_t si = 0; si < scores_temp.size() / 2; si++)
					{
						scores.push_back(scores_temp[si]);
						scores.push_back(scores_temp[si + K]);
					}


					vector<size_t> order;  //which score (> threshold) index
					for (size_t s = 0; s < scores.size(); s++)
					{
						if (scores[s] > thresh)
						{
							order.push_back(s);
						}
					}

					proposals = selectByindex(proposals, order);
					scores = selectByindex(scores, order);


					if (stride == 4 && decay4 < 1.0f)
					{
						for (size_t s = 0; s < scores.size(); s++)
						{
							scores[s] *= decay4;
						}
					}

					if (_flip)
					{
						for (size_t f = 0; f < proposals.size(); f++)
						{
							mx_float oldx0 = proposals[f][0];
							mx_float oldx2 = proposals[f][2];
							proposals[f][0] = temp.cols - oldx2 - 1;
							proposals[f][2] = temp.cols - oldx0 - 1;
						}
					}

					if (scale != 1.0f)
					{
						for (size_t f = 0; f < proposals.size(); f++)
						{
							//restore to src resolution
							proposals[f][0] /= scale;
							proposals[f][1] /= scale;
							proposals[f][2] /= scale;
							proposals[f][3] /= scale;
						}
					}

					scores_list.insert(scores_list.end(), scores.begin(), scores.end());
					proposals_list.insert(proposals_list.end(), proposals.begin(), proposals.end());
					//OK
					if (!vote&&use_landmarks)
					{
						idx++;
						vector<vector<Point2f>> landmark_deltas;
						vector<mx_float> landmarks_data;
						exec->outputs[idx].SyncCopyToCPU(&landmarks_data, exec->outputs[idx].Size());
						for (size_t z = 0; z < landmarks_data.size() / 20; z++) //transpose reshape
						{
							vector<Point2f> landmarks_temp1;
							vector<Point2f> landmarks_temp2;
							for (size_t y = 0; y < 5; y++)
							{
								Point2f p1(landmarks_data[2 * y*K + z], landmarks_data[(2 * y + 1)*K + z]);
								landmarks_temp1.push_back(p1);
							}
							for (size_t y = 5; y < 10; y++)
							{
								Point2f p2(landmarks_data[2 * y*K + z], landmarks_data[(2 * y + 1)*K + z]);
								landmarks_temp2.push_back(p2);
							}

							landmark_deltas.push_back(landmarks_temp1);
							landmark_deltas.push_back(landmarks_temp2);
						}

						vector<vector<Point2f>> landmarks = landmark_pred(anchors, landmark_deltas);
						landmarks = selectByindex(landmarks, order);

						if (_flip)
						{
							for (size_t l = 0; l < landmarks.size(); l++)
							{
								for (size_t k = 0; k < 5; k++)
								{
									landmarks[l][k].x = temp.cols - landmarks[l][k].x - 1;
								}
							}
							for (size_t l = 0; l < landmarks.size(); l++)
							{
								Point2f p0(landmarks[l][0]);
								Point2f p1(landmarks[l][1]);
								Point2f p3(landmarks[l][3]);
								Point2f p4(landmarks[l][4]);
								landmarks[l][0] = p1;
								landmarks[l][1] = p0;
								landmarks[l][3] = p4;
								landmarks[l][4] = p3;
							}
						}

						if (scale != 1.0)
						{
							for (size_t l = 0; l < landmarks.size(); l++)
							{
								for (size_t k = 0; k < 5; k++)
								{
									landmarks[l][k].x /= scale;
									landmarks[l][k].y /= scale;
								}
							}
						}

						landmarks_list.insert(landmarks_list.end(), landmarks.begin(), landmarks.end());
					}
				}
				delete[] data_img;
				data_img = nullptr;
				delete exec;
				exec = nullptr;
				temp.release();
				r_img.release();
				g_img.release();
				b_img.release();
			}
		}
		if (proposals_list.size() == 0)
		{
			return vector<Face>();
		}

		vector<size_t> order = sort_indexes_e(scores_list, true);
		proposals_list = selectByindex(proposals_list, order);
		scores_list = selectByindex(scores_list, order);
		if (!vote&&use_landmarks)
		{
			landmarks_list = selectByindex(landmarks_list, order);
		}
		if (!vote)
		{
			vector<size_t> keep = nms(proposals_list, scores_list, nms_threshold);
			proposals_list = selectByindex(proposals_list, keep);
			scores_list = selectByindex(scores_list, keep);
			if (use_landmarks)
			{
				landmarks_list = selectByindex(landmarks_list, keep);
			}
			vector<Face> faces;
			for (size_t f = 0; f < proposals_list.size(); f++)
			{
				int x = proposals_list[f][0];
				int y = proposals_list[f][1];
				int w = proposals_list[f][2] - proposals_list[f][0];
				int h = proposals_list[f][3] - proposals_list[f][1];
				Rect facebox(x, y, w, h);
				mx_float score = scores_list[f];
				vector<Point2f> lankmarks;
				if (use_landmarks)
				{
					lankmarks = landmarks_list[f];
				}
				Face face;
				/*{ facebox, score, lankmarks };*/
				face.boundingbox = facebox;
				face.score = score;
				face.landmarks = lankmarks;
				faces.push_back(face);
			}
			return faces;
		}
		else
		{
			return vector<Face>();
		}
	}

	/*
	imgs:all images must same resolution
	*/
	vector<vector<Face>> detect(vector<Mat> imgs, float thresh = 0.8, vector<float> scales = { 1.0 }, bool do_flip = false)
	{
		vector<vector<vector<mx_float>>> v_proposals_list(imgs.size());
		vector<vector<mx_float>> v_scores_list(imgs.size());
		vector<vector<vector<Point2f>>> v_landmarks_list(imgs.size());
		vector<vector<Face>> result(imgs.size());
		vector<int> flips = { 0 };
		if (do_flip)
		{
			flips = { 0, 1 };
		}
		for (int i = 0; i < scales.size(); i++)
		{
			float scale = scales[i];
			for (int j = 0; j < flips.size(); j++)
			{
				int _flip = flips[j];
				int width = 0;
				int height = 0;
				float* data_img = nullptr;
				for (size_t i_img = 0; i_img < imgs.size(); i_img++)
				{
					Mat img = imgs[i_img];
					Mat temp;
					if (scale != 1.0)
					{
						resize(img, temp, Size(), scale, scale);
					}
					else
					{
						img.copyTo(temp);
					}

					if (width == 0 && height == 0)
					{
						width = temp.cols;
						height = temp.rows;
						data_img = new float[imgs.size()*temp.channels()*height*width]; //batch_size*channel*height*width
					}

					if (_flip)
					{
						flip(temp, temp, _flip);
					}

					temp.convertTo(temp, CV_32FC3);
					Mat bgr[3];
					split(temp, bgr);
					Mat b_img = bgr[0];
					Mat g_img = bgr[1];
					Mat r_img = bgr[2];
					int len_with_onechannel = temp.cols * temp.rows;
					int img_offset = temp.channels()*len_with_onechannel;
					memcpy(data_img + i_img*img_offset + 0 * len_with_onechannel, r_img.data, len_with_onechannel * sizeof(*data_img));
					memcpy(data_img + i_img*img_offset + 1 * len_with_onechannel, g_img.data, len_with_onechannel * sizeof(*data_img));
					memcpy(data_img + i_img*img_offset + 2 * len_with_onechannel, b_img.data, len_with_onechannel * sizeof(*data_img));
					temp.release();
					r_img.release();
					g_img.release();
					b_img.release();
				}

				NDArray data = data2ndarray(*Ctx, data_img, imgs.size(), 3, height, width);

				args["data"] = data;
				Executor *exec = sym_net.SimpleBind(*Ctx, args, map<string, NDArray>(), map<string, OpReqType>(), aux);
				exec->Forward(false);

				//process outputs
				for (int _idx = 0; _idx < _feat_stride_fpn.size(); _idx++)
				{
					int stride = _feat_stride_fpn[_idx];
					int A = _num_anchors[stride];
					int idx = 0;
					/*if (use_landmarks)
					idx = _idx * 3;
					else
					idx = _idx * 2;*/
					idx = _idx * 3;
					//score
					vector<mx_float> score_data;
					vector<mx_uint> score_data_shape = exec->outputs[idx].GetShape();
					int size_dim = score_data_shape[2] * score_data_shape[3];
					exec->outputs[idx].SyncCopyToCPU(&score_data, exec->outputs[idx].Size());
					int offset = (A * size_dim);
					int offset_perimage = score_data_shape[1] * score_data_shape[2] * score_data_shape[3];
					vector<vector<mx_float>> v_scores_temp;
					for (size_t v = 0; v < imgs.size(); v++)
					{
						vector<mx_float> scores_temp(score_data.begin() + v*offset_perimage + offset, score_data.begin() + (v + 1)*offset_perimage);
						v_scores_temp.push_back(scores_temp);
					}

					//bbox
					idx++;
					vector<mx_float> bbox_data;
					vector<vector<vector<mx_float>>> v_bbox_deltas;
					vector<mx_uint> bbox_data_shape = exec->outputs[idx].GetShape();
					int data_height = bbox_data_shape[2];
					int data_width = bbox_data_shape[3];
					exec->outputs[idx].SyncCopyToCPU(&bbox_data, exec->outputs[idx].Size());
					int K = data_height*data_width;
					offset_perimage = bbox_data_shape[1] * bbox_data_shape[2] * bbox_data_shape[3];
					for (size_t v = 0; v < imgs.size(); v++)
					{
						vector<mx_float> t_bbox_data(bbox_data.begin() + v*offset_perimage, bbox_data.begin() + (v + 1)*offset_perimage);
						vector<vector<mx_float>> bbox_deltas;
						for (size_t b = 0; b < t_bbox_data.size() / 8; b++)
						{
							vector<mx_float> box1(4);
							vector<mx_float> box2(4);
							box1[0] = t_bbox_data[b + 0 * K];
							box1[1] = t_bbox_data[b + 1 * K];
							box1[2] = t_bbox_data[b + 2 * K];
							box1[3] = t_bbox_data[b + 3 * K];
							box2[0] = t_bbox_data[b + 4 * K];
							box2[1] = t_bbox_data[b + 5 * K];
							box2[2] = t_bbox_data[b + 6 * K];
							box2[3] = t_bbox_data[b + 7 * K];
							bbox_deltas.push_back(box1);
							bbox_deltas.push_back(box2);
						}
						v_bbox_deltas.push_back(bbox_deltas);
					}
					vector<vector<mx_float>> anchors;

					vector<vector<mx_float>> anchors_fpn = _anchors_fpn[stride];

					vector<vector<vector<vector<mx_float>>>> all_anchors = anchors_plane(data_height, data_width, stride, anchors_fpn);
					for (size_t h = 0; h < all_anchors.size(); h++)
					{
						vector<vector<vector<mx_float>>> hh = all_anchors[h];
						for (size_t w = 0; w < hh.size(); w++)
						{
							vector<vector<mx_float>> ww = hh[w];
							for (size_t a = 0; a < ww.size(); a++)
							{
								vector<mx_float> anchor = ww[a];
								anchors.push_back(anchor);
							}
						}
					}
					
					//√
					vector<vector<vector<mx_float>>> v_proposals;
					for (size_t v = 0; v < v_bbox_deltas.size(); v++)
					{
						vector<vector<mx_float>> proposals = bbox_pred_and_clip_boxes(anchors, v_bbox_deltas[v], width, height);
						v_proposals.push_back(proposals);
					}
					vector<vector<mx_float>> v_scores;
					for (size_t v = 0; v < v_scores_temp.size(); v++)
					{
						vector<mx_float> scores;
						for (size_t si = 0; si < v_scores_temp[v].size() / 2; si++)
						{
							scores.push_back(v_scores_temp[v][si]);
							scores.push_back(v_scores_temp[v][si + K]);
						}
						v_scores.push_back(scores);
					}

					vector<vector<size_t>> v_order;
					for (size_t v = 0; v < v_scores.size(); v++)
					{
						vector<size_t> order;
						for (size_t s = 0; s < v_scores[v].size(); s++)
						{
							if (v_scores[v][s] > thresh)
							{
								order.push_back(s);
							}
						}
						v_order.push_back(order);
					}

					for (size_t v = 0; v < v_proposals.size(); v++)
					{
						v_proposals[v] = selectByindex(v_proposals[v], v_order[v]);
						v_scores[v] = selectByindex(v_scores[v], v_order[v]);
					}

					if (stride == 4 && decay4 < 1.0f)
					{
						for (size_t v = 0; v < v_scores.size(); v++)
						{
							for (size_t s = 0; s < v_scores[v].size(); s++)
							{
								v_scores[v][s] *= decay4;
							}
						}
					}

					if (_flip)
					{
						for (size_t v = 0; v < v_proposals.size(); v++)
						{
							for (size_t f = 0; f < v_proposals[v].size(); f++)
							{
								mx_float oldx0 = v_proposals[v][f][0];
								mx_float oldx2 = v_proposals[v][f][2];
								v_proposals[v][f][0] = width - oldx2 - 1;
								v_proposals[v][f][2] = width - oldx0 - 1;
							}
						}
					}
					if (scale != 1.0f)
					{
						for (size_t v = 0; v < v_proposals.size(); v++)
						{
							for (size_t f = 0; f < v_proposals[v].size(); f++)
							{
								//restore to src resolution
								v_proposals[v][f][0] /= scale;
								v_proposals[v][f][1] /= scale;
								v_proposals[v][f][2] /= scale;
								v_proposals[v][f][3] /= scale;
							}
						}
					}
					for (size_t v = 0; v < imgs.size(); v++)
					{
						v_scores_list[v].insert(v_scores_list[v].end(), v_scores[v].begin(), v_scores[v].end());
					}
					for (size_t v = 0; v < imgs.size(); v++)
					{
						v_proposals_list[v].insert(v_proposals_list[v].end(), v_proposals[v].begin(), v_proposals[v].end());
					}

					//OK
					if (!vote&&use_landmarks)
					{
						idx++;
						vector<vector<vector<Point2f>>> v_landmarks;
						vector<mx_float> landmarks_data_t;
						vector<mx_uint> landmarks_data_shape = exec->outputs[idx].GetShape();
						exec->outputs[idx].SyncCopyToCPU(&landmarks_data_t, exec->outputs[idx].Size());
						offset_perimage = landmarks_data_shape[1] * landmarks_data_shape[2] * landmarks_data_shape[3];
						for (size_t v = 0; v < imgs.size(); v++)
						{
							vector<vector<Point2f>> landmark_deltas;
							vector<mx_float> landmarks_data(landmarks_data_t.begin() + v*offset_perimage, landmarks_data_t.begin() + (v + 1)*offset_perimage);
							for (size_t z = 0; z < landmarks_data.size() / 20; z++) //transpose reshape
							{
								vector<Point2f> landmarks_temp1;
								vector<Point2f> landmarks_temp2;
								for (size_t y = 0; y < 5; y++)
								{
									Point2f p1(landmarks_data[2 * y*K + z], landmarks_data[(2 * y + 1)*K + z]);
									landmarks_temp1.push_back(p1);
								}
								for (size_t y = 5; y < 10; y++)
								{
									Point2f p2(landmarks_data[2 * y*K + z], landmarks_data[(2 * y + 1)*K + z]);
									landmarks_temp2.push_back(p2);
								}

								landmark_deltas.push_back(landmarks_temp1);
								landmark_deltas.push_back(landmarks_temp2);
							}
							vector<vector<Point2f>> landmarks = landmark_pred(anchors, landmark_deltas);
							landmarks = selectByindex(landmarks, v_order[v]);
							v_landmarks.push_back(landmarks);
						}


						if (_flip)
						{
							for (size_t v = 0; v < v_landmarks.size(); v++)
							{
								for (size_t l = 0; l < v_landmarks[v].size(); l++)
								{
									for (size_t k = 0; k < 5; k++)
									{
										v_landmarks[v][l][k].x = width - v_landmarks[v][l][k].x - 1;
									}
								}
								for (size_t l = 0; l < v_landmarks[v].size(); l++)
								{
									Point2f p0(v_landmarks[v][l][0]);
									Point2f p1(v_landmarks[v][l][1]);
									Point2f p3(v_landmarks[v][l][3]);
									Point2f p4(v_landmarks[v][l][4]);
									v_landmarks[v][l][0] = p1;
									v_landmarks[v][l][1] = p0;
									v_landmarks[v][l][3] = p4;
									v_landmarks[v][l][4] = p3;
								}
							}

						}
						if (scale != 1.0)
						{
							for (size_t v = 0; v < v_landmarks.size(); v++)
							{
								for (size_t l = 0; l < v_landmarks[v].size(); l++)
								{
									for (size_t k = 0; k < 5; k++)
									{
										v_landmarks[v][l][k].x /= scale;
										v_landmarks[v][l][k].y /= scale;
									}
								}
							}

						}
						for (size_t v = 0; v < v_landmarks.size(); v++)
						{
							v_landmarks_list[v].insert(v_landmarks_list[v].end(), v_landmarks[v].begin(), v_landmarks[v].end());
						}
					}
				}
				delete[] data_img;
				data_img = nullptr;
				delete exec;
				exec = nullptr;
			}
		}
		for (size_t v = 0; v < v_proposals_list.size(); v++)
		{
			vector<vector<mx_float>> proposals_list = v_proposals_list[v];
			vector<mx_float> scores_list = v_scores_list[v];
			vector<vector<Point2f>> landmarks_list = v_landmarks_list[v];
			if (proposals_list.size() == 0)
			{
				continue;
			}
			vector<size_t> order = sort_indexes_e(scores_list, true);
			proposals_list = selectByindex(proposals_list, order);
			scores_list = selectByindex(scores_list, order);
			if (!vote&&use_landmarks)
			{
				landmarks_list = selectByindex(landmarks_list, order);
			}
			if (!vote)
			{
				vector<size_t> keep = nms(proposals_list, scores_list, nms_threshold);
				proposals_list = selectByindex(proposals_list, keep);
				scores_list = selectByindex(scores_list, keep);
				if (use_landmarks)
				{
					landmarks_list = selectByindex(landmarks_list, keep);
				}
				vector<Face> faces;
				for (size_t f = 0; f < proposals_list.size(); f++)
				{
					int x = proposals_list[f][0];
					int y = proposals_list[f][1];
					int w = proposals_list[f][2] - proposals_list[f][0];
					int h = proposals_list[f][3] - proposals_list[f][1];
					Rect facebox(x, y, w, h);
					mx_float score = scores_list[f];
					vector<Point2f> lankmarks;
					if (use_landmarks)
					{
						lankmarks = landmarks_list[f];
					}
					Face face;
					/*{ facebox, score, lankmarks };*/
					face.boundingbox = facebox;
					face.score = score;
					face.landmarks = lankmarks;
					faces.push_back(face);
				}
				result[v] = faces;
			}
		}
		return result;
	}
};
