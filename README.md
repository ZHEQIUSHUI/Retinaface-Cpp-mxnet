# Retinaface-Cpp-mxnet

## Update 2019.10.6
[Faster-Retinaface-Cpp-mxnet](https://github.com/ZHEQIUSHUI/Faster-RetinaFace-Cpp-mxnet):faster and lighter

## Update 2019.9.9
Fixed bugs with demo.the demo is write in github file edit...so you know...

## Update 2019.8.28
Upload Example....i forgot it at first...

## Update

debug:flip result

debug:when use_lankmarks=false,output extract idx is wrong. 

reduce packing and unboxing matrix times and time

No “vote” for the time being,i think it is not often to used,so i have no debug it,maybe.....Coming soon


## Model from deepinsight/insightface/RetinaFace

[deepinsight](https://github.com/deepinsight/insightface/tree/master/RetinaFace):

Pretrained Model: RetinaFace-R50 ([baidu cloud](https://pan.baidu.com/s/1C6nKq122gJxRhb37vK0_LQ) or [dropbox](https://www.dropbox.com/s/53ftnlarhyrpkg2/retinaface-R50.zip?dl=0)) is a medium size model with ResNet50 backbone.
It can output face bounding boxes and five facial landmarks in a single forward pass.

WiderFace validation mAP: Easy 96.5, Medium 95.6, Hard 90.4. 

To avoid the confliction with the WiderFace Challenge (ICCV 2019), we postpone the release time of our best model.

## Third-party Models

[yangfly](https://github.com/yangfly): 

RetinaFace-MobileNet0.25 ([baidu cloud](https://pan.baidu.com/s/1P1ypO7VYUbNAezdvLm2m9w)).
WiderFace validation mAP: Hard 82.5. (model size: 1.68Mb)



## Reference

opencv cpp lib

mxnet cpp lib

cuda 10.0

cudnn 7.5
