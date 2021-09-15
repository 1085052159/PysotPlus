# PySOT Training Tutorial

This implements training of SiamRPN with backbone architectures, such as ResNet, AlexNet.
### Add PySOT to your PYTHONPATH
```bash
export PYTHONPATH=/path/to/pysotplus:$PYTHONPATH
```

## Prepare training dataset
Prepare training dataset, detailed preparations are listed in [training_dataset](training_dataset) directory.
* [VID](http://image-net.org/challenges/LSVRC/2017/)
* [YOUTUBEBB](https://research.google.com/youtube-bb/) (New link for cropped data, [BaiduYun](https://pan.baidu.com/s/1nXe6cKMHwk_zhEyIm2Ozpg), extract code: h964. **NOTE: Data in old link is not correct. Please use cropped data in this new link.**)
* [DET](http://image-net.org/challenges/LSVRC/2017/)
* [COCO](http://cocodataset.org)
* [GOT10K]()

## Download pretrained backbones
The pretrained models are in the "pretrained_models" directory

## Training & Testing & Evaling
To train a model (SiamRPN++), run `union.sh`