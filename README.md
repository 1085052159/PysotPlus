# PySOT_Plus
**My English is poor. And the docs may not be complete. I'm busy during this period. 
I'll improve the documents when I'm not busy. If you have any problems or find any bugs, 
please open an issue. If you like this project, please give me a little star.**
 
**PySOT_Plus** adds some contents to pysot to make it more convenient. Thank you very much for sensetime's open source. 
This is the **un-official implementation**. Based on their source codes, I have added the following contents.
1. Introduce the [SiamBAN](https://arxiv.org/pdf/2003.06761.pdf) into this source code
2. Introduce [MIXED PRECISION TRAINING]() into this source code
3. optimize toolkit to make it support the predicted txts of pytracking, pysot and official website. 
The txts with separator "," or "space(\ )" or "tab(\t)" works fine.
4. Introduce [optuna]() to do hyper-param finetune 
5. Add some shells to make it more convenient. Including train.sh(.py), test.sh(.py), eval.sh(.py), union.sh(.py).
 union.py: the integration of training, testing and evaluation
6. Add visdom tools to vis Heatmap  

## Introduction
Evaluation toolkit can support the following datasets:

:paperclip: [OTB2015](http://faculty.ucmerced.edu/mhyang/papers/pami15_tracking_benchmark.pdf) 
:paperclip: [VOT16/18/19](http://votchallenge.net) 
:paperclip: [VOT18-LT](http://votchallenge.net/vot2018/index.html) 
:paperclip: [LaSOT](https://arxiv.org/pdf/1809.07845.pdf) 
:paperclip: [UAV123](https://arxiv.org/pdf/1804.00518.pdf)
:paperclip: [GOT10K]()
:paperclip: [TrackingNet]()

Besides, you can also generate the results figure like Success Plots and Precision Plots. If you want to show the Plots,
please install latex.

## Model Zoo and Baselines

Please refer to the [PySOT Model Zoo](MODEL_ZOO.md).
The model is from official PySOT Model Zoo.

## Installation

Please find installation instructions for PyTorch and PySOT in [`INSTALL.md`](INSTALL.md).

## Quick Start: Using PySOT

### Add PySOT to your PYTHONPATH
```bash
export PYTHONPATH=/path/to/pysotplus:$PYTHONPATH
```

### Download models
Download models in [PySOT Model Zoo](MODEL_ZOO.md) and put the model.pth in the correct directory in experiments

### Webcam demo
```bash
python tools/demo.py \
    --config experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
    --snapshot experiments/siamrpn_r50_l234_dwxcorr/model.pth
    # --video demo/bag.avi # (in case you don't have webcam)
```

### Download testing datasets
Download datasets and put them into `testing_dataset` directory. Jsons of commonly used datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/10cfXjwQQBQeu48XMf2xc_W1LucpistPI) or [BaiduYun](https://pan.baidu.com/s/1js0Qhykqqur7_lNRtle1tA#list/path=%2F). If you want to test tracker on new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to setting `testing_dataset`. 

### Test&Eval tracker
```bash
cd shell
sh test.sh
```
The testing results will in the current directory(results/dataset/model_name/)


###  Training :wrench:
See [TRAIN.md](TRAIN.md) for detailed instruction.


### Getting Help :hammer:
If you meet problem, try searching our GitHub issues first. We intend the issues page to be a forum in which the community collectively troubleshoots problems. But please do **not** post **duplicate** issues. If you have similar issue that has been closed, you can reopen it.

- `ModuleNotFoundError: No module named 'pysot'`

:dart:Solution: Run `export PYTHONPATH=path/to/pysot` first before you run the code.

- `ImportError: cannot import name region`

:dart:Solution: Build `region` by `python setup.py build_ext —-inplace` as decribled in [INSTALL.md](INSTALL.md).


## References

- [PySOT](https://github.com/STVIR/pysot)

- [Siamese Box Adaptive Network for Visual Tracking](https://arxiv.org/pdf/2003.06761.pdf)
  CVPR, 2020

  
## Contributors

- myself

## License
If you think this project help you a lot. It's my honor.
Please cite this project if you haved used it.
I am a little busy, I have no time to finish the docs. If you have any problems about this project, please open an issue.

