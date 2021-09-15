# Installation

This document contains detailed instructions for installing dependencies for PySOT_Plus. 
The code is tested on an Ubuntu 18.04 system with Nvidia GPU. Windows 10 is ok but not recommend.

### Requirments
* Conda with Python >= 3.6
* Nvidia GPU.
* PyTorch >= 0.4.1
* yacs
* pyyaml
* matplotlib
* tqdm
* OpenCV
* visdom
* apex

## Step-by-step instructions

#### Create environment and activate
```bash
conda create --name pysot python=3.7
conda activate pysot
```

#### Install numpy/pytorch/opencv/visdom
install [pytorch](https://pytorch.org/)
```
conda install numpy
pip install opencv-python
pip install visdom
```

#### Install other requirements
```
pip install pyyaml yacs tqdm colorama matplotlib cython tensorboardX
```

#### Build extensions
```
python setup.py build_ext --inplace
```

### Install Apex
```
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install --cuda_ext --cpp_ext
```




## Try with scripts
```
bash install.sh /path/to/your/conda pysot (untested)
```
