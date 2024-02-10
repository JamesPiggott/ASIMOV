# [ASIMOV](https://github.com/JamesPiggott/ASIMOV)

![License](https://img.shields.io/github/license/JamesPiggott/ASIMOV)

ASIMOV or Asymmetric Secure Isomorphic Verification is a Face Detection & Recognition application built using Python 3 and TensorFlow 2.0. It uses the latest versions of RetinaFace and ArcFace for detection and recognition. This repository is a trial version of the larger ASIMOV project and is intended for feedback and debugging.

## Installation

The installation is relatively simple, as there is no need to first create the models or convert them to the SavedModel format. 

```bash
git clone https://github.com/JamesPiggott/ASIMOV.git
cd ASIMOV
```

Afterwards install the required libraries defined in 'requirements.txt' using pip. Then check out the sample script 
located in the 'test' folder.

### CUDA

However, for the application to run optimally you will need to make use of a NVIDIA CUDA enabled GPU.  

#### CUDA on Linux

Linux is straightforward, it requires python, pip and Miniconda. For the complete procedure read the guide on [TensorFlow.org](https://www.tensorflow.org/install/pip).

```bash
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create virtual env
conda create --name asimov python=3.10
```

```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
pip install -r requirements.txt

# Verify install:
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

#### CUDA on Windows 10 / 11

Installation on Windows can be more problematic as for the CUDA drivers, toolkit and cuDNN libraries specific versions need to be used. There is an excellent article online, see link to [Towards Data Science](https://towardsdatascience.com/setting-up-tensorflow-gpu-with-cuda-and-anaconda-onwindows-2ee9c39b5c44) that specifies the procedure step-by-step. A simple alternative is to install the packages using conda. Install Microsoft Visual C++ Redistributable and Anaconda. Open `Anaconda Prompt` and enter the following commands:

```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
python -m pip install "tensorflow<2.10"
pip install -r requirements.txt

# Verify install:
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Note the specific requirement that the TensorFlow version should not exceed 2.9.

## Testing

There are several Python scripts built around the API intended for testing. These can be used as the basis for any 
application of your own. The first is `video_detection_test.py` which tests face detection using RetinaFace on a 
sample video. The second is `webcam_test.py` which does almost the same but uses as input your webcam. Finally, there 
is `image_recognition_test.py` which is a more comprehensive suite that tests cropping, alignment and face comparison. The latter of course uses the ArcFace model to create the necessary face vectors for comparison.

```bash
python video_detection_test.py
python webcam_test.py
python image_recognition_test.py
```

## References

The following resources were invaluable for creating the ASIMOV implementation, of which this repository is but a small 
sample. Regardless, credit should go to where it is due, as such my thanks to the maintainers of the following 
GitHub repo's.

- https://github.com/peteryuX/retinaface-tf2
    - RetinaFace for TensorFlow 2.0
- https://github.com/peteryuX/arcface-tf2
    - ArcFace for TensorFlow 2.0
