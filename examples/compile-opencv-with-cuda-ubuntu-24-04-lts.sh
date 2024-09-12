#!/bin/bash

echo "NOTE: You will need to download the NVIDIA Video Codec SDK from https://developer.nvidia.com/nvidia-video-codec-sdk and unzip it into $HOME/builds/opencv_build/SDK"
echo "This is intended as a sort of note-to-self on Ubuntu 24.04 LTS."
echo "Use only if you understand what you're doing."

exit 0

# after getting the SDK package:
unzip ~/Downloads/Video_Codec_SDK_12.2.72.zip -d ~/builds/
sudo cp ~/builds/Video_Codec_SDK_12.2.72/Interface/* /usr/include/

# set to gcc 12
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12

# Install dependencies
sudo apt update
sudo apt install gcc-12 g++-12
sudo apt install build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbb2 libtbb-dev libdc1394-22-dev libopenblas-dev liblapacke-dev

# Clone OpenCV and OpenCV contrib
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

# Create build directory
mkdir -p opencv/build && cd opencv/build

# Run CMake configuration
cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_C_COMPILER=/usr/bin/gcc-12 \
      -D CMAKE_CXX_COMPILER=/usr/bin/g++-12 \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=1 \
      -D WITH_V4L=ON \
      -D BUILD_opencv_python3=ON \
      -D PYTHON3_EXECUTABLE=$(which python3) \
      -D CUDA_ARCH_BIN="8.6" \
      -D WITH_LAPACK=ON \
      -D LAPACK_LIBRARIES="/usr/lib/x86_64-linux-gnu/liblapack.so;/usr/lib/x86_64-linux-gnu/libblas.so" \
      -D LAPACK_INCLUDE_DIR="/usr/include;/usr/include/x86_64-linux-gnu" \
      -D CBLAS_INCLUDE_DIR="/usr/include/x86_64-linux-gnu/openblas-pthread" \
      -D BLAS_LIBRARIES="/usr/lib/x86_64-linux-gnu/libblas.so" \
      -D NVCUVID_HEADER_DIR=$HOME/builds/Video_Codec_SDK_12.2.72/Interface \
      -D NVENCODEAPI_HEADER_DIR=$HOME/builds/Video_Codec_SDK_12.2.72/Interface \
      -D CMAKE_VERBOSE_MAKEFILE=ON \
      ..

# Compile OpenCV
make -j$(nproc)

# Install OpenCV
sudo make install
sudo ldconfig
