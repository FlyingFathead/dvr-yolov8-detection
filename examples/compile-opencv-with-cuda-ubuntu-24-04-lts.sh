#!/bin/bash

echo "ATTENTION: You will need to download the NVIDIA Video Codec SDK from https://developer.nvidia.com/nvidia-video-codec-sdk and unzip it into $HOME/builds/opencv_build/SDK"
echo "This is intended as a sort of note-to-self on Ubuntu 24.04 LTS."
echo "Use only if you understand what you're doing."
echo ""
echo "NOTE: You _WILL_ have to change the CUDA level according to your Nvidia GPU's CUDA level specs."

exit 0

# after getting the SDK package:
unzip ~/Downloads/Video_Codec_SDK_12.2.72.zip -d ~/builds/
sudo cp ~/builds/Video_Codec_SDK_12.2.72/Interface/* /usr/include/

# NOTE: More than likely requires a separately compiled ffmpeg with -fPIC
# Find and compile it with i.e.:
# ./configure \
#     --enable-shared \
#     --enable-pic \
#     CFLAGS="-fPIC" \
#     CXXFLAGS="-fPIC" \
#     --prefix=$HOME/builds/ffmpeg

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

# set to gcc 11
export CC=/usr/bin/gcc-11 &&
export CXX=/usr/bin/g++-11 &&
export CXXFLAGS="-std=c++17"

# # v0.3
CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 cmake \
  -D CMAKE_BUILD_TYPE=RELEASE \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D CMAKE_C_COMPILER=/usr/bin/gcc-11 \
  -D CMAKE_CXX_COMPILER=/usr/bin/g++-11 \
  -D INSTALL_PYTHON_EXAMPLES=ON \
  -D INSTALL_C_EXAMPLES=OFF \
  -D OPENCV_ENABLE_NONFREE=ON \
  -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
  -D PYTHON_EXECUTABLE=$(which python3) \
  -D BUILD_EXAMPLES=ON \
  -D WITH_CUDA=ON \
  -D WITH_CUDNN=ON \
  -D OPENCV_DNN_CUDA=ON \
  -D ENABLE_FAST_MATH=1 \
  -D CUDA_FAST_MATH=1 \
  -D CUDA_ARCH_BIN=$cuda_arch_bin \
  -D CUDA_ARCH_PTX=$cuda_arch_bin \
  -D CUDA_INCLUDE_DIRS=/usr/local/cuda/targets/x86_64-linux/include \
  -D CMAKE_INCLUDE_PATH=/usr/local/cuda/targets/x86_64-linux/include \
  -D WITH_CUBLAS=1 \
  -D WITH_GSTREAMER=ON \
  -D WITH_LIBV4L=ON \
  -D BUILD_opencv_python3=ON \
  -D PYTHON3_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])") \
  -D PYTHON3_PACKAGES_PATH=$(python3 -c "import site; print(site.getsitepackages()[0])") \
  -D OpenCL_LIBRARY=/usr/local/cuda-12.4/lib64/libOpenCL.so \
  -D OpenCL_INCLUDE_DIR=/usr/local/cuda-12.4/include \
  -D BUILD_opencv_java=OFF \
  -D BUILD_TESTS=OFF \
  -D BUILD_PERF_TESTS=OFF \
  -D BUILD_DOCS=OFF \
  -D BUILD_EXAMPLES=ON \
  -D BUILD_opencv_apps=ON \
  -D BUILD_opencv_ts=ON \
  -D BUILD_opencv_rgbd=OFF \
  -D BUILD_SHARED_LIBS=ON \
  -D WITH_OPENGL=ON \
  -D WITH_OPENCL=ON \
  -D WITH_IPP=ON \
  -D WITH_TBB=ON \
  -D WITH_EIGEN=ON \
  -D WITH_V4L=ON \
  -D WITH_QT=ON \
  -D WITH_GLEW=ON \
  -D FFMPEG_INCLUDE_DIR=$HOME/builds/ffmpeg/include \
  -D FFMPEG_LIBRARIES="$HOME/builds/ffmpeg/lib/libavcodec.a;$HOME/builds/ffmpeg/lib/libavformat.a;$HOME/builds/ffmpeg/lib/libavutil.a;$HOME/builds/ffmpeg/lib/libswscale.a;$HOME/builds/ffmpeg/lib/libswresample.a" \
  -D AVRESAMPLE_INCLUDE_DIR=/usr/include/x86_64-linux-gnu \
  -D AVRESAMPLE_LIBRARIES=/usr/lib/x86_64-linux-gnu/libswresample.so \
  -D GLEW_LIBRARY=/usr/lib/x86_64-linux-gnu/libGLEW.so \
  -D NVCUVID_HEADER_DIR=$NVCUVID_HEADER_DIR \
  -D NVENCODEAPI_HEADER_DIR=$NVENCODEAPI_HEADER_DIR \
  -D CUVIDDEC_HEADER_DIR=$CUVIDDEC_HEADER_DIR \
  -D BLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/blas/libblas.so \
  -D BLAS_INCLUDE_DIRS=/usr/include/x86_64-linux-gnu \
  -D TBB_LIBRARIES="/usr/lib/x86_64-linux-gnu/libtbb.so;/usr/lib/x86_64-linux-gnu/libtbbmalloc.so;/usr/lib/x86_64-linux-gnu/libtbbmalloc_proxy.so" \
  -D TBB_INCLUDE_DIRS=/usr/include/tbb \
  -D OpenBLAS_INCLUDE_DIR=/usr/include/x86_64-linux-gnu \
  -D OpenBLAS_LIB=/usr/lib/x86_64-linux-gnu/libopenblas.so \
  -D OpenGL_GL_PREFERENCE=GLVND \
  -D LAPACK_INCLUDE_DIRS=/usr/include \
  -D LAPACK_LIBRARIES=/usr/lib/x86_64-linux-gnu/liblapack.so \
  -D LAPACKE_INCLUDE_DIRS=/usr/include \
  -D LAPACKE_LIBRARIES=/usr/lib/x86_64-linux-gnu/liblapacke.so \
  -D CMAKE_CXX_STANDARD=17 ..

# # v0.1
# # Run CMake configuration
# cmake -D CMAKE_BUILD_TYPE=Release \
#       -D CMAKE_C_COMPILER=/usr/bin/gcc-12 \
#       -D CMAKE_CXX_COMPILER=/usr/bin/g++-12 \
#       -D CMAKE_INSTALL_PREFIX=/usr/local \
#       -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
#       -D WITH_CUDA=ON \
#       -D WITH_CUDNN=ON \
#       -D OPENCV_DNN_CUDA=ON \
#       -D ENABLE_FAST_MATH=1 \
#       -D CUDA_FAST_MATH=1 \
#       -D WITH_CUBLAS=1 \
#       -D WITH_V4L=ON \
#       -D BUILD_opencv_python3=ON \
#       -D PYTHON3_EXECUTABLE=$(which python3) \
#       -D CUDA_ARCH_BIN="8.6" \
#       -D WITH_LAPACK=ON \
#       -D LAPACK_LIBRARIES="/usr/lib/x86_64-linux-gnu/liblapack.so;/usr/lib/x86_64-linux-gnu/libblas.so" \
#       -D LAPACK_INCLUDE_DIR="/usr/include;/usr/include/x86_64-linux-gnu" \
#       -D CBLAS_INCLUDE_DIR="/usr/include/x86_64-linux-gnu/openblas-pthread" \
#       -D BLAS_LIBRARIES="/usr/lib/x86_64-linux-gnu/libblas.so" \
#       -D NVCUVID_HEADER_DIR=$HOME/builds/Video_Codec_SDK_12.2.72/Interface \
#       -D NVENCODEAPI_HEADER_DIR=$HOME/builds/Video_Codec_SDK_12.2.72/Interface \
#       -D CMAKE_VERBOSE_MAKEFILE=ON \
#       ..

# Compile OpenCV
make -j$(nproc)

# Install OpenCV
sudo make install
sudo ldconfig
