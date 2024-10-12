# Use NVIDIA CUDA base image with Ubuntu 22.04
# FROM nvidia/cuda:12.4.0-base-ubuntu22.04
# FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# nvidia-cuda-toolkit \
# Add NVIDIA’s repo key and set up the proper repository for the latest CUDA versions
# Get `wget` first
RUN apt-get update && apt-get install -y --no-install-recommends wget

# Then, run the keyring in
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    nvidia-cuda-dev \
    nvidia-cuda-gdb \
    gcc-10 g++-10 \
    build-essential \
    libcudnn9-cuda-12 \
    libcudnn9-dev-cuda-12 \
    cuda-toolkit-12-4 \
    cmake \
    git \
    wget \
    unzip \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    libgl1 \
    python3-dev \    
    python3-pip \
    espeak-ng \
    libespeak-ng1 \    
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip for Python 3.12
RUN python3 -m pip install --upgrade pip

# Install Python dependencies
RUN python3 -m pip install --no-cache-dir numpy

# Define OpenCV version
ENV OPENCV_VERSION=4.10.0

# Create a directory for OpenCV
WORKDIR /opt/opencv_build

# Download OpenCV and OpenCV_contrib
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
RUN wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip && \
RUN unzip opencv.zip
RUN unzip opencv_contrib.zip
RUN rm opencv.zip
RUN rm opencv_contrib.zip
RUN mv opencv-${OPENCV_VERSION} opencv
RUN mv opencv_contrib-${OPENCV_VERSION} opencv_contrib

# Create build directory
WORKDIR /opt/opencv_build/opencv/build

# Configure OpenCV with CUDA and set GCC/G++ 10
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_C_COMPILER=gcc-10 \
    -D CMAKE_CXX_COMPILER=g++-10 \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_build/opencv_contrib/modules \
    -D WITH_CUDA=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D BUILD_EXAMPLES=OFF \
    -D WITH_GSTREAMER=ON \
    -D WITH_LIBV4L=ON \
    -D WITH_QT=ON \
    -D WITH_OPENGL=ON \
    -D BUILD_opencv_python3=ON \
    -D BUILD_TESTS=OFF \      
    ..

# Build OpenCV
RUN make -j$(nproc)
RUN make install
RUN rm -rf /opt/opencv_build
RUN ldconfig

RUN python3 -m pip install --upgrade pip

# Install YOLOv8 and other Python dependencies
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify OpenCV installation and CUDA support in a single RUN command
RUN echo "==============================================================" && \
    echo "If OpenCV compilation was successful, it should show up below:" && \
    echo "==============================================================" && \
    python3 -c "import cv2; print('OpenCV Version:', cv2.__version__)" && \
    python3 -c "import cv2; print('CUDA Enabled Devices:', cv2.cuda.getCudaEnabledDeviceCount())" && \
    echo "==============================================================" && \
    echo "Verification completed at $(date)"

# Create a working directory for the application
WORKDIR /app

# Copy your application code into the container
COPY . /app

# install repository requirements
RUN pip install --no-cache-dir -r requirements.txt

# Expose any necessary ports (if applicable)
# Example: EXPOSE 8000

# Define the entrypoint or command

# Run the detection with GUI (requires x11docker or a X11 passthrough method!)
# CMD ["python3", "yolov8_live_rtmp_stream_detection.py"]

# Run headless (unless you're configuring X11 passthrough, this might be easier)
CMD ["python3", "yolov8_live_rtmp_stream_detection.py", "--headless"]
