#!/bin/bash
# install_and_compile_opencv_with_cuda.sh
#
# >> June 22, 2024 <<
#
# This script installs and compiles OpenCV from git source with CUDA support.
# (Or at least attempts to.)
#
# Requires Debian/Ubuntu-based Linux to run.

# options
SKIP_INSTALL_PACKAGES=false

# Parse command line arguments
while getopts "s" opt; do
  case ${opt} in
    s )
      SKIP_INSTALL_PACKAGES=true
      ;;
    \? )
      echo "Usage: cmd [-s]"
      exit 1
      ;;
  esac
done

# log this 
LOGFILE="opencv_compile_install.log"
exec > >(tee -i $LOGFILE)
exec 2>&1

# hz line function
function viivo() {
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
}

viivo &&

# Ensure proper path for CUDA and OpenCL
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
echo "CUDA and OpenCL paths added to PATH and LD_LIBRARY_PATH."

# Verify which libOpenCL is being used
echo "Checking linked libraries for your binary:"
ldd path/to/your/binary | grep libOpenCL

# Check available libOpenCL libraries
echo "Available libOpenCL libraries:"
ldconfig -p | grep libOpenCL

# Inspect specific library versions
echo "Inspecting library versions:"
readelf -d /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 | grep SONAME
readelf -d /usr/local/cuda/lib64/libOpenCL.so.1 | grep SONAME

# Function to check critical version compatibility
function check_versions() {
    local min_gcc_version="11.4"
    local min_python_version="3.10"
    local min_cuda_version="11.5"
    local min_nvidia_driver_version="550.90.07"

    echo "Checking system compatibility..."

    # Check GCC version compatibility
    gcc_version=$(gcc -dumpfullversion)  # This should fetch the full version.
    if [[ "$(printf '%s\n' "$min_gcc_version" "$gcc_version" | sort -V | head -n1)" != "$min_gcc_version" ]]; then
        echo "GCC version mismatch. Found: $gcc_version, required: $min_gcc_version or newer."
        return 1
    fi

    # Check Python version compatibility
    python_version=$(python3 --version | cut -d ' ' -f2)
    if [[ "$(printf '%s\n' "$min_python_version" "$python_version" | sort -V | head -n1)" != "$min_python_version" ]]; then
        echo "Python version mismatch. Found: $python_version, required: $min_python_version or newer."
        return 1
    fi

    # Check CUDA version compatibility
    cuda_version=$(nvcc --version | grep release | cut -d ',' -f2 | cut -d ' ' -f3)
    if [[ "$(printf '%s\n' "$min_cuda_version" "$cuda_version" | sort -V | head -n1)" != "$min_cuda_version" ]]; then
        echo "CUDA version mismatch. Found: $cuda_version, required: $min_cuda_version or newer."
        return 1
    fi

    # Check NVIDIA driver version compatibility
    nvidia_driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | uniq)
    if [[ "$(printf '%s\n' "$min_nvidia_driver_version" "$nvidia_driver_version" | sort -V | head -n1)" != "$min_nvidia_driver_version" ]]; then
        echo "NVIDIA driver version mismatch. Found: $nvidia_driver_version, required: $min_nvidia_driver_version or newer."
        return 1
    fi

    echo "All version checks passed. System is compatible."
    return 0
}

function verify_libraries() {
    libraries=(
        "/usr/lib/x86_64-linux-gnu/libGL.so"
        "/usr/lib/x86_64-linux-gnu/libGLU.so"
        "/usr/lib/x86_64-linux-gnu/libGLEW.so"
        "/usr/lib/x86_64-linux-gnu/libGLX.so"
    )

    missing_libraries=()

    for lib in "${libraries[@]}"; do
        if [ ! -f "$lib" ]; then
            missing_libraries+=("$lib")
        fi
    done

    if [ ${#missing_libraries[@]} -ne 0 ]; then
        echo "The following libraries are missing:"
        for lib in "${missing_libraries[@]}"; do
            echo "$lib"
        done
        exit 1
    else
        echo "All required libraries are present."
    fi
}

# Run system checks
check_versions || exit 1

# Verify required libraries
verify_libraries || exit 1

# Ensure proper path for CUDA
if [ -d "/usr/local/cuda/bin" ] && [ -d "/usr/local/cuda/lib64" ]; then
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    echo "CUDA paths added to PATH and LD_LIBRARY_PATH."
else
    echo "CUDA directories not found, please check your CUDA installation."
    exit 1
fi

function check_debian_based() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        if [[ "$ID_LIKE" == *"debian"* ]] || [[ "$ID" == *"debian"* ]] || [[ "$ID_LIKE" == *"ubuntu"* ]] || [[ "$ID" == *"ubuntu"* ]]; then
            echo "Debian-based Linux distribution detected: $NAME"
        else
            echo "This script requires a Debian-based Linux distribution (e.g., Debian, Ubuntu)."
            exit 1
        fi
    else
        echo "/etc/os-release file not found. Unable to determine the Linux distribution."
        exit 1
    fi
}

function get_cuda_arch_bin() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "nvidia-smi could not be found, make sure NVIDIA drivers are installed and nvidia-smi is in your PATH"
        exit 1
    fi

    # Get the compute capability of the GPU
    cuda_arch_bin=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)

    if [ -z "$cuda_arch_bin" ]; then
        echo "Unable to determine CUDA Compute Capability. Ensure that you have a compatible NVIDIA GPU and CUDA installed."
        exit 1
    fi

    echo "Detected CUDA Compute Capability: $cuda_arch_bin"
    echo "Using CUDA_ARCH_BIN=$cuda_arch_bin for building OpenCV"
}

function get_nvidia_driver_version() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "nvidia-smi could not be found, make sure NVIDIA drivers are installed and nvidia-smi is in your PATH"
        exit 1
    fi

    driver_version=$(nvidia-smi --version | grep "Driver Version" | awk '{print $4}' | cut -d '.' -f 1)

    if [ -z "$driver_version" ]; then
        echo "Unable to determine NVIDIA driver version. Ensure that you have a compatible NVIDIA driver installed."
        exit 1
    fi

    echo "Detected NVIDIA driver version: $driver_version"
}

function install_packages() {
    driver_version=$1
    packages=(
        build-essential cmake git unzip pkg-config libjpeg-dev libpng-dev libtiff-dev
        libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev
        libgtk-3-dev libatlas-base-dev gfortran python3-dev python3-numpy libtbb2 libtbb-dev libdc1394-dev
        qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools libqt5x11extras5-dev
        libqt6core5compat6 libqt6core5compat6-dev
        libnvidia-decode-${driver_version} libnvidia-encode-${driver_version}
        gcc-11 g++-11 
        opencl-headers mesa-common-dev
        libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev
    )

    failed_packages=()

    for pkg in "${packages[@]}"; do
        echo "Installing $pkg..."
        if ! sudo apt-get install -y "$pkg"; then
            echo "Failed to install $pkg"
            failed_packages+=("$pkg")
        fi
    done

    if [ ${#failed_packages[@]} -ne 0 ]; then
        echo "The following packages failed to install:"
        for pkg in "${failed_packages[@]}"; do
            echo "$pkg"
        done
        exit 1
    fi
}

function ensure_tool_installed() {
    tool=$1
    if ! command -v $tool &> /dev/null; then
        echo "$tool is not installed. Please install $tool and re-run the script."
        exit 1
    fi
}

function update_or_clone_repo() {
    repo_url=$1
    repo_dir=$2
    repo_name=$(basename "$repo_url" .git)
    
    if [ -d "$repo_dir/$repo_name" ]; then
        echo "Updating $repo_name..."
        cd "$repo_dir/$repo_name" && git pull && cd -
    else
        echo "Cloning $repo_name..."
        git clone "$repo_url" "$repo_dir/$repo_name"
    fi
}

function check_nv_sdk_headers() {
    if [ ! -f /usr/local/cuda/NVIDIA-Video-SDK/include/nvcuvid.h ] || [ ! -f /usr/local/cuda/NVIDIA-Video-SDK/include/nvEncodeAPI.h ]; then
        echo "NVIDIA Video Codec SDK headers not found. Please download and install them from the NVIDIA website."
        echo "Download from: https://developer.nvidia.com/nvidia-video-codec-sdk"
        echo "Extract the contents to /usr/local/cuda/NVIDIA-Video-SDK"
        exit 1
    fi
}

function set_nv_sdk_paths() {
    export NVCUVID_HEADER_DIR="/usr/local/cuda/NVIDIA-Video-SDK/include"
    export NVENCODEAPI_HEADER_DIR="/usr/local/cuda/NVIDIA-Video-SDK/include"
}

# Introductory message
viivo &&
echo "FYI: This script attempts to compile and install OpenCV from source with CUDA support."
echo ""
echo "- It's *NOT* recommended to run this script if you have no idea as to what you are doing,"
echo "- This is literally a 'works on my PC' type of script."
echo "- This is NOT intended for anything else except as a compiling assistance."
echo "- There's nowhere to send bug reports on this to."
echo ""
echo "Please ensure you have the necessary permissions and that your system meets the prerequisites."
echo "We're installing system-wide packages with 'apt-get' and also installing OpenCV with 'sudo'."
echo ""
echo "AGAIN: You have been warned, this is a system-/build-specific build and install script."
echo "IF YOU USE THIS SCRIPT; YOU DO SO COMPLETELY AT YOUR OWN RISK. "
echo ""
echo "This script has been tested on the following configuration:"
echo "- Ubuntu Linux 22.04.4 LTS (Jammy Jellyfish)"
echo "- x86_64 / 30-series Nvidia RTX"
echo "- in June 2024"
echo ""
echo "NOTE: Everything might've changed since."
viivo &&

# Confirmation prompt to continue
echo "Are you absolutely sure you want to continue? This script is like a wild roller coaster ride: thrilling but potentially dangerous. Type 'yes' if you have a strong stomach or 'no' to chicken out. Again, you have been warned."
read -r confirmation
if [[ "$confirmation" != "yes" ]]; then
    echo "Wise choice! Exiting the script now. Maybe next time, champ!"
    exit 0
fi

# Ensure essential tools are installed
ensure_tool_installed "git"
ensure_tool_installed "cmake"
ensure_tool_installed "make"

# Check if the OS is Debian-based
check_debian_based

# Get the CUDA Compute Capability
viivo
get_cuda_arch_bin

# Get NVIDIA driver version
viivo
driver_version=$(get_nvidia_driver_version)

# Install the apt-get prerequisites
if [ "$SKIP_INSTALL_PACKAGES" = false ]; then
    viivo
    echo "::: Installing prerequisites via apt-get... NOTE: This portion requires 'sudo'."
    viivo
    install_packages $driver_version
else
    echo "::: Skipping package installation as per user request."
fi

# Update or clone the repositories
viivo
echo "::: Updating or cloning OpenCV and OpenCV Contrib repositories..."
viivo
update_or_clone_repo "https://github.com/opencv/opencv.git" "." &&
update_or_clone_repo "https://github.com/opencv/opencv_contrib.git" "."

# Check if the build directory already exists and offer to delete it
viivo &&
if [ -d "opencv/build" ]; then
    echo "The build directory already exists and must therefor be deleted before continuing."
    read -p "Do you want to delete it and continue? (Y/n): " choice
    case "$choice" in 
        y|Y ) 
            echo "Deleting the build directory..."
            rm -rfv opencv/build
            ;;
        n|N ) 
            echo "Installation cannot continue. Exiting."
            exit 1
            ;;
        * ) 
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac
fi

# Create a build directory and enter it
viivo &&
echo "::: Creating build directory..."
mkdir -p opencv/build && cd opencv/build
viivo &&

# Function to check CUDA version using nvidia-smi
function check_cuda_version() {
    required_version="12.4"
    cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')

    if [ "$cuda_version" != "$required_version" ]; then
        echo "CUDA version $cuda_version is installed, but version $required_version is required."
        exit 1
    else
        echo "CUDA version $cuda_version is compatible."
    fi
}

# Function to check GCC version
function check_gcc_version() {
    required_version="11.4.0"
    gcc_version=$(gcc --version | head -n1 | awk '{print $3}')

    if [ "$gcc_version" != "$required_version" ]; then
        echo "GCC version $gcc_version is installed, but version $required_version is required."
        exit 1
    else
        echo "GCC version $gcc_version is compatible."
    fi
}

# Check versions
# check_cuda_version
# check_gcc_version

# # If both checks pass, proceed with building OpenCV
# echo "Both CUDA and GCC versions are compatible. Proceeding with OpenCV build..."
# viivo &&

# Run CMake configuration with updated flags
viivo &&
echo "Running CMake configuration..." &&
viivo &&
if ! CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 cmake \
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
           -D WITH_CUBLAS=1 \
           -D WITH_GSTREAMER=ON \
           -D WITH_LIBV4L=ON \
           -D BUILD_opencv_python3=ON \
           -D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
           -D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
           -D OpenCL_LIBRARY=/lib/x86_64-linux-gnu/libOpenCL.so \
           -D BUILD_opencv_java=OFF \
           -D BUILD_TESTS=OFF \
           -D BUILD_PERF_TESTS=OFF \
           -D BUILD_DOCS=OFF \
           -D BUILD_EXAMPLES=ON \
           -D BUILD_opencv_apps=ON \
           -D BUILD_opencv_ts=ON \
           -D BUILD_SHARED_LIBS=ON \
           -D WITH_OPENGL=ON \
           -D WITH_OPENCL=ON \
           -D WITH_IPP=ON \
           -D WITH_TBB=ON \
           -D WITH_EIGEN=ON \
           -D WITH_V4L=ON \
           -D WITH_QT=ON \
           -D WITH_GLEW=ON \
           -D GLEW_LIBRARY=/usr/lib/x86_64-linux-gnu/libGLEW.so \
           -D OPENCV_GENERATE_PKGCONFIG=ON \
           -D CUDA_HOST_COMPILER=/usr/bin/gcc-11 \
           -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
           -D CUDA_INCLUDE_DIRS=/usr/local/cuda/include \
           -D OpenGL_GL_PREFERENCE=GLVND \
           -D CMAKE_CXX_FLAGS="-Wno-deprecated-declarations -ftemplate-depth=1024 -Wno-error=deprecated-declarations -Wno-error=unused-parameter -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++17" \
           -D CUDA_NVCC_FLAGS="-std=c++17 -Xcompiler -Wno-deprecated-declarations -Xcompiler -Wno-class-memaccess -D_FORCE_INLINES --expt-relaxed-constexpr -Wno-deprecated-gpu-targets" \
           -D CMAKE_C_FLAGS="-Wno-error=deprecated-declarations -Wno-error=unused-parameter" \
           -D CMAKE_CXX_STANDARD=17 \
           -D OpenGL_GL_LIBRARIES="/usr/lib/x86_64-linux-gnu/libGL.so;/usr/lib/x86_64-linux-gnu/libGLU.so;/usr/lib/x86_64-linux-gnu/libGLEW.so;/usr/lib/x86_64-linux-gnu/libGLX.so" ..; then
    echo "CMake configuration failed."
    exit 1
fi

# Compile OpenCV
viivo
echo "::: Compiling OpenCV... This might take a while."
viivo
if ! make -j$(nproc); then
    echo "OpenCV compilation failed."
    exit 1
fi

# Install OpenCV with confirmation prompt
viivo 
echo "::: Ready to install OpenCV. Do you want to proceed? (yes/no)"
read -r response
if [[ "$response" == "yes" ]]; then
    viivo
    echo "Installing OpenCV..."
    viivo
    if ! sudo make install; then
        echo "OpenCV installation failed."
        exit 1
    fi
    sudo ldconfig
else
    echo "Installation aborted by the user."
    exit 0
fi

# Success message
viivo
echo "::: OpenCV installed with CUDA support."
viivo
cd ../../
