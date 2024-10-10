## 🐳 Dockerized Setup 🐳

Setting up `dvr-yolov8-detection` using Docker streamlines the deployment process, ensuring a consistent environment across different systems. This section guides you through building and running the Docker container, leveraging GPU acceleration for optimal performance.

### 🔧 **Prerequisites**

Before proceeding, ensure the following are installed and configured on your system:

1. **Docker Engine**
   - **Installation Guide:** [Docker Installation](https://docs.docker.com/engine/install/)

2. **NVIDIA Drivers**
   - **Ensure** that your system has the latest NVIDIA drivers installed to support CUDA GPU acceleration.
   - **Installation Guide:** [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)

3. **NVIDIA Container Toolkit**
   - Enables Docker to access the GPU resources on the host machine.
   - **Installation Steps:**
     ```bash
     # Add the package repositories
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/stable/$distribution/nvidia-container-toolkit.list | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
     
     # Install the NVIDIA Container Toolkit
     sudo apt-get update
     sudo apt-get install -y nvidia-docker2
     
     # Restart the Docker daemon to apply changes
     sudo systemctl restart docker
     ```
   - **Reference:** [NVIDIA Docker Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

4. **GitHub Repository Clone**
   - Clone the `dvr-yolov8-detection` repository to your local machine:
     ```bash
     git clone https://github.com/FlyingFathead/dvr-yolov8-detection.git
     cd dvr-yolov8-detection
     ```

### 🛠️ **Building the Docker Image**

1. **Navigate to the Project Directory:**
   Ensure you're in the root directory of the cloned repository where the `Dockerfile` is located.
   ```bash
   cd dvr-yolov8-detection
   ```

2. **Build the Docker Image:**
   Execute the following command to build the Docker image named `yolov8_detection:latest`:
   ```bash
   docker build -t yolov8_detection:latest .
   ```
   - **Explanation:**
     - `docker build`: Command to build a Docker image from a Dockerfile.
     - `-t yolov8_detection:latest`: Tags the image with the name `yolov8_detection` and the tag `latest`.
     - `.`: Specifies the current directory as the build context.

   - **Note:** The build process may take some time, especially when compiling OpenCV with CUDA support.

If you're missing the GUI part in your Dockerized setup guide, the section on Docker should include specific instructions on enabling GUI applications within the container. Since Docker doesn't natively support GUI applications easily without some additional setup (especially on systems that require GPU passthrough), you could add a section that discusses how to run GUI-based applications inside the container using something like X11 or X11Docker.

Here’s a quick update you could make to include the GUI setup:

### 🖥️ **Running the GUI Applications (OpenCV GUI)**

To display OpenCV GUI windows from within the Docker container, you will need to configure X11 forwarding or use a tool like **x11docker** to handle GUI applications securely. Here's how to set it up:

#### **Option 1: X11 Forwarding (Linux)**

1. **Install X11 dependencies:**
   Ensure you have `xhost` installed on your host system:
   ```bash
   sudo apt-get install x11-xserver-utils
   ```

2. **Allow Docker to use your X11 display:**
   Run the following command to permit access to your display:
   ```bash
   xhost +local:docker
   ```

3. **Run the Docker container with X11 forwarding:**
   You can pass the display environment variable and mount the X11 socket to enable GUI windows:
   ```bash
   docker run --gpus all --network=host --rm \
     -e DISPLAY=$DISPLAY \
     -v /tmp/.X11-unix:/tmp/.X11-unix \
     yolov8_detection:latest
   ```

#### **Option 2: Using `x11docker` for Enhanced Security**

1. **Install `x11docker`:**
   x11docker provides a secure way to run GUI applications from within Docker containers:
   ```bash
   sudo apt-get install x11docker
   ```

2. **Run the container using `x11docker`:**
   With `x11docker`, you can run the container while forwarding the display securely:
   ```bash
   x11docker --gpu --hostdisplay yolov8_detection:latest
   ```

   - **Flags Explained:**
     - `--gpu`: Grants the container access to your GPU.
     - `--hostdisplay`: Shares your host's X11 display with the container.

   **Note:** x11docker ensures better isolation and security for running GUI applications inside Docker.

#### **Option 3: Running GUI on Windows (WSL2)**

If you're using **Docker Desktop** on Windows with **WSL2**, you will need a third-party X server like **VcXsrv** to forward the display. Install VcXsrv, run it, and configure Docker to use the X server by setting the display environment variable in your Docker run command:
```bash
docker run --gpus all --network=host --rm \
  -e DISPLAY=host.docker.internal:0.0 \
  yolov8_detection:latest
```

Make sure VcXsrv is running before executing the container.

---

### 🚀 **Running the Docker Container**

You have two options to run the Docker container: **Manually** or using the provided **`run_dockerized.sh` script**.

#### **Option 1: Manual Execution**

1. **Run the Container:**
   ```bash
   docker run --gpus all --network=host --rm yolov8_detection:latest
   ```
   - **Flags Explained:**
     - `--gpus all`: Grants the container access to all available GPUs.
     - `--network=host`: Shares the host's network stack with the container, allowing seamless access to services like RTMP servers running on `localhost`.
     - `--rm`: Automatically removes the container once it stops, keeping your system clean.

2. **Mount Configuration and Output Directories (Optional):**
   To customize configurations or persist detection outputs outside the container, use volume mounts:
   
   ```bash
   docker run --gpus all --network=host --rm \
     -v ./config.ini:/app/config.ini \
     -v ./yolo_detections:/app/yolo_detections \
     -v ./logs:/app/logs \
     yolov8_detection:latest
   ```

   - **Flags Explained:**
     - `-v ./config.ini:/app/config.ini`: Mounts your custom `config.ini` into the container.
     - `-v ./yolo_detections:/app/yolo_detections`: Persists detection images to the host.
     - `-v ./logs:/app/logs`: Stores log files on the host for easy access.

You can also try running the application with X11 forwarding on Linux systems:

   ```bash
   docker run --gpus all --network=host --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
     -v ./config.ini:/app/config.ini \
     -v ./yolo_detections:/app/yolo_detections \
     -v ./logs:/app/logs \
     yolov8_detection:latest
   ```

#### **Option 2: Using the `run_dockerized.sh` Script**

The `run_dockerized.sh` script automates the process of checking prerequisites and running the Docker container with the appropriate configurations.

1. **Ensure the Script is Executable:**
   ```bash
   chmod +x run_dockerized.sh
   ```

2. **Run the Script:**
   ```bash
   ./run_dockerized.sh
   ```
   - **Script Functionality:**
     - **Docker Installation Check:** Verifies if Docker is installed.
     - **Docker Service Status Check:** Ensures the Docker daemon is running.
     - **User Permissions Check:** Determines if the current user is part of the `docker` group to run Docker commands without `sudo`.
     - **Docker Image Availability Check:** Checks if the `yolov8_detection:latest` image exists locally; prompts to build it if not found.
     - **Container Execution:** Runs the Docker container with GPU access and host networking.

   - **Advantages:**
     - **Automated Checks:** Reduces manual verification steps.
     - **User-Friendly:** Provides clear messages and prompts for user actions.
     - **Enhanced Security:** Handles permission nuances seamlessly.

### ⚙️ **Configuration**

`dvr-yolov8-detection` utilizes a `config.ini` file to manage various operational parameters. Here's how to integrate it with your Docker setup:

1. **Locate or Create `config.ini`:**
   - If you have a custom `config.ini`, ensure it's prepared with your desired settings.
   - If not, refer to the default configuration provided in the repository and modify as needed.

2. **Mount `config.ini` into the Container:**
   ```bash
   docker run --gpus all --network=host --rm \
     -v ./config.ini:/app/config.ini \
     yolov8_detection:latest
   ```
   - **Benefit:** Allows you to modify configurations without rebuilding the Docker image.

3. **Persist Detection Outputs and Logs:**
   To ensure that detection results and logs are stored outside the container for later review:
   ```bash
   docker run --gpus all --network=host --rm \
     -v ./config.ini:/app/config.ini \
     -v ./yolo_detections:/app/yolo_detections \
     -v ./logs:/app/logs \
     yolov8_detection:latest
   ```
   - **Explanation:**
     - **Detections Volume (`/app/yolo_detections`):** Stores images with detected objects.
     - **Logs Volume (`/app/logs`):** Contains log files detailing detection events and system messages.

### 🗃️ **Data Persistence and Volume Management**

To maintain data across container restarts or to access outputs directly on your host system, utilize Docker volumes effectively.

1. **Create Host Directories:**
   ```bash
   mkdir -p ./yolo_detections
   mkdir -p ./logs
   ```
   
2. **Run the Container with Volume Mounts:**
   ```bash
   docker run --gpus all --network=host --rm \
     -v ./config.ini:/app/config.ini \
     -v ./yolo_detections:/app/yolo_detections \
     -v ./logs:/app/logs \
     yolov8_detection:latest
   ```
   - **Benefits:**
     - **Persistent Storage:** Data remains intact even if the container is removed.
     - **Easy Access:** Access detection images and logs directly from your host machine.

### 🐞 **Troubleshooting Tips**

- **No GPU Detected:**
  - Ensure that NVIDIA drivers and the NVIDIA Container Toolkit are correctly installed.
  - Verify Docker has access to the GPU by running:
    ```bash
    docker run --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
    ```
    - **Expected Output:** Displays your GPU details.

- **RTMP Stream Issues:**
  - Confirm that your RTMP server is running and accessible.
  - Ensure the stream URL in `config.ini` matches your RTMP server settings.

- **Webcam Access Problems:**
  - Verify that the container has access to the webcam device.
  - You might need to specify device permissions or use privileged mode (use cautiously):
    ```bash
    docker run --gpus all --network=host --rm \
      --device=/dev/video0:/dev/video0 \
      yolov8_detection:latest
    ```

- **Permission Denied Errors:**
  - Ensure your user is part of the `docker` group.
  - Re-run the `run_dockerized.sh` script or use `sudo` if necessary.

- **Missing `requirements.txt`:**
  - Ensure that `requirements.txt` is present in the build context when building the Docker image.
  - Verify the `Dockerfile` has the correct `COPY` path for `requirements.txt`.

### 📝 **Additional Notes**

- **Host Networking Limitations:**
  - The `--network=host` flag is **only supported on Linux**. If you're using Docker Desktop on Windows or macOS, consider alternative networking configurations, such as using `host.docker.internal` or Docker Compose with a shared network.

- **Configuration Flexibility:**
  - By mounting the `config.ini` file, you can easily switch between different configurations without modifying the Docker image.

- **Automated Scripts:**
  - Utilize the provided `run_dockerized.sh` script for streamlined execution and to handle common setup checks automatically.

By following this Dockerized setup, you can efficiently deploy and manage the `dvr-yolov8-detection` application, leveraging the power of containerization and GPU acceleration for real-time detection tasks.

---
