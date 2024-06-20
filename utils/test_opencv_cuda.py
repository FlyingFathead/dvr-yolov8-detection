# python test_opencv_cuda.py 
# a quick util to check that OpenCV & torch are installed with CUDA and is running OK

import torch
import cv2
import torchvision
import torchaudio
import logging

def check_opencv_cuda():
    logging.info("Checking OpenCV build information for CUDA support...")
    info = cv2.getBuildInformation()
    logging.info(info)
    if 'CUDA' in info:
        logging.info("OpenCV is built with CUDA support.")
    else:
        logging.error("OpenCV is not built with CUDA support.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("OpenCV version:", cv2.__version__)
    print("CUDA support:", cv2.cuda.getCudaEnabledDeviceCount() > 0)
    print("Torch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("Torchaudio version:", torchaudio.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device:", torch.cuda.get_device_name(0))

    check_opencv_cuda()
