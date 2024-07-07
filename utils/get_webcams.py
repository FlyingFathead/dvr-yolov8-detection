#!/usr/bin/env python3

import cv2
import platform
import os
import subprocess

def check_v4l2_installed():
    try:
        subprocess.check_output("which v4l2-ctl", shell=True)
        return True
    except subprocess.CalledProcessError:
        return False

def list_video_devices_v4l2():
    try:
        output = subprocess.check_output("v4l2-ctl --list-devices", shell=True, stderr=subprocess.STDOUT)
        devices_output = output.decode().split("\n")
        devices = {}
        current_device = None
        for line in devices_output:
            if not line.startswith('\t') and line.strip():
                current_device = line.strip()
                devices[current_device] = []
            elif line.startswith('\t'):
                devices[current_device].append(line.strip())
        return devices
    except subprocess.CalledProcessError as e:
        print(f"Error running v4l2-ctl: {e.output.decode()}")
        return {}

def list_webcams_opencv(max_tested=10):
    available_webcams = []
    for index in range(max_tested):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_webcams.append(index)
            cap.release()
    return available_webcams

def main():
    system_platform = platform.system()
    print(f"Running on: {system_platform}")

    if system_platform == "Linux":
        if check_v4l2_installed():
            print("v4l2-ctl is installed. Using it to list webcams.")
            devices = list_video_devices_v4l2()
            if devices:
                print("Available webcams:")
                for device, paths in devices.items():
                    print(f"{device}:")
                    for path in paths:
                        if "video" in path:
                            index = int(path.split('/dev/video')[-1])
                            print(f"  Index {index}: {path}")
            else:
                print("No webcams detected with v4l2-ctl.")
        else:
            print("v4l2-ctl is not installed. Falling back to OpenCV.")
            webcams = list_webcams_opencv()
            if webcams:
                print("Available webcams (using OpenCV):")
                for index in webcams:
                    print(f"Webcam index: {index}")
            else:
                print("No webcams detected with OpenCV.")
    else:
        print("Using OpenCV to list webcams.")
        webcams = list_webcams_opencv()
        if webcams:
            print("Available webcams:")
            for index in webcams:
                print(f"Webcam index: {index}")
        else:
            print("No webcams detected.")

if __name__ == "__main__":
    main()
