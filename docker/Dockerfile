# This is an auto generated Dockerfile for ros:ros-core
# generated from docker_images_ros2/create_ros_core_image.Dockerfile.em
FROM ubuntu:noble

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    ca-certificates \
    curl \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*


# Setup ROS Apt sources
RUN curl -L -s -o /tmp/ros2-apt-source.deb https://github.com/ros-infrastructure/ros-apt-source/releases/download/1.1.0/ros2-apt-source_1.1.0.noble_all.deb \
    && echo "35441f3092fd05773a3c397fab38661bec466584c7a1f1c05366579997cb5fe7 /tmp/ros2-apt-source.deb" | sha256sum --strict --check \
    && apt-get update \
    && apt-get install /tmp/ros2-apt-source.deb \
    && rm -f /tmp/ros2-apt-source.deb \
    && rm -rf /var/lib/apt/lists/*

# setup environment
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

ENV ROS_DISTRO=jazzy

# Set up headless camera operation
ENV DISPLAY=:99
ENV LIBCAMERA_LOG_LEVELS=ERROR
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/local/lib/aarch64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH

# install ros2 packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-ros-core=0.11.0-1* \
    && rm -rf /var/lib/apt/lists/*

# setup entrypoint
COPY ./ros_entrypoint.sh /

RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


# install packages required for camera feed
RUN apt update \
    && apt install -y --no-install-recommends \
    python3-pip git python3-jinja2

# more dependencies teehee
RUN apt install -y --no-install-recommends \
    libepoxy-dev \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    libopencv-dev \
    python3-opencv \
    kmod \
    v4l-utils \
    ffmpeg

RUN apt install -y --no-install-recommends \
    qtbase5-dev \
    libqt5core5a \
    libqt5gui5 \
    libqt5widgets5 \
    libavcodec-dev \
    libavdevice-dev \
    libavformat-dev \
    libswresample-dev


# install one bajillion libcamera dependencies
RUN apt install -y --no-install-recommends \
    libboost-dev \
    libgnutls28-dev openssl libtiff5-dev pybind11-dev \
    qtbase5-dev libqt5core5a libqt5gui5 libqt5widgets5 \
    meson cmake \
    python3-yaml python3-ply \
    libglib2.0-dev libgstreamer-plugins-base1.0-dev

RUN apt install -y --no-install-recommends cmake libboost-program-options-dev \
    libdrm-dev \
    libexif-dev \
    meson \
    ninja-build \
    pkg-config 

# clone and build libcamera
RUN git clone https://github.com/raspberrypi/libcamera.git \
    && cd libcamera \
    && meson setup build --buildtype=release -Dpipelines=rpi/vc4,rpi/pisp -Dipas=rpi/vc4,rpi/pisp -Dv4l2=true -Dgstreamer=enabled -Dtest=false -Dlc-compliance=disabled -Dcam=disabled -Dqcam=disabled -Ddocumentation=disabled -Dpycamera=enabled \
    && ninja -C build \
    && ninja -C build install > libcam_ninja.log


# build rpicam-apps 
RUN cd ../ && \
    git clone https://github.com/raspberrypi/rpicam-apps.git \
    && cd rpicam-apps \
    && meson setup build \
    -Denable_libav=enabled -Denable_drm=enabled \
    -Denable_egl=enabled -Denable_qt=enabled \
    -Denable_opencv=disabled -Denable_tflite=disabled \
    -Denable_hailo=disabled \
    && meson compile -C build \
    && meson install -C build > rpicam-meson.log

# Configure dynamic linker run-time bindings
RUN ldconfig > ldconfig.log

# Setup raspberry pi repositories with apt - COMMENTED OUT DUE TO DEPENDENCY CONFLICTS
# RUN echo deb http://archive.raspberrypi.org/debian/ bullseye main \ 
#     | tee /etc/apt/sources.list.d/raspberrypi.list

# RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 7FA3303E
RUN apt-get update

# ONE LAST DEPENDENCY GODDAMMIT
RUN apt install -y --no-install-recommends \
    libcap-dev \
    libfmt-dev \
    v4l-utils

# ok now our python packages
COPY requirements.txt .
RUN pip3 install -r requirements.txt --break-system-packages > pip_install.log

# Try to install pykms for picamera2 compatibility
RUN pip3 install pykms --break-system-packages || echo "pykms not available via pip"

# Create a symbolic link for kms++ bindings if available
RUN python3 -c "\
try:\
    import kms;\
    import sys;\
    import os;\
    pykms_path = os.path.join(os.path.dirname(kms.__file__), '..', 'pykms.py');\
    with open(pykms_path, 'w') as f:\
        f.write('from kms import *\\n');\
    print('Created pykms alias for kms');\
except ImportError:\
    print('kms module not available, relying on mock');\
" || echo "Failed to create kms alias"

# Just to make sure everything is working - commented out since we removed Pi repo
# RUN rpicam-still --version > test.txt

# Link our libcam install to the system python library
RUN ln -s \
    /usr/local/lib/aarch64-linux-gnu/python3.12/site-packages/libcamera \
    /usr/local/lib/python3.12/dist-packages/libcamera

# Create a script to load camera kernel modules
RUN echo '#!/bin/bash\n\
# Load camera kernel modules for Raspberry Pi Camera Module 3\n\
modprobe imx708 2>/dev/null || echo "imx708 module already loaded or not available"\n\
modprobe dw9807_vcm 2>/dev/null || echo "dw9807_vcm module already loaded or not available"\n\
modprobe rp1_cfe 2>/dev/null || echo "rp1_cfe module already loaded or not available"\n\
modprobe pisp_be 2>/dev/null || echo "pisp_be module already loaded or not available"\n\
echo "Camera modules loaded"\n' > /load_camera_modules.sh \
    && chmod +x /load_camera_modules.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
