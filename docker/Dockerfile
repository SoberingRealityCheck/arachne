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
    libopencv-dev

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

# clone and build libcamera
RUN git clone https://github.com/raspberrypi/libcamera.git \
    && cd libcamera \
    && meson setup build --buildtype=release -Dpipelines=rpi/vc4,rpi/pisp -Dipas=rpi/vc4,rpi/pisp -Dv4l2=true -Dgstreamer=enabled -Dtest=false -Dlc-compliance=disabled -Dcam=disabled -Dqcam=disabled -Ddocumentation=disabled -Dpycamera=enabled \
    && ninja -C build \
    && ninja -C build install


# build rpicam-apps 
RUN apt install -y --no-install-recommends cmake libboost-program-options-dev \
    libdrm-dev \
    libexif-dev \
    meson \
    ninja-build

RUN cd ../ && \
    git clone https://github.com/raspberrypi/rpicam-apps.git \
    && cd rpicam-apps \
    && meson setup build -Denable_libav=enabled -Denable_drm=enabled -Denable_egl=enabled -Denable_qt=enabled -Denable_opencv=disabled -Denable_tflite=disabled -Denable_hailo=disabled \
    && meson compile -C build \
    && meson install -C build

# Configure dynamic linker run-time bindings
RUN ldconfig

# ONE LAST DEPENDENCY GODDAMMIT
RUN apt install -y --no-install-recommends \
    libcap-dev

# ok now our python packages
COPY requirements.txt .
RUN pip3 install -r requirements.txt --break-system-packages

# Just to make sure everything is working
RUN rpicam-still --version > test.txt

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
