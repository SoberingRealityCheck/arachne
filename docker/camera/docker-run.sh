docker run -it --rm \
  --privileged \
  --net=host \
  -v /dev:/dev/ \
  -v /run/udev/:/run/udev/ \
  --group-add video \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  camera_ros \
  ros2 run camera_ros camera_node
