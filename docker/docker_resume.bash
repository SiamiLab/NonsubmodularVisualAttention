#!/bin/bash

xhost +local:root

docker start nonsubmodular_visual_attention_container

docker exec -it \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
  nonsubmodular_visual_attention_container zsh
xhost -local:root

