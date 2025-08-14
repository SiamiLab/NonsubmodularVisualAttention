#!/bin/bash

xhost +local:root
docker run -it \
    --name nonsubmodular_visual_attention_container \
    --gpus all \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --env="NO_AT_BRIDGE=1" \
    --env="LIBGL_ALWAYS_INDIRECT=0" \
    --env="__GLX_VENDOR_LIBRARY_NAME=nvidia" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="/etc/machine-id:/etc/machine-id:ro" \
    --volume="/home/siamilab/Euroc:/root/Euroc:rw" \
    --device=/dev/dri \
    -p 3333:22 \
    nonsubmodular_visual_attention zsh
xhost -local:root