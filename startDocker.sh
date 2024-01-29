#!/bin/bash
jupyter=$1

if [ "$jupyter" = "jupyter" ]; then
    echo "Starting Jupyter Notebook"
    xhost +local:docker
    docker run -it --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/workspace/cvspace/ \
        -w /workspace/cvspace/notebook/ -p 8888:8888 --name cv-dev codesteller/pytorch2-cv:23.12-py3 \
        jupyter-lab --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.token=''
else
    # Start docker with docker run command and pwd as volume to mount with display
    xhost +local:docker
    docker run -it --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/workspace/cvspace/ \
        -w /workspace/cvspace --name cvspace codesteller/pytorch2-cv:23.12-py3
fi

