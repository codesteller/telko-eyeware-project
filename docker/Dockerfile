# /**
#  * @ Author: Pallab Maji
#  * @ Create Time: 2024-01-22 09:46:13
#  * @ Modified by: Pallab
#  * @ Modified time: 2024-01-22 09:49:10
#  * @ Description:
#  */

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y git neovim tmux 
RUN apt-get install -y software-properties-common build-essential
RUN apt-get install -y git wget curl 
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY telko-eyeware-project/docker/requirements.txt .
RUN pip install -r requirements.txt

# Open port 8888 for jupyter notebook
EXPOSE 8888

RUN git clone https://huggingface.co/briaai/RMBG-1.4 /workspace/&& cd RMBG-1.4 && pip install -r requirements.txt

