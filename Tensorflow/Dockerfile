FROM tensorflow/tensorflow:2.3.2-gpu
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
WORKDIR /tf
# install unix libraries
RUN apt-get update -y --fix-missing
RUN apt-get install -y ffmpeg
RUN ["apt-get", "install", "-y", "libsm6", "libxext6", "libxrender-dev"]
COPY requirements.txt requirements.txt
# install pip libraries
RUN pip install --upgrade cython
RUN pip install -r requirements.txt
# run jupyter
RUN pip install jupyter
