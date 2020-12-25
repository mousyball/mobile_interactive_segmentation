FROM ubuntu:20.04
LABEL maintainer="Mo1cibo"

# Setup working directory
COPY . /root/moblie_is/
WORKDIR /root/moblie_is/

# Initial update
RUN apt update && apt install -y --no-install-recommends build-essential

# Essential packages for Python3
RUN apt install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev
# [Optional] Install tkinter if visualization is needed.
# RUN apt install -y --no-install-recommends \
#     python3-tk

# Other packages
RUN apt install -y --no-install-recommends \
    vim

# Packages for pyenv
RUN DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
    libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
RUN curl https://pyenv.run | bash

# Install pipenv
RUN pip3 install pipenv

# Download pytorch model
WORKDIR /root/moblie_is/weights/
RUN bash gdown_model.sh

# Install python packages by pipenv
WORKDIR /root/moblie_is/
RUN pipenv install
