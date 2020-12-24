FROM ubuntu:20.04
LABEL maintainer="Mo1cibo"

WORKDIR /root/moblie_is/
#VOLUME ./pytorch/ /root/moblie_is/

# Initial update
RUN apt update && apt install -y --no-install-recommends build-essential

# Other packages
RUN apt install -y --no-install-recommends \
    vim \
    python3 \
    python3-pip

# Packages for pyenv
RUN DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
    libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
RUN curl https://pyenv.run | bash

# Install pipenv
RUN pip3 install pipenv

# Download pytorch model
COPY . /root/moblie_is
WORKDIR /root/moblie_is/weights
# RUN echo $PWD && ls -a
# RUN cd $PWD && ls -a ..
# RUN ls
RUN bash gdown_model.sh

# Install python packages by pipenv
RUN pipenv install

#WORKDIR /root/moblie_is/weights