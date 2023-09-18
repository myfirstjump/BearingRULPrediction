FROM python:3.9

# 複製requirements
COPY requirements.txt /app/requirements.txt

# 设置工作目录
WORKDIR /app

# 安装依赖项
RUN apt-get update && apt-get install -y \
    build-essential \
    libsdl1.2-dev \
    libgl1-mesa-dev \
    libsm6 \
    libxrender1 \
    libpq-dev \
    libjpeg-dev \
    libboost-all-dev \
    cmake \
	pip \
    swig \
    wget \
    curl \
    x11-apps

# 更新pip並清除緩存
RUN pip install --upgrade pip && pip cache purge

# 安装 OpenAI Gym
RUN pip install -r requirements.txt


