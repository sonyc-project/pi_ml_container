# nohup docker build --no-cache=true -t sonyc_ml:0.1 . &

FROM python:3.7-slim

LABEL MAINTAINER="Charlie Mydlarz <cmydlarz@nyu.edu>"

RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN ln -sf pip3 /usr/bin/pip

RUN apt-get -y update \
    && apt-get install --no-install-recommends -y \
    gcc \
    g++ \
    make \
    python3-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    # python3-h5py \
    libhdf5-serial-dev \
    libopenblas-dev \
    libatlas-base-dev \
    libffi-dev \
    llvm-dev \
    libsndfile1 \
    && apt-get autoremove \
    && apt-get clean \
    && rm -rf /root/.cache \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --compile numpy \
    && LLVM_CONFIG=/usr/bin/llvm-config pip install --no-cache-dir --compile llvmlite \
    && pip install --no-cache-dir --compile grpcio \
    && pip install --no-cache-dir --compile https://www.piwheels.org/simple/scipy/scipy-1.3.1-cp37-cp37m-linux_armv7l.whl \
    && pip install --no-cache-dir --compile Cython \
    && HDF5_DIR=/usr/lib/arm-linux-gnueabihf/hdf5/serial pip install --no-cache-dir --compile h5py \
    && pip install --no-cache-dir --compile kapre \
    && pip install --no-cache-dir --compile keras \
    # && pip install --no-cache-dir --compile https://github.com/lhelontra/tensorflow-on-arm/releases/download/v1.14.0-buster/tensorflow-1.14.0-cp37-none-linux_armv7l.whl \
    && pip install --no-cache-dir --compile https://dl.google.com/coral/python/tflite_runtime-1.14.0-cp37-cp37m-linux_armv7l.whl \
    && rm -rf /root/.cache \
    && rm -rf /tmp/pip_build_root/

RUN apt-get remove -y --auto-remove \
    make \
    # llvm-dev \
    && apt-get remove -y --auto-remove python3-dev \
    && apt-get autoremove && apt-get clean

CMD ["/bin/bash"]
