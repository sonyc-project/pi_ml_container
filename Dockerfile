FROM python:3.6-slim
MAINTAINER Charlie Mydlarz (cmydlarz@nyu.edu)

# COPY ./Requirements.txt /pyml/Requirements.txt

RUN echo "**** installing python packages ****" && \
    pip3 install --no-cache numpy==1.15.0

RUN echo "**** installing python packages ****" && \
    pip3 install --no-cache scipy==0.19.1

RUN echo "**** installing python packages ****" && \
    pip3 install --no-cache kapre==0.1.4

RUN echo "**** installing python packages ****" && \
    pip3 install --no-cache PySoundFile==0.9.0.post1

RUN echo "**** installing python packages ****" && \
    pip3 install --no-cache resampy==0.2.1

RUN echo "**** installing python packages ****" && \
    pip3 install --no-cache h5py==2.8.0

RUN echo "**** installing python packages ****" && \
    pip3 install --no-cache tensorflow==1.12.0

RUN echo "**** installing python packages ****" && \
    pip3 install --no-cache keras==2.0.9
