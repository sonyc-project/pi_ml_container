# SONYC Raspberry Pi based machine listening Docker container
This hosues all things related to the RPi ML container for SONYC. The Dockfiles allow for the building of a Docker container with the required packages installed to run ML inference on an `arm7l` based sensor node.

## Running image build
It is recommended to use nohup to build the images on your test Raspberry Pi so you can leave the build process detached either when running locally or over SSH. The following can be run when within the directory of the Dockerfile:

`nohup docker build --no-cache=true -t <IMAGE_NAME>:<IMAGE_VERSION> . &`


## Docker Hub
TBD when latest build is complete
