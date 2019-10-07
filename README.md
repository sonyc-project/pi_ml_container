# SONYC Raspberry Pi based machine listening Docker container
This hosues all things related to the RPi ML container for SONYC. The Dockfiles allow for the building of a Docker container with the required packages installed to run ML inference on an `arm7l` based sensor node.

## Running image build
It is recommended to use nohup to build the images on your test Raspberry Pi so you can leave the build process detached either when running locally or over SSH. The following can be run when within the directory of the Dockerfile:

`nohup docker build --no-cache=true -t <IMAGE_NAME>:<IMAGE_VERSION> . &`

There are currently two versions:
* `sonyc_ml/` - only includes tflite runtime components (fully untested) - 985MB
* `sonyc_ml_full_tf/` - includes tflite runtime and full tensorflow - 1.3GB

## Docker Hub
To pull the docker image from Docker Hub to your test RPi use the following command:
`docker pull cmydlarz/sonyc_ml:<IMAGE_VERSION>`

To setup your test RPi to push updated images you need to add your credentials locally and login:
`docker login --username=yourhubusername --email=youremail@company.com`

Make sure you have been added as a collaborator to the `cmydlarz/sonyc_ml` repository: https://cloud.docker.com/repository/docker/cmydlarz/sonyc_ml

For docker push commands to complete from RPi, you need to set `max-concurrent-uploads` to `1` when pushing large images up:

`nano /etc/docker/daemon.json`
```
{
  "max-concurrent-uploads": 1
}
```
`systemctl restart docker`

Then to push an updated image:
`docker push cmydlarz/sonyc_ml:<IMAGE_VERSION>`
