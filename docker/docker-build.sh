#!/bin/bash
set -e
# set ray version here
RAY_VERSION=2.23.0
# get requirements from ray repo
rm -rf python
git clone -b releases/$RAY_VERSION https://github.com/ray-project/ray.git
mkdir -p python/requirements/ml
mkdir -p python/requirements/docker
cp ray/python/*requirements.txt ./python 
cp ray/python/requirements/ml/*requirements.txt ./python/requirements/ml
cp ray/python/requirements/docker/*requirements.txt ./python/requirements/docker
cp ray/python/*requirements_compiled.txt ./python
rm -rf ray
#Build and push docker image
DOCKER_BUILDKIT=1 docker build --secret id=wgetrc,src=wgetrc --build-arg BASE_IMAGE=$RAY_VERSION --progress=plain --tag registry.service.consul:4443/ray-ml:$RAY_VERSION .
docker push registry.service.consul:4443/ray-ml:$RAY_VERSION