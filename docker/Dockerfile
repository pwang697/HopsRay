# syntax=docker/dockerfile:experimental
ARG BASE_IMAGE
ARG FULL_BASE_IMAGE=rayproject/ray:"$BASE_IMAGE"
FROM "$FULL_BASE_IMAGE"

# Install HopsFS dependencies
USER root
RUN apt-get update && \
    apt install -y openjdk-8-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64

COPY install-hopsfs-docker-requirements.sh /root/
RUN --mount=type=secret,id=wgetrc,dst=/root/.wgetrc \
    chmod +x /root/install-hopsfs-docker-requirements.sh \
    && /root/install-hopsfs-docker-requirements.sh \
    && rm /root/install-hopsfs-docker-requirements.sh
ENV HADOOP_HOME=/srv/hops/hadoop
ENV PATH=${HADOOP_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${JAVA_HOME}/jre/lib/amd64/server:${HADOOP_HOME}/lib/native:${LD_LIBRARY_PATH}
ENV LD_PRELOAD=${HADOOP_HOME}/lib/native/libhdfs-golang/libhdfs.so
USER ray

# The python/* paths only exist in civ2, so we put them as non-first arguments. Docker
# will ignore non-existent paths if they are non-first arguments.
#
# TODO(can): simplify this once civ1 is completely deprecated.
COPY python/*requirements.txt \
    python/requirements/ml/*requirements.txt  \
    python/requirements/docker/*requirements.txt ./
COPY python/*requirements_compiled.txt ./
COPY install-ml-docker-requirements.sh ./

RUN sudo chmod +x install-ml-docker-requirements.sh \
    && ./install-ml-docker-requirements.sh

# Export installed packages
RUN $HOME/anaconda3/bin/pip freeze > /home/ray/pip-freeze.txt

# Make sure tfp is installed correctly and matches tf version.
RUN python -c "import tensorflow_probability"
