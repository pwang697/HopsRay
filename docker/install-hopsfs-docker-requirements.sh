#!/bin/bash

set -e

HADOOP_VERSION_EE=3.2.0.13-EE-SNAPSHOT
if test -f "/root/.wgetrc"; then
    wget https://nexus.hops.works/repository/hopshadoop/hops-$HADOOP_VERSION_EE.tgz
    mkdir -p /srv/hops
    tar -C /srv/hops/ -zxf hops-$HADOOP_VERSION_EE.tgz
    ln -s /srv/hops/hadoop-$HADOOP_VERSION_EE /srv/hops/hadoop
    rm hops-$HADOOP_VERSION_EE.tgz
    rm -rf /srv/hops/hadoop/etc
    rm -rf /srv/hops/hadoop/sbin
    chown ray:users -R /srv/hops
else
    exit 1
fi
