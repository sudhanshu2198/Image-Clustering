#! /bin/bash

mkdir output
docker build -t wobot .
docker run -v /workspaces/Image-Clustering/output/:/code/output/ wobot
