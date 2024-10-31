#! /bin/bash

mkdir output
docker build -t wobot .
docker run -v $(pwd)/output/:/code/output/ wobot
