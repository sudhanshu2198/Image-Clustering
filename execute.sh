mkdir output
docker build -t wobot .
docker run -v output/:/code/output/ wobot
