
# Image Clustering

**The main goal of project is to organise images based on individual .**
- The problem falls under unsupervised learning as the target labels are not given.
- The solution involves utilizing pretrained model for features extraction and then clustering images based on the kmeans clustering .
- The Resnet50 pertained model without classifier is used to extract features vectors of dimension 1268.
 - Principal component analysis is performed to reduce the dimensionality of the feature vector to avoid the curse of dimensionality.
- In order to determine the number of clusters , kmeans clustering algorithm is utilized by varying the number of clusters between 2 and 25.
- Inertia vs no_of_cluster and silhoutte_score vs no_of_cluster plots are used to determine the number of clusters.
- The optimal number of clusters is determined at 16.


  ## Directory Tree
```bash
├── data/
├── Dockerfile
├── README.md
├── execute.sh
├── main.py
└── requirements.txt
```

## Run the program

```bash
  ./execute.sh
```
