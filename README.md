# Image Clustering

- The problem falls under unsupervised learning as the target labels are not given.
- The solution involves utilizing pretrained model for features extraction and then
clustering images based on the kmeans clustering .
- The Resnet50 pertained model without classifier is used to extract features vectors of
dimension 1268.
- Principal component analysis is performed to reduce the dimensionality of the feature
vector to avoid the curse of dimensionality.
- In order to determine the number of clusters , kmeans clustering algorithm is utilized
by varying the number of clusters between 2 and 25.
- Inertia vs no_of_cluster and silhoutte_score vs no_of_cluster plots are used to
determine the number of clusters.
- The optimal number of clusters is determined at 16.

## Alternative Approach
- The problem with approach 1 is that many images containing different people but with
the same image background are clustered together.
- In order to remove the image background for contributing to image clustering, the
background should be masked and then followed by approach 1.
- Image threshold followed by erosion and dilation is performed for segmenting images,
but it performed poorly in segmenting individuals from background.
- With classical thresholding not working, deeplab degmentaion model is used for
segmenting individuals from background.
- The Deeplab model succeeded in segmenting images, but due to similar clothing, pose
and obstructed images it also yielded poor results in image clustering.

  ## Directory Tree
```bash
├── data/
├── Dockerfile
├── README.md
├── execute.sh
├── main.py
└── requirements.txt
```

## Run Locally

Clone the project

```bash
git clone https://github.com/sudhanshu2198/Image-Clustering
```

Change to project directory

```bash
cd Image-Clustering
```

If Docker is installed
```bash
./execute.sh
```


Else Create Virtaul Environment and install dependencies

```bash
  python -m venv venv
  venv/Scripts/activate
  pip install -r requirements.txt
```

Run Locally
```bash
  python main.py
```
