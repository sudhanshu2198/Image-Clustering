import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.decomposition import PCA
import shutil
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

# function for separting and organzing images according to person 
def clustering():
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    data_dir="data"

    #resnet model for feature extraction from the image
    weights = ResNet50_Weights.IMAGENET1K_V2
    model=resnet50(weights=weights)
    features=nn.Sequential(*list(model.children())[:-1])
    
    names=[]
    hold=[]
    img_list=sorted(list(os.listdir(data_dir)))
    
    print("Extracting features from the input image")
    for i in tqdm(range(len(os.listdir(data_dir)))):
        img_name=img_list[i]
        names.append(img_name)
        img_pth=os.path.join(data_dir,img_name)
        img=cv2.imread(img_pth)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=cv2.resize(img,(256,256))
        img=(img/255.0-mean)/std
        img_tensor=torch.from_numpy(img).to(dtype=torch.float32)
        img_tensor=img_tensor.permute(2,0,1).unsqueeze(0)
        
        features.eval()
        with torch.no_grad():
            output=features(img_tensor)
            hold.append(output.view(-1).tolist())

    
    #dimensionality reduction to prevent cod
    df=pd.DataFrame(data=np.array(hold))
    pca = PCA(n_components=13,random_state=0)
    inter_df=pca.fit_transform(df)
        
    # kmeans clustering the image features
    cs = []
    silhouette_scores = []
    
    print("Determining the number of cluster using KMeans")
    for i in tqdm(range(2, 25)):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 1000, 
                        n_init = 10, random_state = 0)
        kmeans.fit(inter_df)
        cs.append(kmeans.inertia_)
        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(inter_df, labels))
         
    #plot for determining the number of clusters
    plt.plot(np.arange(2,25) , cs , 'o')
    plt.plot(np.arange(2 ,25) , cs , '-' , alpha = 0.5)
    plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
    plt.savefig("Inertia.jpg")
            
    plt.plot(range(2, 25), silhouette_scores, marker='o')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.savefig("Silhouette_Score.jpg")
        
    print("Based on Silhoute Score graph the number of cluster is set as 16")
    kmeans = KMeans(n_clusters = 16, init = 'k-means++', max_iter = 1000, 
                    n_init = 10, random_state = 0)
    kmeans.fit(inter_df)
    labels = kmeans.labels_.tolist()
    sub=pd.DataFrame({"name":names,
                      "label":labels})
    sub.head()
        
    cwd=os.getcwd()
    output_dir=os.path.join(cwd,"output")
        
    #organzing images based on their label in output folder
    for i in range(1,sub["label"].nunique()+1):
        subfolder=os.path.join(output_dir,f"person_{i}")
        os.mkdir(subfolder)
            
    for i in range(len(sub)):
        name=sub.loc[i,"name"]
        label=sub.loc[i,"label"]
        current_path=os.path.join(data_dir,name)
        output_path=os.path.join(output_dir,f"person_{label+1}",name)
            
        img=cv2.imread(current_path)
        cv2.putText(img,f'{label}', (5,12),cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,(255,0,0),1,1)
        cv2.imwrite(output_path, img)
            
if __name__ == "__main__":
    clustering()
    print("Image Clustering Completed")
    
