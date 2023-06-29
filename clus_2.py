
from sklearn.metrics import silhouette_score
import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
# This file has IDs and CPV
destination_file = r"E:\2ndYear\Slivia's project\Silvia Dataset\destination.csv"


df = pd.read_csv(destination_file)
# Count the number of rows with NaN values
num_rows_with_nan = df.isna().any(axis=1).sum()
print("Number of rows with NaN values:", num_rows_with_nan)

# Find the rows that have NaN values
rows_with_nan = df[df.isna().any(axis=1)]
# Print the rows with NaN values
print(rows_with_nan)
# Drop rows with nan values 
df = df.dropna()
df = df.set_index('Sample.ID')
df = df.drop('Sentrix.ID', axis=1)
# Check for null values
df.isnull().sum()

from sklearn.decomposition import PCA
preprocessor = Pipeline(
    [
        ("pca", PCA(n_components=2, random_state=42)),
    ]
)

clusterer = Pipeline(
   [
       (
           "kmeans",
           KMeans(
               n_clusters=10,
               init="k-means++",
               n_init=50,
               max_iter=500,
               random_state=42,
           ),
       ),
   ]
)

pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("clusterer", clusterer)
    ]
)
pipe.fit(df)

preprocessed_data = pipe["preprocessor"].transform(df)

predicted_labels = pipe["clusterer"]["kmeans"].labels_

silhouette_score(preprocessed_data, predicted_labels)

####################### Plot


pcadf = pd.DataFrame(
    pipe["preprocessor"].transform(df),
    columns=["component_1", "component_2"],
)


pcadf["predicted_cluster"] = pipe["clusterer"]["kmeans"].labels_
#pcadf["true_label"] = label_encoder.inverse_transform(true_labels)

plt.style.use("fivethirtyeight")
plt.figure(figsize=(8, 8))

scat = sns.scatterplot(
    "component_1",
    "component_2",
    s=50,
    data=pcadf,
    hue="predicted_cluster",
    #style="true_label",
    palette="Set2",
)

scat.set_title(
    "Clustering results from TCGA Pan-Cancer\nGene Expression Data"
)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

plt.show()


#######################  Clustring ###########################

# Import KMeans
from sklearn.cluster import KMeans
def find_best_clusters(df, maximum_K):
    
    clusters_centers = []
    k_values = []
    
    for k in range(1, maximum_K):
        
        kmeans_model = KMeans(n_clusters = k)
        kmeans_model.fit(df)
        
        clusters_centers.append(kmeans_model.inertia_)
        k_values.append(k)
        
    
    return clusters_centers, k_values

def generate_elbow_plot(clusters_centers, k_values):
    
    figure = plt.subplots(figsize = (12, 6))
    plt.plot(k_values, clusters_centers, 'o-', color = 'orange')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Cluster Inertia")
    plt.title("Elbow Plot of KMeans")
    plt.show()
    
clusters_centers, k_values = find_best_clusters(df, 20)

generate_elbow_plot(clusters_centers, k_values)


kmeans_model = KMeans(n_clusters = 3)

kmeans_model.fit(df)


df["clusters"] = kmeans_model.labels_

df["clusters"].value_counts()
df.describe()

plt.scatter(df["MDM4"],df["PPM1D"] ,c = df["clusters"])
print(df['clusters'].head(20),df['mgmt.Status'].head(20))
####################### PCA ########################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from numpy.random import randn, uniform
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler

def Best_Clustering(data = df, max_clusters = 20, scaling = True, visualization = True):
    
    n_clusters_list=[]
    silhouette_list=[]
    
    if scaling:
    #Data Scaling
        scaler = MinMaxScaler()
        data_std = scaler.fit_transform(data)
    else:    
        data_std = data
        
    for n_c in range(2,max_clusters+1): 
        kmeans_model = KMeans(n_clusters=n_c, random_state=42).fit(data_std) 
        labels = kmeans_model.labels_
        n_clusters_list.append(n_c)
        silhouette_list.append(silhouette_score(data_std, labels, metric='euclidean'))
    
    # Best Parameters
    param1 = n_clusters_list[np.argmax(silhouette_list)]
    param2 = max(silhouette_list)
    best_params = param1,param2
    
    # Data labeling with the best model
    kmeans_best = KMeans(n_clusters= param1 , random_state=42).fit(data_std) 
    labels_best = kmeans_best.labels_
    labeled_data = np.concatenate((data,labels_best.reshape(-1,1)),axis=1)
        
    if visualization:
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(111)
        ax.plot(n_clusters_list,silhouette_list, linewidth=3,
                label = "Silhouette Score Against # of Clusters")
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("Silhouette score")
        ax.set_title('Silhouette score according to number of clusters')
        ax.grid(True)
        plt.plot(param1,param2, "tomato", marker="*",
             markersize=20, label = 'Best Silhouette Score')
    
        plt.legend(loc="best",fontsize = 'large')
        plt.show();
        print( " Best Clustering corresponds to the following point : \
        Number of clusters = %i & Silhouette_score = %.2f."%best_params) 
    else:
        return best_params, labeled_data
    
# We don't need to scale the data since we only have one variable
Best_Clustering(data = df, scaling = False)
best_params , my_labeled_data = Best_Clustering(data = df, scaling = False, visualization = False)
best_params
