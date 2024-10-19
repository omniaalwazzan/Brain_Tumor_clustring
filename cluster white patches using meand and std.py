# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 01:13:57 2024

@author: Omnia
"""

import shutil
import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt

# extract color features from a patch
def extract_color_features(image):
    if len(image.shape) == 3 and image.shape[-1] == 3:
        image_rgb = image
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # mean and standard deviation for each channel
    mean_r = np.mean(image_rgb[:, :, 0])
    mean_g = np.mean(image_rgb[:, :, 1])
    mean_b = np.mean(image_rgb[:, :, 2])
    
    std_r = np.std(image_rgb[:, :, 0])
    std_g = np.std(image_rgb[:, :, 1])
    std_b = np.std(image_rgb[:, :, 2])
    
    # Calculate ratios between channels
    ratio_r_b = mean_r / (mean_b + 1e-6)
    ratio_g_b = mean_g / (mean_b + 1e-6)
    
    # Return feature vector
    return [mean_r, mean_g, mean_b, std_r, std_g, std_b, ratio_r_b, ratio_g_b]

#%%
patch_folder = "/data/DERI-MMH/DNA_meth/HE_patches/NH22-1339 A1_H&E"
patch_paths = [os.path.join(patch_folder, fname) for fname in os.listdir(patch_folder)]

X = []  # Feature list
valid_patch_paths = []  # To keep track of valid paths with proper images

for patch_path in patch_paths:
    patch_image = cv2.imread(patch_path)
    if patch_image is not None:
        features = extract_color_features(patch_image)
        X.append(features)
        valid_patch_paths.append(patch_path)  # Only add valid patches

# Apply K-Means Clustering with 3 clusters (stains?)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
#%%
# cluster assignments
labels = kmeans.labels_

# Organize patches by cluster
cluster_0_patches = [valid_patch_paths[i] for i in range(len(labels)) if labels[i] == 0]
cluster_1_patches = [valid_patch_paths[i] for i in range(len(labels)) if labels[i] == 1]
cluster_2_patches = [valid_patch_paths[i] for i in range(len(labels)) if labels[i] == 2]
#%%

#%%
# visualize some patches
def visualize_patches(patch_paths, cluster_title, num_patches=5):
    plt.figure(figsize=(15, 5))
    for i, patch_path in enumerate(patch_paths[:num_patches]):
        patch_image = cv2.imread(patch_path)
        patch_image_rgb = cv2.cvtColor(patch_image, cv2.COLOR_BGR2RGB)
        
        plt.subplot(1, num_patches, i + 1)
        plt.imshow(patch_image_rgb)
        plt.axis('off')
        plt.title(f"{cluster_title} {i + 1}")

    plt.suptitle(cluster_title)
    plt.show()

# Visualize patches from both clusters
print("Visualizing patches from Cluster 0:")
visualize_patches(cluster_0_patches, "Cluster 0 Patch")

print("Visualizing patches from Cluster 1:")
visualize_patches(cluster_1_patches, "Cluster 1 Patch")

print("Visualizing patches from Cluster 2:")
visualize_patches(cluster_2_patches, "Cluster 2 Patch")
#%%

patch_name = "NH22-1339 A1_H&E"

# Use the variable in the f-string
output_folder_cluster_0 = f"/data/scratch/acw636/BrainTumor_patches/{patch_name}_cluster_0_patches"
output_folder_cluster_1 = f"/data/scratch/acw636/BrainTumor_patches/{patch_name}_cluster_1_patches"
output_folder_cluster_2 = f"/data/scratch/acw636/BrainTumor_patches/{patch_name}_cluster_2_patches"


# Create directories if they do not exist
os.makedirs(output_folder_cluster_0, exist_ok=True)
os.makedirs(output_folder_cluster_1, exist_ok=True)
os.makedirs(output_folder_cluster_2, exist_ok=True)

# Copy patches in each cluster to the corresponding folder
for patch_path in cluster_0_patches:
    file_name = os.path.basename(patch_path)
    new_path = os.path.join(output_folder_cluster_0, file_name)
    shutil.copy(patch_path, new_path)

for patch_path in cluster_1_patches:
    file_name = os.path.basename(patch_path)
    new_path = os.path.join(output_folder_cluster_1, file_name)
    shutil.copy(patch_path, new_path)

for patch_path in cluster_2_patches:
    file_name = os.path.basename(patch_path)
    new_path = os.path.join(output_folder_cluster_2, file_name)
    shutil.copy(patch_path, new_path)

print("All patches have been copied to their respective cluster folders.")

# Plot the distribution of patches in each cluster
clusters = [0, 1, 2]
counts = [len(cluster_0_patches), len(cluster_1_patches), len(cluster_2_patches)]

plt.figure(figsize=(8, 6))
plt.bar(clusters, counts, color=['blue', 'green', 'red'])
plt.xlabel('Cluster')
plt.ylabel('Number of Patches')
plt.title('Distribution of Patches in Each Cluster')
plt.xticks(clusters, ['Cluster 0', 'Cluster 1', 'Cluster 2'])
plt.show()
#%%
# Convert the feature list to a NumPy array for easier processing
features_array = np.array(X)

# Split features into clusters
cluster_0_features = [features_array[i] for i in range(len(labels)) if labels[i] == 0]
cluster_1_features = [features_array[i] for i in range(len(labels)) if labels[i] == 1]
cluster_2_features = [features_array[i] for i in range(len(labels)) if labels[i] == 2]

# Plot histograms for the mean R, G, B values of each cluster
clusters = [cluster_0_features, cluster_1_features, cluster_2_features]
cluster_titles = ["Cluster 0", "Cluster 1", "Cluster 2"]

for i, cluster_features in enumerate(clusters):
    cluster_features = np.array(cluster_features)
    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(cluster_features[:, 0], bins=20, color='red', alpha=0.7)
    plt.title(f"{cluster_titles[i]} - Mean Red")
    
    plt.subplot(1, 3, 2)
    plt.hist(cluster_features[:, 1], bins=20, color='green', alpha=0.7)
    plt.title(f"{cluster_titles[i]} - Mean Green")
    
    plt.subplot(1, 3, 3)
    plt.hist(cluster_features[:, 2], bins=20, color='blue', alpha=0.7)
    plt.title(f"{cluster_titles[i]} - Mean Blue")
    
    plt.suptitle(f"Feature Distribution in {cluster_titles[i]}")
    plt.show()
