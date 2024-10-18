# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 22:11:26 2024

@author: Omnia
"""
import cv2
import numpy as np
image_path = r"C:\Users\Omnia\Desktop\Phd\DNA_methy\plot_att_dual_late\NH22-3295   11A_H&E\NH22-3295   11A_H&E d-4.3385_x-4434_y-4434_w-2221_h-2221.tif"


#%%
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
patch_folder = r"C:\Users\Omnia\Desktop\Phd\DNA_methy\plot_att_dual_late\NH22-3295   11A_H&E"
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
