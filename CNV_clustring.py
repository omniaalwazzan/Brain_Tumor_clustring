import matplotlib.pyplot as plt

import pandas as pd 

# tex_file = pd.read_csv(r"C:\Users\Omnia\Desktop\temp\Metrics.txt",delimiter=("\t"))

# xls = pd.read_excel(r"D:\2ndYear\Slivia's project\Silvia Dataset\Copy_Illumina Array ProjectVersion.xlsx")

# xls.columns

# columns_to_copy = ['Sample.ID', 'Sentrix.ID', 'MDM4', 'MYCN', 'GLI2',
# 'FGFR3_TACC3', 'PDGFRA', 'TERT', 'MYB', 'EGFR', 'CDK6', 'MET',
# 'KIAA1549_BRAF', 'FGFR1_TACC1', 'MYBL1', 'MYC', 'CDKN2A_B', 'PTCH1',
# 'PTEN', 'MGMT', 'CCND1', 'CCND2', 'CDK4', 'MDM2', 'RB1', 'TP53', 'NF1',
# 'PPM1D', 'C19MC', 'SMARCB1', 'NF2']  # Replace with the actual column names you want to copy
# df_copy = xls[columns_to_copy].copy()

# This file has IDs and CPV
destination_file = r"D:\2ndYear\Slivia's project\Silvia Dataset\destination.csv"
#df_copy.to_csv(destination_file, index=False)



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


kmeans_model = KMeans(n_clusters = 16)

kmeans_model.fit(df)


df["clusters"] = kmeans_model.labels_

df.head()


plt.scatter(df["MDM4"],df["PPM1D"] ,c = df["clusters"])
