
# LIBRARIES
import numpy as np
import pandas as pd
import random
import sys
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# PARAMETERS
IMG_ROOT = str(sys.argv[1])

INPUT_FILEPATH = str(sys.argv[2])
OUTPUT_FILEPATH = str(sys.argv[3])


K = 3 # cluster size for k-means clustering
K_TRIAL_CNT = 10 # number of trials of clustering agorithms to find min
MIN_CLUSTER_SIZE = 15 # minimum size for the least accurate cluster

# print ('Argument List:', str(sys.argv))
# print ('in file:', str(sys.argv[1]))




# ----------------------------------------
# --- STEP 2: Isolate Inaccurate Cluster
# ----------------------------------------
# print('\nSTEP 2: Isolating Innaccurate Cluster ----------------------- ')

# We want to find a cluster with the lowest accuracy
 # - h

    
# Create several different clusters, then check the accuracy of all of them
def basicClustering(x,y_true,y_pred,k,trial_cnt):
    sys.stdout.flush()
    result_accuracy=[1]*k
    result_clusters=1
    
    for t in range(10):
        accuracy=[None]*k
        size=[None]*k

        kmeans = KMeans(k)
        clusters = kmeans.fit(x) 
        
        # calculate the accuracy for each of the clusters
        for i in np.unique(clusters.labels_):
            mask = (clusters.labels_ == i)
            cluster_y_true = y_true[mask]
            cluster_y_pred = y_pred[mask]

            accuracy[i] = accuracy_score(cluster_y_true,cluster_y_pred)
            size[i] = cluster_y_true.shape[0]

        k_min = accuracy.index(min(accuracy))

        # check if the cluster with the smallest accuracy is the min so far
        if min(accuracy)<min(result_accuracy):
            result_accuracy=accuracy
            result_clusters = clusters

        print('trial '+str(t)+': '+str(accuracy[k_min])+' ('+str(size[k_min])+')')
    return(result_accuracy,result_clusters)

# sub cluster

def getLeastAccurateCluster(x,y_true,y_pred,k,n_min):
    accuracy=[None]*k
    size=[None]*k
    clusters = KMeans(k).fit(x)
    for i in np.unique(clusters.labels_):
        mask = (clusters.labels_ == i)
        if sum(mask) >= n_min:
            cluster_x = x[mask,:]
            cluster_y_true = y_true[mask]
            cluster_y_pred = y_pred[mask]

            accuracy[i] = accuracy_score(cluster_y_true,cluster_y_pred)
        else:
            accuracy[i] = 1
        size[i]=sum(mask)
    k_min = accuracy.index(min(accuracy))
    mask = clusters.labels_ == k_min
    center = clusters.cluster_centers_[k_min,:]

    return(mask,min(accuracy),center)

# next two functions are to find the least accurate cluster recursively
def getLeastAccurateCluster_Recursive(x,y_true,y_pred,k,n_min):
    # print('TOP x shape:',x.shape)
    n = x.shape[0]
    indices = list(range(0, n))
    first_mask,first_accuracy,first_center = getLeastAccurateCluster(x,y_true,y_pred,k,n_min)

    return_mask = first_mask
    return_accuracy = first_accuracy
    return_center = first_center
    
    if sum(first_mask) > n_min:
        new_x = x[first_mask,:]
        new_y_true = y_true[first_mask]
        new_y_pred = y_pred[first_mask]
        
        full_i = np.array(list(range(0, n)))
        sub_i = full_i[first_mask]
        
        sub_mask,sub_accuracy,sub_center = getLeastAccurateCluster_Recursive(new_x,new_y_true,new_y_pred,k,n_min)
        
        if sub_accuracy < return_accuracy:
            return_mask = np.full(n,False)
            for i in sub_i[sub_mask]:
                return_mask[i] = True
            return_accuracy = sub_accuracy
            return_center = sub_center

    return(return_mask,return_accuracy,return_center)


def subClustering(x,y_true,y_pred,k,n_min,trial_cnt):
    result_accuracy=1
    result_center=1
    result_mask = []
    
    for t in range(10):
        
        mask,accuracy,center = getLeastAccurateCluster_Recursive(x,y_true,y_pred,k,n_min)

        if accuracy<result_accuracy:
            result_accuracy=accuracy
            result_center = center
            result_mask = mask

        print('trial '+str(t)+': '+str(accuracy)+' ('+str(sum(1*mask))+')')
    return(result_accuracy,result_center,result_mask)


# setting up data

test_df = pd.read_csv(INPUT_FILEPATH)
y_cols = ['y_true','y_pred_score','y_pred']
X_df =  test_df.loc[:, ~test_df.columns.isin(y_cols)]

X = X_df.to_numpy()
X_scaled = StandardScaler().fit_transform(X)

y_true = test_df['y_true']
y_pred = test_df['y_pred']

print(' using basic clustering...')
clust_accuracy, clusters = basicClustering(X_scaled,y_true,y_pred,K,K_TRIAL_CNT)
k_min = clust_accuracy.index(min(clust_accuracy))
print('  lowest accuracy: '+str(min(clust_accuracy)))
print()
print(' using worst cluster search...')
accuracy,center,mask = subClustering(X_scaled,y_true,y_pred,K,MIN_CLUSTER_SIZE,K_TRIAL_CNT)
print('  lowest accuracy: '+str(accuracy))
print()
print('For next part will use cluster with lowest accuracy from worst cluster search:')
print(' accuracy: '+str(accuracy))
print(' size: '+str(sum(mask)))

test_df['clust_mask']=mask
test_df.to_csv(OUTPUT_FILEPATH,index=False)
