
# LIBRARIES
import numpy as np
import pandas as pd
import random
import sys
import torch
import matplotlib.pyplot as plt

from aif360.datasets import GermanDataset # <-- DATASET

from sklearn.metrics import accuracy_score,r2_score, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

# PARAMETERS
IMG_ROOT = 'img/german/'

PERCENT_TEST = 0.33 

BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS = 300

K = 5 # cluster size for k-means clustering
K_TRIAL_CNT = 10 # number of trials of clustering agorithms to find min
MIN_CLUSTER_SIZE = 10 # minimum size for the least accurate cluster

# GET DATA
label_map = {0.0: 'Good Credit', 1.0: 'Bad Credit'}
protected_attribute_maps = [{1.0: 'Male', 0.0: 'Female'}]

data = GermanDataset(
    protected_attribute_names=['sex'],
    privileged_classes=[['male']],
    metadata={'label_map':label_map,'protected_attribute_maps': protected_attribute_maps})
data.labels = data.labels- 1

X = data.features
y = data.labels.ravel()

sc = StandardScaler()
X_scaled = sc.fit_transform(X)

print('Dataset Retrieved')
print(" Shape of X: {}".format(X.shape))
print(" Shape of y: {}".format(y.shape))

# ----------------------------------------
# --- STEP 1: Train Initial Model
# ----------------------------------------

class dataset(Dataset):
    
    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
        self.length = self.x.shape[0]
 
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
    def __len__(self):
        return self.length

class Net(nn.Module):
    def __init__(self,input_shape):
        super(Net,self).__init__()
        self.predict=nn.Linear(input_shape,1)
        
    def forward(self,x):
        x = torch.sigmoid(self.predict(x))
        return x


print('\nSTEP 1: Train Initial Model --------------------------------- ')
print('  splitting data...')
sys.stdout.flush()

n = X.shape[0]
test_n = int(n*PERCENT_TEST)
test_i = random.sample(range(n),test_n)

X_test = np.take(X,test_i,axis=0)
X_train = np.delete(X,test_i,axis=0)

X_scaled_test = np.take(X_scaled,test_i,axis=0)
X_scaled_train = np.delete(X_scaled,test_i,axis=0)

y_test = np.take(y,test_i,axis=0)
y_train = np.delete(y,test_i,axis=0)

print("   Train Size: {}".format(X_train.shape[0]))
print("   Test Size:  {}".format(X_test.shape[0]))


print('  training model...')
sys.stdout.flush()

# SETUP
trainset = dataset(X_scaled_train,y_train)
trainloader = DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=False)

model = Net(input_shape=X_scaled_train.shape[1])
optimizer = torch.optim.SGD(model.parameters(),lr=LEARNING_RATE)
loss_fn = nn.BCELoss()


# TRAINING
epoch = []
losses = []
accur = []

for i in range(EPOCHS):
    for j,(x_this,y_this) in enumerate(trainloader):

        #calculate output
        output = model(x_this)

        #calculate loss
        loss = loss_fn(output,y_this.reshape(-1,1))

        acc = (output.reshape(-1).detach().numpy().round() == y_this.detach().numpy()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if i%50 == 0:
        epoch.append(i)
        losses.append(loss.item())
        accur.append(acc)
        print("epoch {}\tloss : {}\t accuracy : {}".format(i,loss,acc))
        

plt.figure(figsize=(15,5))

plt.subplot(1, 2, 1)
plt.plot(epoch,losses)
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('loss')

plt.subplot(1, 2, 2)
plt.plot(epoch, accur)
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
# plt.show()
plt.savefig(IMG_ROOT+'training_performance.png')

# GET OVERALL TRAIN RESULTS
x_tensor = torch.tensor(X_scaled_train,dtype=torch.float32)
y_pred = model(x_tensor)
y_train_pred = y_pred.detach().numpy()[:,0]

print('TRAIN Results')
accuracy = accuracy_score(y_train,y_train_pred.round())
r2 = r2_score(y_train,y_train_pred.round())
auc = roc_auc_score(y_train,y_train_pred.round())

print('Accuracy: '+str(accuracy))
print('R2:       '+str(r2))
print('AUC:      '+str(auc))

x_tensor = torch.tensor(X_scaled_test,dtype=torch.float32)
y_pred = model(x_tensor)
y_test_pred = y_pred.detach().numpy()[:,0]


print('TEST Results')
accuracy = accuracy_score(y_test,y_test_pred.round())
r2 = r2_score(y_test,y_test_pred.round())
auc = roc_auc_score(y_test,y_test_pred.round())

print('Accuracy: '+str(accuracy))
print('R2:       '+str(r2))
print('AUC:      '+str(auc))

# Build dataframe to use in next part
test_df = pd.DataFrame(X_test, columns=data.feature_names)
test_df['y_true'] = y_test
test_df['y_pred_score'] = y_test_pred
test_df['y_pred'] = y_test_pred.round()

# show confusion matrix
print(pd.crosstab(test_df['y_true'],test_df['y_pred'], rownames=['Actual'], colnames=['Predicted']))





# ----------------------------------------
# --- STEP 2: Isolate Inaccurate Cluster
# ----------------------------------------
print('\nSTEP 2: Isolating Innaccurate Cluster ----------------------- ')

# We want to find a cluster with the lowest accuracy
 # - h

    
# Create several different clusters, then check the accuracy of all of them
def basicClustering(x,y_true,y_pred,k,trial_cnt):
    print('hi')
    sys.stdout.flush()
    result_accuracy=[1]*k
    result_clusters=1
    
    for t in range(10):
        accuracy=[None]*k
        size=[None]*k

        print('about to cluster')
        sys.stdout.flush()
        kmeans = KMeans(k)
        clusters = kmeans.fit(x) 

        # print('y_true shape: '+str(y_true.shape))
        # print('x shape: '+str(x.shape))
        sys.stdout.flush()
        
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
    kmeans = KMeans(k)
    clusters = kmeans.fit(x)
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

print(' using worst cluster search...')
accuracy,center,mask = subClustering(X_scaled,y_true,y_pred,K,MIN_CLUSTER_SIZE,K_TRIAL_CNT)
print('  lowest accuracy: '+str(accuracy))

print('For next part will use cluster with lowest accuracy from worst cluster search:')
print(' accuracy: '+str(accuracy))
print(' size: '+str(sum(mask)))

# ----------------------------------------
# --- STEP 3: Identify Features
# ----------------------------------------
print('\nSTEP 3: Identify Features ----------------------------------- ')


