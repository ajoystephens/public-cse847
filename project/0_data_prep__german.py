
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
IMG_ROOT = str(sys.argv[1])
RESULT_FILEPATH = str(sys.argv[2])

PERCENT_TEST = 0.33 

BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS = 300


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


# print('\nSTEP 1: Train Initial Model --------------------------------- ')
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

test_df.to_csv(RESULT_FILEPATH,index=False) # save to csv

