
# LIBRARIES
# LIBRARIES
import numpy as np
import pandas as pd
import random
import sys
import matplotlib.pyplot as plt
from operator import itemgetter

from sklearn import utils
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# PARAMETERS
IMG_ROOT = str(sys.argv[1])

INPUT_FILEPATH = str(sys.argv[2])

PERCENT_TEST = 0.33
N_NEIGHBORS = 5
P_MAX = 10 # THE MAX NUMBER OF FEATURES TO GRAB

# GET DATA
df = pd.read_csv(INPUT_FILEPATH)


y_cols = ['y_true','y_pred_score','y_pred']
clust_col = ['clust_mask']
X_df =  df.loc[:, ~df.columns.isin(y_cols+clust_col)]

X = X_df.to_numpy()
X_scaled = StandardScaler().fit_transform(X)
mask = df['clust_mask'].to_numpy()
cluster = X[mask]
custer_scaled = X_scaled[mask]

# SPLIT DATA INTO TRAIN/TEST
n_clust = np.sum(mask)
n_nclust = np.sum(np.invert(mask))
lda_X_clust = X_scaled[mask]
lda_y_clust = mask*1
lda_X_nclust = X_scaled[np.invert(mask)]
lda_y_nclust = np.invert(mask)*1


test_n = int(n_clust*PERCENT_TEST)
train_n = n_clust - test_n
test_i_clust = random.sample(range(n_clust),test_n)
test_i_nclust = random.sample(range(n_nclust),test_n)
train_i_clust = random.sample(range(n_clust),train_n)
train_i_nclust = random.sample(range(n_nclust),train_n)

X_train = np.concatenate((
    np.take(lda_X_clust,train_i_clust,axis=0),
    np.take(lda_X_nclust,train_i_clust,axis=0))
    ,axis=0)
y_train = np.concatenate((
    np.take(lda_y_clust,train_i_clust,axis=0),
    np.take(lda_y_nclust,train_i_nclust,axis=0)),
    axis=0)
X_train, y_train = utils.shuffle(X_train, y_train)

X_test = np.concatenate((
    np.take(lda_X_clust,test_i_clust,axis=0),
    np.take(lda_X_nclust,test_i_nclust,axis=0))
    ,axis=0)
y_test = np.concatenate((
    np.take(lda_y_clust,test_i_clust,axis=0),
    np.take(lda_y_nclust,test_i_nclust,axis=0)),
    axis=0)
X_test, y_test = utils.shuffle(X_test, y_test)

print("Split Train/test Using Undersampling")
print(" samples in full dataset: "+str(X.shape[0]))
print(" samples in cluster: "+str(n_clust))
print(" samples in training: "+str(X_train.shape[0]))
print(" samples in test: "+str(X_test.shape[0]))
print()


# LDA ------------------------------------------
print('Using LDA --------------------------------------------')
clf = LDA()
clf.fit(X_train, y_train)
clf.coef_[0,:].shape

coef = clf.coef_[0,:]
lda_result_df = pd.DataFrame()
lda_result_df['i'] = np.absolute(coef).argsort()[::-1]
lda_result_df['col'] = X_df.columns[lda_result_df['i']]
lda_result_df['val'] = coef[lda_result_df['i']]

y_train_pred = clf.predict(X_train)
# (1*least_mask)

acc = accuracy_score(y_train,y_train_pred)
print('Train accuracy:',acc)

y_test_pred = clf.predict(X_test)
# (1*least_mask)

acc = accuracy_score(y_test,y_test_pred)
print('Test accuracy:',acc)

print(pd.crosstab(y_test,y_test_pred, rownames=['Actual'], colnames=['Predicted']))

print("Largest Coef:")

print(lda_result_df[['col','val']].head(P_MAX).to_string(index=False))



print('Using KNN --------------------------------------------')

# def forwardStepwise_knearest(X_train,X_test,y_train,y_test,k,p_max,col_i,col_names):
def forwardStepwise_knearest(X_train,X_test,y_train,y_test,k,p_max,col_names,results=[]):
    if np.shape(results)[0]> 0: col_i = list(map(itemgetter('i'), results))
    else: col_i=[]
    col_cnt = X_train.shape[1]
    cols_to_try = list(range(col_cnt))
    cols_to_try = list(set(cols_to_try) - set(col_i))
    
    test_acc = [0]*col_cnt
    
    for i in cols_to_try:
    
        temp_col_i = col_i.copy()
        temp_col_i.append(i)
        
        this_train_X = np.take(X_train,temp_col_i,axis=1)
        this_test_X  = np.take(X_test,temp_col_i,axis=1)
        
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(this_train_X, y_train)
        y_test_pred = neigh.predict(this_test_X)

        test_acc[i]=accuracy_score(y_test,y_test_pred)
        
    best_acc = max(test_acc)
    best_acc_i = test_acc.index(best_acc)
    
    # col_i.append(best_acc_i)
    results.append(
        {
            'i': best_acc_i,
            'added_column': col_names[best_acc_i],
            'accuracy':  best_acc
        }
    )
    
    # print('best subset:',np.take(col_names,col_i).to_list())
    # print('  acc:',best_acc)
    
    # if len(col_i)<p_max:
        # col_i = forwardStepwise_knearest(X_train,X_test,y_train,y_test,k,p_max,col_i,col_names)
    if np.shape(results)[0]<p_max:
        results = forwardStepwise_knearest(X_train,X_test,y_train,y_test,k,p_max,col_names,results)
    
    return(results)


        
# k = 3
# p_max = 10
col_i = []
col_names = X_df.columns
knn_results = forwardStepwise_knearest(X_train,X_test,y_train,y_test,N_NEIGHBORS,P_MAX,col_names)
knn_results = pd.DataFrame(knn_results)
print(knn_results[['added_column','accuracy']].to_string(index=False))


def getEQ(y_true,y_pred,prot):
    prot_vals = np.unique(prot)
    y_true_pos = y_true==1
    y_pred_pos = y_pred==1
    prot_0 = prot == 0
    prot_1 = prot ==1

    pos_0 = sum((y_true_pos & prot_0)*1)
    # tpr_0 = sum((y_pred_pos & y_true_pos & prot_0)*1)/pos_0

    pos_1 = sum((y_true_pos & prot_1)*1)
    # tpr_1 = sum((y_pred_pos & y_true_pos & prot_1)*1)/pos_1
    
    if pos_0==0 or pos_1==0:result = 'nan'
    else: 
        tpr_0 = sum((y_pred_pos & y_true_pos & prot_0)*1)/pos_0
        tpr_1 = sum((y_pred_pos & y_true_pos & prot_1)*1)/pos_1
        result = np.abs(tpr_0-tpr_1)

    return(result)

print()
print("Calculating Equality of Opportunity for top binary LDA features")
for i in lda_result_df.head(P_MAX)['i']:
    col_name = X_df.columns[i]
    if np.unique(X[:,i]).shape[0] ==2:
        eq = getEQ(df['y_true'].to_numpy(),df['y_pred'].to_numpy(),X[:,i])
        print(str(col_name)+', eq: '+str(eq))
    else: print(str(col_name)+', not binary')

print()
print("Calculating Equality of Opportunity for top binary KNN features")
for i in knn_results['i']:
    col_name = X_df.columns[i]
    if np.unique(X[:,i]).shape[0] ==2:
        eq = getEQ(df['y_true'].to_numpy(),df['y_pred'].to_numpy(),X[:,i])
        print(str(col_name)+', eq: '+str(eq))
    else: print(str(col_name)+', not binary')

# knn_top_i = knn_results['i'].iloc[0]
# knn_top_col = knn_results['added_column'].iloc[0]
# knn_top = X[:,knn_top_i]
# if np.unique(knn_top).shape[0] ==2: 
#     print('Top knn col is binary')
#     print(getEQ(df['y_true'].to_numpy(),df['y_pred'].to_numpy(),knn_top))
    

# knn_top_min = knn_top.min()
# knn_top_max = knn_top.max()
# knn_top_group = ((knn_top >= knn_top_min) & (knn_top <= knn_top_max))*1

# print(knn_top_col)
# print(knn_top_min)
# print(knn_top_max)
# print(knn_top_group)
# print(sum(knn_top_group))
# print(getEQ(y_true,y_pred,knn_top))

# def getEQ(y_true,y_pred,prot):
#     prot_vals = np.unique(prot)
#     pos_0 = sum((y_true==1 & prot == 0)*1)
#     tpr_0 = sum((y_pred==1 & y_true==1 & prot == 0)*1)/pos_0

#     pos_1 = sum((y_true==1 & prot == 1)*1)
#     tpr_1 = sum((y_pred==1 & y_true==1 & prot == 1)*1)/pos_1

#     return(np.abs(tpr_0-tpr1))

