# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 17:02:43 2017

@author: Liyi Li
MM PCA
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 01:18:26 2017

@author: Liyi Li
high frequency data Visualization
PCA, KPCA, SPCA
"""
#####################################
#1 import all the modules
###############################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy.matlib
from sklearn.preprocessing import LabelEncoder
import itertools

#######################################
#2 define all the useful functions
###################################
#: center data function
def center_data(data):
    dim = data.shape
    data_matrix = np.mat(data)
    mean_data= np.mean(data_matrix,axis= 0)
    data_Mean=np.matlib.repmat((mean_data), dim[0], 1)
    data_center = np.mat(data)-data_Mean
    return data_center

#create confusion matrix plot

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, '{:.2f}'.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        else: 
            plt.text(j, i, '{:.0f}'.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()    

#plot ROC curve
def plot_roc(fpr, tpr, roc_auc):
    """Plots the ROC curve for the win probability model along with
    the AUC.
    """
    fig, ax = plt.subplots()
    ax.set(title='Receiver Operating Characteristic',
           xlim=[0, 1], ylim=[0, 1], xlabel='False Positive Rate',
           ylabel='True Positive Rate')
    ax.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.legend(loc='lower right')
    
#read the data file: MM complete data file
original_MM = pd.read_csv("D:/lly/2017MM/PHASE2/final_totoal/final_final_final.csv", sep=",")
print original_MM.head()
# change the format of label
label = LabelEncoder() 

original_MM.loc[:, 'result'] = label.fit_transform(original_MM.loc[:, 'result'].astype(str))
original_MM.loc[:, 'game_id'] = original_MM.loc[:,'game_id'].astype(str)
original_MM.loc[:, 'team1_id'] = original_MM.loc[:,'team1_id'].astype(str)
original_MM.loc[:, 'season'] = original_MM.loc[:,'season'].astype(str)
########################################################################################
#######DATA REDUCTION
##3 ALL task-relevant data 2002-2017 Preparation to learn from model first
##########################################################################################
X_all_his = original_MM.loc[original_MM['season'].isin(["2002","2003","2004","2005","2006", "2007","2008", "2009", "2010", "2011", "2012","2013","2014","2015", "2016", "2017"])]
X_all_his_data = X_all_his.iloc[:,11:101]
X_all_his_data_label= X_all_his.loc[:, 'result']
X_02_17_meta = X_all_his.loc[:,['game_id', 'season', 'result']]
X_02_17_meta.to_excel("D:/lly/2017MM/PHASE2/final_totoal/metadata02_17.xlsx")

#centering data 
#centering data
X_all_his_center = center_data(X_all_his_data)
#PCA ANALYSIS first
from sklearn.decomposition import PCA
from matplotlib import cm
pca = PCA()
#fit model
PCA_OUTPUT=pca.fit(X_all_his_center)
#project data into new coordinate system
X_pca = pca.fit_transform(X_all_his_center)

###########
#visualization of variances
###############
print "pca explained variance ratio:\n", PCA_OUTPUT.explained_variance_ratio_
plt.plot(PCA_OUTPUT.explained_variance_ratio_, marker='o', linestyle='--', color='r', label='variance ratio')
plt.xlabel('PCs of PCA')
plt.ylabel('pca explained variance ratio')
plt.title('PCs Variance Ratio of PCA for MM 2002-2017 data')
plt.legend()
plt.show()

# we need at least 99% explained variance ratios (this is actually very high)
cutoff=0.99
EVP= pca.explained_variance_ratio_
accumulated_explained_variance_ratio_ =np.cumsum(EVP)/np.sum(EVP)
print("\n accumulated explained variance ratios: \n")
print(accumulated_explained_variance_ratio_)

#do dimension reduction
k=-1; # default dimension
dim = X_all_his_center.shape
idx=(accumulated_explained_variance_ratio_>= cutoff)
for i in range(0,dim[1]):
    if(idx[i]==True):
        k=i
        break
K=k+1
print("\n We choose the first " + str(K) + " PCs for dimension reduction \n")
PC = PCA_OUTPUT.components_
np.savetxt("D:/lly/2017MM/PHASE2/final_totoal/PCA_PCs_2002-2017_Reduced.csv", PC[:, 0:K], delimiter=",")

reduced_data = X_pca[:,:K]
np.savetxt("D:/lly/2017MM/PHASE2/final_totoal/PCA_2002-2017_Reduced_data.csv", reduced_data, delimiter=",")
#################################
#visualization of new data from PCA-2d
###################################
reds = X_all_his_data_label == 0
blues = X_all_his_data_label == 1
greens = X_all_his_data_label == 2

#visualize of original data
fig = plt.figure()
#plt.subplot(1, 3, 1)
l1 = plt.plot(X_all_his_center[reds, 0], X_all_his_center[reds, 1],"ro", markersize=5)
l2 = plt.plot(X_all_his_center[blues, 0],X_all_his_center[blues, 1], "b^", alpha = 0.5)
l3 = plt.plot(X_all_his_center[greens, 0], X_all_his_center[greens, 1], "g+")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Original data")
plt.legend('LWN')
plt.grid('on')
plt.show()

#visualization of new data from PCA-2d
fig = plt.figure()
plt.plot(X_pca[reds, 0], X_pca[reds, 1], "ro", markersize=10)
plt.plot(X_pca[blues, 0], X_pca[blues, 1],"b^", alpha = 0.5)
plt.plot(X_pca[greens, 0], X_pca[greens, 1], "g+")
plt.xlabel('PC 1')
plt.title("2D-PCA plot")
plt.legend('LWN')
plt.draw()

#visualization of new data from PCA-3d
fig = plt.figure()
ax = Axes3D(fig)
plt.plot(X_pca[reds, 0], X_pca[reds, 1],X_pca[reds, 2], "ro", markersize=10)
plt.plot(X_pca[blues, 0], X_pca[blues, 1], X_pca[blues, 2],"b^", alpha = 0.5)
plt.plot(X_pca[greens, 0], X_pca[greens, 1],X_pca[greens, 2], "g+")
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
ax.set_title("3D-PCA plot")
ax.legend('LWN')
ax.view_init(elev=5., azim=60)
plt.show()

###################
#KPCA analysis
###################

#create KPCA object
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
kpca = KernelPCA(kernel = 'rbf', fit_inverse_transform = True)

take_percent = 0.995
rest_data, remove_data= train_test_split(data_center, test_size=take_percent, random_state=42)

print("\n training_data size:{:d}".format(len(rest_data)))


KPCA_OUTPUT = kpca.fit(X_train)
#new data in new coordinate system
X_kpca = kpca.fit_transform(X_train)

#2d-visualization-KPCA-data projection in higher dimensional space 
plt.figure()
plt.plot(X_kpca[:, 0], X_kpca[:, 1], "bo")
plt.title("newData under two PCs--KPCA")
plt.xlabel("$1^{st}$ PC")
plt.ylabel("$2^{nd}$ PC")
plt.grid('on')

# 3d-visualization-KPCA-data projection in higher dimensional space 
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_kpca[:, 0], X_kpca[:, 1], X_kpca[:, 2],c = training_data_label, cmap=cm.summer)
ax.set_xlabel('kernel PC 1')
ax.set_ylabel('kernel PC 2')
ax.set_zlabel('kernel PC 3')
ax.set_title("3D-KPCA")
plt.show()


##############################
#Sparse PCA analysis
################################
#Sparse Principal Components Analysis (SparsePCA)
#SparsePCA
"""
Finds the set of sparse components that can optimally reconstruct the data. 
The amount of sparseness is controllable by the coefficient of the L1 penalty, 
given by the parameter alpha.
"""
from sklearn.decomposition import SparsePCA
#SparsePCA(n_components=None, alpha=1, ridge_alpha=0.01, max_iter=1000, tol=1e-08, 
#method='lars', n_jobs=1, U_init=None, V_init=None, 
#verbose=False, random_state=None)
#method : {‘lars’, ‘cd’}
#alpha: higher value--sparser components
spca = SparsePCA(method = 'lars')
SPCA_OUTPUT = spca.fit(X_all_his_center)
X_spca = spca.fit_transform(X_all_his_center)
np.savetxt("D:/lly/2017MM/PHASE2/final_totoal/SPCA_MM_PCs.csv", SPCA_OUTPUT.components_, delimiter=",")


#2d-visualization-SPCA-data projection in higher dimensional space 
fig = plt.figure()
plt.plot(X_spca[reds, 0], X_spca[reds, 1], "ro",markersize=10)
plt.plot(X_spca[blues, 0], X_spca[blues, 1],"b^", alpha = 0.5)
plt.plot(X_spca[greens, 0], X_spca[greens, 1], "g+")
plt.legend('LWN')
plt.title("newData under two PCs--SPCA")
plt.xlabel("$1^{st}$ PC")
plt.ylabel("$2^{nd}$ PC")

plt.grid('on')

# 3d-visualization-KPCA-data projection in higher dimensional space 
fig = plt.figure()
ax = Axes3D(fig)
plt.plot(X_spca[reds, 0], X_spca[reds, 1],X_spca[reds, 2], "ro", markersize=10)
plt.plot(X_spca[blues, 0], X_spca[blues, 1], X_spca[blues, 2],"b^", alpha = 0.5)
plt.plot(X_spca[greens, 0], X_spca[greens, 1],X_spca[greens, 2], "g+")
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
ax.set_title("3D-PCA plot")
ax.legend('LWN')
ax.view_init(elev=5., azim=60)
plt.show()
##########################################################
#4 ML on reduced data--- training and testing 
##########################################################
#after combine metadata and reduced data
MM = pd.read_csv("D:/lly/2017MM/PHASE2/final_totoal/FINAL_reduced_data_2002-2017.csv", sep=",")

#str label
label = LabelEncoder() 

MM.loc[:, 'result'] = label.fit_transform(MM.loc[:, 'result'].astype(str))
MM.loc[:, 'game_id'] = MM.loc[:,'game_id'].astype(str)
MM.loc[:, 'season'] = MM.loc[:,'season'].astype(str)
#generate training data and testing data
X_pre = MM.loc[MM['season'].isin(["2002","2003","2004","2005","2006", "2007","2008", "2009", "2010", "2011", "2012","2013","2014","2015"])]

training_data = X_pre.iloc[:,3:]
training_data_label= X_pre.loc[:, 'result']


#testing data
X_pre2 = MM.loc[MM['season'].isin(["2016"])]

test_data = X_pre2.iloc[:,3:]
test_data_label= X_pre2.loc[:, 'result']
    
#centering data
X_train = center_data(training_data)
X_test = center_data(test_data)

#######################
#Prediction Models
######################
from sklearn import ensemble, metrics
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from matplotlib import cm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import auc, classification_report, roc_auc_score, accuracy_score, f1_score, log_loss, roc_curve, confusion_matrix, precision_score, recall_score

def vaules_output(test_data_label, y_predict):
    
    confusion = confusion_matrix(test_data_label, y_predict)
    accuracy = accuracy_score(test_data_label, y_predict)
    precision = precision_score(test_data_label, y_predict)
    recall= recall_score(test_data_label, y_predict)
    f1 = f1_score(test_data_label, y_predict)
    log_loss_scores = log_loss(test_data_label, y_predict)
    print 'confusion\n', confusion,"\n", 'accuracy\n', accuracy,"\n", 'precision\n', precision,"\n", "recall\n", recall,"\n", "f1\n", f1,"\n", "log_loss_scores\n", log_loss_scores
    #output =  pd.DataFrame({'confusion': confusion,'accuracy': accuracy,'precision': precision, "recall": recall, "f1": f1, "log_loss_scores": log_loss_scores})
    #print output
    
###MODEL 1 SVM
## training by using default parameter setting
"""
kernel = "linear", "poly", "rbf", "sigmoid"
Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels
keep others default
"""
clf = svm.SVC(C=1, kernel='rbf', degree=3, gamma="auto", coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)
clf.fit(X_train, training_data_label)

y_predict = clf.predict(X_test)

confusion = metrics.confusion_matrix(test_data_label, y_predict)

#mean accuracy of test data and labels
print(clf.score(X_test, test_data_label))


#Confusion Matrix
print confusion

#plot
np.set_printoptions(precision=4)

plt.figure(figsize=(15,8))
plt.subplot(211)

plot_confusion_matrix(confusion, classes=label.inverse_transform([0,1]),
                      title='Confusion matrix, without normalization')

plt.subplot(212)
plot_confusion_matrix(confusion, classes=label.inverse_transform([0,1]), normalize=True,
                      title='Normalized confusion matrix')



#####MODEL2 KNN
#create KNN object
from sklearn import neighbors
for i in range(2,40):
    clf = neighbors.KNeighborsClassifier(n_neighbors=i)
    clf.fit(X_train, training_data_label)
    y_predict = clf.predict(X_train)

    confusion = metrics.confusion_matrix(training_data_label, y_predict)
    print i
    #mean accuracy of test data and labels
    print(clf.score(X_train, training_data_label))


    #Confusion Matrix
    print confusion

#MODEL chosen: NN = 4
clf = neighbors.KNeighborsClassifier(n_neighbors=25)
clf.fit(X_train, training_data_label)
y_predict = clf.predict(X_test) # use to evaluate models
vaules_output(test_data_label, y_predict)
confusion = metrics.confusion_matrix(test_data_label, y_predict)

#mean accuracy of test data and labels
print(clf.score(X_train, training_data_label))

#plot ROC curve
fpr, tpr, thresholds = roc_curve(test_data_label.values, y_predict)
roc_auc = auc(fpr, tpr)
plot_roc(fpr, tpr, roc_auc)

#plot confusion matrix
np.set_printoptions(precision=4)

plt.figure(figsize=(15,8))
plt.subplot(211)

plot_confusion_matrix(confusion, classes=label.inverse_transform([0,1]),
                      title='Confusion matrix, without normalization\n for KNN-3')

plt.subplot(212)
plot_confusion_matrix(confusion, classes=label.inverse_transform([0,1]), normalize=True,
                      title='Normalized confusion matrix\n for KNN-3') 
#############
#Similarly 
#MODEL 3 GradientBoostingClassifier
################

############
#MODEL 4 Random Forest Classifier
###############

##################################################
#Based on 2002-2016, to predict 2017
##################################################
MM = pd.read_csv("D:/lly/2017MM/PHASE2/final_totoal/FINAL_reduced_data_2002-2017.csv", sep=",")

#str label
label = LabelEncoder() 

MM.loc[:, 'result'] = label.fit_transform(MM.loc[:, 'result'].astype(str))
MM.loc[:, 'game_id'] = MM.loc[:,'game_id'].astype(str)
MM.loc[:, 'season'] = MM.loc[:,'season'].astype(str)
#generate training data and testing data
X_pre = MM.loc[MM['season'].isin(["2002","2003","2004","2005","2006", "2007","2008", "2009", "2010", "2011", "2012","2013","2014","2015", "2016"])]

training_data = X_pre.iloc[:,3:]
training_data_label= X_pre.loc[:, 'result']


#testing data
X_pre2 = MM.loc[MM['season'].isin(["2017"])]

test_data = X_pre2.iloc[:,3:]
test_data_label= X_pre2.loc[:, 'result']
    
#centering data
X_train = center_data(training_data)
X_test = center_data(test_data)

#MODEL 1 SVM
## training by using default parameter setting
"""
kernel = "linear", "poly", "rbf", "sigmoid"
Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels
keep others default
"""
clf = svm.SVC(C=1, kernel='sigmoid', degree=3, gamma="auto", coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)
clf.fit(X_train, training_data_label)
y_predict = clf.predict(X_train) # use to evaluate models
predictions = clf.predict(X_test) #targets

confusion = metrics.confusion_matrix(training_data_label, y_predict)
#mean accuracy of test data and labels
print(clf.score(X_train, training_data_label))

#Confusion Matrix
print confusion

np.set_printoptions(precision=4)

plt.figure(figsize=(15,8))
plt.subplot(211)

plot_confusion_matrix(confusion, classes=label.inverse_transform([0,1]),
                      title='Confusion matrix, without normalization\n for KNN')

plt.subplot(212)
plot_confusion_matrix(confusion, classes=label.inverse_transform([0,1]), normalize=True,
                      title='Normalized confusion matrix\n for KNN') 


##################
#get output file
##################
import pandas as pd

y_test_meta=X_pre2.iloc[:,0:2]
pd_y_predict = pd.DataFrame(predictions)
y_predict_pro = pd.DataFrame(clf.predict_proba(X_test))


y_test_meta.to_excel("D:/lly/2017MM/PHASE2/final_totoal/PCA_SVM_sig_update.xlsx")
y_predict_pro.to_excel("D:/lly/2017MM/PHASE2/final_totoal/PCA_SVM_sig_update-pro.xlsx")
pd_y_predict.to_excel("D:/lly/2017MM/PHASE2/final_totoal/PCA_SVM_sig_update-win-loss.xlsx")

####MODEL 2 KNN
#MODEL chosen: NN = 5
clf = neighbors.KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, training_data_label)
y_predict = clf.predict(X_train) # use to evaluate models
predictions = clf.predict(X_test) #targets

confusion = metrics.confusion_matrix(training_data_label, y_predict)
#mean accuracy of test data and labels
print(clf.score(X_train, training_data_label))

#Confusion Matrix
print confusion

np.set_printoptions(precision=4)

plt.figure(figsize=(15,8))
plt.subplot(211)

plot_confusion_matrix(confusion, classes=label.inverse_transform([0,1]),
                      title='Confusion matrix, without normalization\n for KNN')

plt.subplot(212)
plot_confusion_matrix(confusion, classes=label.inverse_transform([0,1]), normalize=True,
                      title='Normalized confusion matrix\n for KNN') 


##################
#get output file
##################
import pandas as pd

y_test_meta=X_pre2.iloc[:,0:2]
pd_y_predict = pd.DataFrame(predictions)
y_predict_pro = pd.DataFrame(clf.predict_proba(X_test))


y_test_meta.to_excel("D:/lly/2017MM/PHASE2/final_totoal/PCA_KNN_update.xlsx")
y_predict_pro.to_excel("D:/lly/2017MM/PHASE2/final_totoal/PCA_KNN_uodate-pro.xlsx")
pd_y_predict.to_excel("D:/lly/2017MM/PHASE2/final_totoal/PCA_KNN_update-win-loss.xlsx")



####MODEL 3 GradientBoostingClassifier
##############################
"""
#parameter: "n_estimators"--trees, "max_depth"--max depth of each tree,"min_sample_split"--min used to split an internal node

#parameter for classifier: "deviance"--binary classification(offer probability estimates); exponential"--less robust
#'n_estimators': 1000, 'max_depth': 4, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'exponential'
"""
clf = ensemble.GradientBoostingClassifier(n_estimators= 1000, max_depth= 4, min_samples_split= 3, learning_rate= 0.001, loss= 'deviance')
clf.fit(X_train, training_data_label)
y_predict = clf.predict(X_train) # use to evaluate models
predictions = clf.predict(X_test) #targets

confusion = metrics.confusion_matrix(training_data_label, y_predict)
#mean accuracy of test data and labels
print(clf.score(X_train, training_data_label))

#Confusion Matrix
print confusion

np.set_printoptions(precision=4)

plt.figure(figsize=(15,8))
plt.subplot(211)

plot_confusion_matrix(confusion, classes=label.inverse_transform([0,1]),
                      title='Confusion matrix, without normalization\n for GB')

plt.subplot(212)
plot_confusion_matrix(confusion, classes=label.inverse_transform([0,1]), normalize=True,
                      title='Normalized confusion matrix\n for GB') 


##################
#get output file
##################
import pandas as pd

y_test_meta=X_pre2.iloc[:,0:2]
pd_y_predict = pd.DataFrame(predictions)
y_predict_pro = pd.DataFrame(clf.predict_proba(X_test))


y_test_meta.to_excel("D:/lly/2017MM/PHASE2/final_totoal/PCA_GB_update.xlsx")
y_predict_pro.to_excel("D:/lly/2017MM/PHASE2/final_totoal/PCA_GB_uodate-pro.xlsx")
pd_y_predict.to_excel("D:/lly/2017MM/PHASE2/final_totoal/PCA_GB_update-win-loss.xlsx")

#####MODEL 4 Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

#{'n_estimators': 1000, 'criterion': "gini", 'max_depth': None,
#"min_samples_split": 3,'min_samples_leaf':3, 'bootstrap':True
params = [{'n_estimators': 1000, 'criterion': "gini", 'max_depth': None,
           "min_samples_split": 3,'min_samples_leaf':3, 'bootstrap':True
           },
          {'n_estimators': 1000, 'criterion': "gini", 'max_depth': 5,
           "min_samples_split": 3,'min_samples_leaf':1, 'bootstrap':False
           },
          {'n_estimators': 500, 'criterion': "gini", 'max_depth': None,
           "min_samples_split": 3,'min_samples_leaf':1, 'bootstrap':True
           },
          {'n_estimators': 1000, 'criterion': "entropy", 'max_depth': None,
           "min_samples_split": 3,'min_samples_leaf':3, 'bootstrap':True
           },
          {'n_estimators': 1000, 'criterion': "entropy", 'max_depth': 5,
           "min_samples_split": 3,'min_samples_leaf':3, 'bootstrap':False
           },
          {'n_estimators': 500, 'criterion': "entropy", 'max_depth': None,
           "min_samples_split": 3,'min_samples_leaf':1, 'bootstrap':True
           }
         ]

titles = ['gini-boostrap-1000trees','gini-nonboostrap-1000trees', "gini-boostrap-500trees",
          "entropy-boostrap-1000trees",
          "entropy-nonboostrap-1000trees", "entropy-boostrap-500trees"
         ]

plot_args = [{'c': 'red', 'linestyle': '-'},
             {'c': 'green', 'linestyle': '-'},
             {'c': 'blue', 'linestyle': '-'},
             {'c': 'red', 'linestyle': '--'},
             {'c': 'green', 'linestyle': '--'},
             {'c': 'blue', 'linestyle': '--'}
             ]

from sklearn.metrics import roc_curve
def plot_on_dataset(X, y, X_test,y_test):
    # for each dataset, plot learning for each learning strategy
    
    #max_iter = 400
    mlps = []

    for title, param in zip(titles, params):
        print("training: %s" % title)
        mlp = RandomForestClassifier(verbose=0, random_state=0,
                         **param)
        
        mlp.fit(X, y)
        mlps.append(mlp)
        print("Training set score: %f" % mlp.score(X, y))
        
        y_predict = mlp.predict(X_test)

        confusion = metrics.confusion_matrix(y_test, y_predict)
        
       
        np.set_printoptions(precision=2) 
        plt.figure(figsize=(15,8))
        plt.subplot(211)
        plot_confusion_matrix(confusion, classes=label.inverse_transform([0,1]),
                          normalize=False,
                          title= title,
                          cmap=plt.cm.Blues)
        plt.subplot(212)
        plot_confusion_matrix(confusion, classes=label.inverse_transform([0,1]), normalize=True,
                      title=title)
        
        plt.show()

plot_on_dataset(X_train,training_data_label,X_test,test_data_label)

#####
#model chosen:
clf= RandomForestClassifier(n_estimators = 1000, criterion =  "gini", max_depth = None, min_samples_split =  3, min_samples_leaf =1, bootstrap =True)
clf.fit(X_train, training_data_label)
y_predict = clf.predict(X_train) # use to evaluate models
predictions = clf.predict(X_test) #targets

confusion = metrics.confusion_matrix(training_data_label, y_predict)
#mean accuracy of test data and labels
print(clf.score(X_train, training_data_label))

#Confusion Matrix
print confusion

np.set_printoptions(precision=4)

plt.figure(figsize=(15,8))
plt.subplot(211)

plot_confusion_matrix(confusion, classes=label.inverse_transform([0,1]),
                      title='Confusion matrix, without normalization\n for RF-GINI-1000')

plt.subplot(212)
plot_confusion_matrix(confusion, classes=label.inverse_transform([0,1]), normalize=True,
                      title='Normalized confusion matrix\n for RF-GINI-1000') 


##################
#get output file
##################
import pandas as pd

y_test_meta=X_pre2.iloc[:,0:2]
pd_y_predict = pd.DataFrame(predictions)
y_predict_pro = pd.DataFrame(clf.predict_proba(X_test))


y_test_meta.to_excel("D:/lly/2017MM/PHASE2/final_totoal/PCA_RF_update.xlsx")
y_predict_pro.to_excel("D:/lly/2017MM/PHASE2/final_totoal/PCA_RF_update-pro.xlsx")
pd_y_predict.to_excel("D:/lly/2017MM/PHASE2/final_totoal/PCA_RF_update-win-loss.xlsx")

################################
#MODEL 5 mlp Classifier
###########################################
from sklearn.neural_network import MLPClassifier
"""
activation:{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
"""
clf = MLPClassifier(activation='logistic', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
clf.fit(X_train, training_data_label)
y_predict = clf.predict(X_train) # use to evaluate models
predictions = clf.predict(X_test) #targets

confusion = metrics.confusion_matrix(training_data_label, y_predict)
#mean accuracy of test data and labels
print(clf.score(X_train, training_data_label))

#Confusion Matrix
print confusion

np.set_printoptions(precision=4)

plt.figure(figsize=(15,8))
plt.subplot(211)

plot_confusion_matrix(confusion, classes=label.inverse_transform([0,1]),
                      title='Confusion matrix, without normalization\n for MLP-LOGIS')

plt.subplot(212)
plot_confusion_matrix(confusion, classes=label.inverse_transform([0,1]), normalize=True,
                      title='Normalized confusion matrix\n for MLP-LOGIS') 


##################
#get output file
##################
import pandas as pd

y_test_meta=X_pre2.iloc[:,0:2]
pd_y_predict = pd.DataFrame(predictions)
y_predict_pro = pd.DataFrame(clf.predict_proba(X_test))


y_test_meta.to_excel("D:/lly/2017MM/PHASE2/final_totoal/PCA_MLP_LOG_update.xlsx")
y_predict_pro.to_excel("D:/lly/2017MM/PHASE2/final_totoal/PCA_MLP_LOG_update-pro.xlsx")
pd_y_predict.to_excel("D:/lly/2017MM/PHASE2/final_totoal/PCA_MLP_LOG_update-win-loss.xlsx")


