import numpy as np
import csv
import matplotlib.pylab as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from copy import copy
import time
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import rbf_kernel

times=[]

class Regression():
    
    def __init__(self):
        pass
        
    def regularization_loss(self,weights,lamda):
        weights=weights[1:]
        return 0.5*lamda*(sum([i**2 for i in weights]))
    
    def mse_loss(self,predicted,goldset):
        diff=goldset-predicted

        return 0.5* (sum([i**2 for i in diff]))/diff.shape[0]
            
    def total_loss(self,predicted,goldset,lamda):
        return self.mse_loss(predicted,goldset) #+ self.regularization_loss(weights,lamda)
    
    
    def computeGramMatrix(self,X,kernel,sigma):
       
        mat=np.zeros((X.shape[0],X.shape[0]))
        kp=[]
        for i in range(0,X.shape[0]):
            for j in range(0,X.shape[0]):
                norm1=-1*np.linalg.norm(X[i]-X[j])**2
                if i!=j:
                    kp.append(np.sqrt(-1*norm1))
                norm1=norm1/(2*(sigma**2))
                norm1=np.exp(norm1)
                mat[i,j]=norm1
        
        print 'min of kp is ',min(kp)
        print 'max of kp is ',max(kp)
        return mat
            
             
    def kernelStar(self,Xtest,kernel,sigma):
        pdist=[]   
        X=self.X_train
        for i in range(0,X.shape[0]):
           
            
            norm1=np.linalg.norm(X[i]-Xtest)**2
            
            norm1=norm1/(2*sigma**2)
            pdist.append(np.exp(-1*norm1))
           
         
        pdist=np.asarray(pdist)     
        
        return pdist
                

     
    #def grams2(self,X,sigma):
           
     
    def test(self,X,kernel,sigma):
        
        X=np.hstack((np.ones((X.shape[0],1)),X))
        #change here to change the algorithm
        predicted=[]
        for i in range(0,X.shape[0]):
            kstar=self.kernelStar(X[i],kernel,sigma)
            predicted.append(np.dot(kstar,np.dot(np.linalg.inv(self.GramMatrix),self.y_train)))
        
        predicted=np.asarray(predicted)
        return predicted
    
 
        
    def train(self,X,y,kernel,sigma):
     
        self.y_train=y
        #now we have to incorporate the desired Feature Vectors
        X=np.hstack((np.ones((X.shape[0],1)),X))
        
        self.X_train=copy(X)
        #change here for Differnt Algorithm
        self.GramMatrix=self.computeGramMatrix(X,kernel,sigma)
        self.GramMatrix=self.GramMatrix+((sigma**2))*np.eye(self.GramMatrix.shape[0])
        #print 'max of gram matrix is ',max(self.GramMatrix.ravel())
       
        
        
    def crossValidate(self,K=10,lamda=1,sigma=1):
        start_time = time.time()
        datasetsX=[]
        labelsy=[]
        for i in range(1,11):
            Xname="Regression Dataset/fData"+str(i)+".csv"
            fx=open(Xname)
            xReader=csv.reader(fx)
            X=[]
            y=[]
            for row in xReader:
                row=[float(x) for x in row]
                X.append(row)
            X=np.asarray(X)
            datasetsX.append(X)
            
            yname="Regression Dataset/fLabels"+str(i)+'.csv'
            fy=open(yname)
            y=[]
            yReader=csv.reader(fy)
            for row in yReader:
                row=[float(x) for x in row]
                #print 'row is ',row[0]
                y.append(row[0])
            y=np.asarray(y)
            labelsy.append(y)
            
            
        
        #now we make cross validation datasets
        kcross=[]
        for i in range(0,K):
            kcross.append(i)
        for i in range(0,K-1):
            kcross.append(i)
        losses=[]
        for i in range(0,K):
            testX=datasetsX[kcross[i]]
            testY=labelsy[kcross[i]]
            
            r2s=[]
            trainX=datasetsX[kcross[i+1]]
            trainy=labelsy[kcross[i+1]]
            for j in range(i+2,i+K):
                trainX=np.vstack((trainX,datasetsX[kcross[j]]))
                trainy=np.hstack((trainy,labelsy[kcross[j]]))
            kernel=2
            self.train(trainX,trainy,kernel,sigma)
            #print 'Now testing------>'
            predicted=self.test(testX,kernel,sigma)
            predicted_train=self.test(trainX,kernel,sigma)
            meany=np.mean(trainy)
            diff2=predicted_train - meany
            Stt=sum([l**2 for l in diff2])
            Ssm=self.mse_loss(predicted_train,trainy)*2
            r2s.append(Ssm/Stt)
            loss=self.total_loss(predicted,testY,lamda)
          
            losses.append(loss)
        print 'total loss with is ',np.mean(losses),' mean r2 is ',np.mean(r2s)  
        print 'for sigma = ',sigma
        times.append(time.time() - start_time)
        print("--- %s seconds ---" % (time.time() - start_time))        
        return  np.mean(losses) 
        
        
        

lsr=Regression()
sigmas=range(1,7)
losses=[]

for sigma in sigmas:
    losses.append(lsr.crossValidate(K=10,lamda=1,sigma=sigma))


plt.plot(sigmas,losses) 
plt.xlabel('Standard Deviation  in Gaussian Process Regression')
plt.ylabel('Mean loss in each K fold iteration')
plt.title(' Accuracy with increasing Standard Deviation in Gaussian Process Regression with Gaussian Kernel')
plt.show()        


plt.plot(sigmas,times) 
plt.xlabel('Standard Deviation in Gaussian Process Regression')
plt.ylabel('Time in each K fold iteration')
plt.title(' Time taken with increasing Standard Deviation in Gaussian Process Regression with Gaussian Kernel')
plt.show()
