- 👋 Hi, I’m @Craquolette
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...

<!---
Craquolette/Craquolette is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score


#X,y=make_blobs(n_samples=100, n_features=2,centers=2, random_state=0)
X= np.array([[0.23,4],[0.12,2],[0.83,17],[0.65,6]])
y = np.array([0,0,1,1])
#X = np.zeros((100,2))
#y = np.zeros(100)

random.seed(1)
np.random.seed(2)

if True:
    X = list()
    y = list()

    with open("base_de_donnees.csv","r") as f:
        for l in f.readlines():
            t = l.split(',')
            X.append((float(t[1]),float(t[2])))
            if 'B' in t[0]:
                y.append(1)
            else:
                y.append(0)

    X = np.array(X)
    y = np.array(y)

else:

    for i in range(len(X)):
        if random.choice((True,False)):
            y[i]=0
            X[i][0] = random.gauss(0.3,0.2)
            X[i][1] = random.gauss(1,0.5)
        else:
            y[i]=1
            X[i][0] = random.gauss(0.7,0.5)
            X[i][1] = random.gauss(3,2)

y=y.reshape((y.shape[0],1))
print('dimension de x:', X.shape)
print('dimension de y:', y.shape)

#avec ça, on génère les points jaunes et verts, on fait de "dataset"


def initialisation(X):
    w= np.random.randn(X.shape[1],1)
    b = np.random.randn(1)
    return w,b

w,b=initialisation(X)

def model(X,w,b):
    Z=X.dot(w)+b
    A=1/(1+np.exp(-Z))
    return A

A=model(X,w,b)

def log_loss(A,y):
    return 1 / len(y) * np.sum(-y * np.log(A)-(1-y)*np.log(1-A))

def gradients(A,X,y):
    dw= 1/len(y)*np.dot(X.T,A-y)
    db= 1/len(y)*np.sum(A-y)
    return (dw,db)

dw,db=gradients(A,X,y)

def update(dw,db,w,b,learning_rate):
    w=w-learning_rate*dw
    b=b-learning_rate*db
    return (w,b)

def predict(X,w,b):
    A=model(X,w,b)
    return A >= 0.5

def artificial_neuron(X,y,learning_rate=0.1,nbr_iter=200):
    #initialisation de w et b
    w,b=initialisation(X)
    Loss=[]
    for i in range(nbr_iter):
        A=model(X,w,b)
        Loss.append(log_loss(A,y))
        dw,db=gradients(A,X,y)
        w,b=update(dw,db,w,b,learning_rate)
    y_pred=predict(X,w,b)
    print(accuracy_score(y,y_pred))
    plt.plot(Loss)
    plt.show()
    return w,b

w,b=artificial_neuron(X,y)

#new_plant= np.array([0.43,3])

x0 = np.linspace(0,6,10)
x1 = (-w[0]*x0 -b)/w[1]
plt.plot(x0,x1,c='orange')
plt.scatter(X[:,0],X[:,1],c=y,cmap='summer')


if True:
    X_new = list()
    y_new = list()
    score=0

    with open("valeurstest.csv","r") as h:
        for g in h.readlines():
            q = g.split(',')
            X_new.append((float(q[1]),float(q[2])))
            if 'B' in q[0]:
                y_new.append(3)
            else:
                y_new.append(4)
        X_new = np.array(X_new)
        y_new = np.array(y_new)
        y_est = predict(X_new, w, b)
        for i in range (50):
            if (y_est[i] == True and y_new[i] == 3) or (y_est[i] == False and y_new[i] == 4):
                score += 1
                plt.scatter(X_new[:,0], X_new[:,1], c='m')
            else:
                plt.scatter(X_new[:,0], X_new[:,1], c='k')


#plt.scatter(X_new[:, 0], X_new[:, 1], c=y_new, cmap='winter')

score  = score / 50


plt.show()
print("Score :",score,"Parametres :",w,b)

#print(predict(X_new,w,b))

