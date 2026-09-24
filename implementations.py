import numpy as np

def compute_MSEgrad(y,tx,w):
    err= y- tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def compute_MSEloss(y,tx,w):
    return np.sum((y-tx@w)**2)/(len(y)*2)


def mean_squared_error_gd(y, tx, initial_w,max_iters, gamma):
    w=initial_w
    counter=0
    while (counter<max_iters): #eventually implement early stopping by adding OR
        counter+=1
        w=w-gamma*compute_MSEgrad(y,tx,w)
    final_loss=compute_MSEloss(y,tx,w)
    return (w,final_loss)
    




