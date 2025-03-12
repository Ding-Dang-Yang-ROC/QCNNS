import os,sys
import numpy as np
import torch
from Global import *

def Read_MNIST(data):
  fr=open("../../../../../1_data/fmnist/%d_%d/fmnist_%s_%d_%d"%(N_fn,N_fn,data,N_fn,N_fn),"r")
  img=[]
  trg=[]
  for line in fr:
    lx=line.split()
    if len(lx)==1:
      trg.append(int(lx[0]))
    if len(lx)>3:
      for i in range (len(lx)):
        img.append(float(lx[i]))
  fr.close()
  img2=torch.tensor(img).reshape(-1,N_fn*N_fn)
  del img
  return trg,img2

def One_Hot(target):
  t=[]
  for i in range (len(target)):
    if target[i]==0:
      t=t+[1,0,0,0,0,0,0,0,0,0]
    if target[i]==1:
      t=t+[0,1,0,0,0,0,0,0,0,0]
    if target[i]==2:
      t=t+[0,0,1,0,0,0,0,0,0,0]
    if target[i]==3:
      t=t+[0,0,0,1,0,0,0,0,0,0]
    if target[i]==4:
      t=t+[0,0,0,0,1,0,0,0,0,0]
    if target[i]==5:
      t=t+[0,0,0,0,0,1,0,0,0,0]
    if target[i]==6:
      t=t+[0,0,0,0,0,0,1,0,0,0]
    if target[i]==7:
      t=t+[0,0,0,0,0,0,0,1,0,0]
    if target[i]==8:
      t=t+[0,0,0,0,0,0,0,0,1,0]
    if target[i]==9:
      t=t+[0,0,0,0,0,0,0,0,0,1]
  t2=torch.tensor(t).reshape(len(target),10)
  return t2

def Read_Data():
  test_tar,test_img=Read_MNIST("test")
  test_tar_one=One_Hot(test_tar)
  train_tar,train_img=Read_MNIST("train")
  train_tar_one=One_Hot(train_tar)
  return test_tar,test_tar_one,test_img,train_tar,train_tar_one,train_img

def Correct(epoch,data,fw1,pred,target):
  count=0
  fw1.write("%-10s %6d\n"%(data,epoch))
  for i in range (len(target)):
    fw1.write("%3d %3d   "%(pred[i],target[i]))
    if i%10==9:
      fw1.write("\n")
    if pred[i]==target[i]:
      count=count+1
  if i%10!=9:
    fw1.write("\n")
  per=100*count/len(target)
  fw1.write("\n")
  fw1.flush()
  return per

def OutAcc(i,fw,acc_tran,acc_test):
  fw.write("%8d %7.2f %7.2f\n"%(i,acc_tran,acc_test))
  fw.flush()


