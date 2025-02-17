import os,sys
import torch
import numpy as np
import random
import datetime
from torch import optim
from Global import *
from Read_Data.read_data import *

def Para():
  par=torch.rand((64,10),device=dev,dtype=Dtype,requires_grad=True)
  return par

def Train_Pyt(one,img):
  sam=one.shape[0]
  n_batch=int(np.ceil(sam/Batch))
  num=np.arange(sam)
  random.shuffle(num)
  batch=Batch
  for i in range (n_batch):
    beg_frm=i*Batch
    end_frm=(i+1)*Batch
    if end_frm>sam:
      end_frm=sam
      batch=end_frm-beg_frm
    one_b=one[num[beg_frm:end_frm]]
    pred0=img[num[beg_frm:end_frm]].reshape(batch,64).pow(2)
    err=torch.matmul(pred0,Par)-one_b
    loss=torch.square(err).sum()
    Opt.zero_grad()
    loss.backward()
    Opt.step()

def Pred_Pyt(img,par):
  pdig=[]
  sam=img.shape[0]
  n_batch=int(np.ceil(sam/Batch))
  num=np.arange(sam)
  batch=Batch
  for i in range (n_batch):
    beg_frm=i*Batch
    end_frm=(i+1)*Batch
    if end_frm>sam:
      end_frm=sam
      batch=end_frm-beg_frm
    pred0=img[num[beg_frm:end_frm]].reshape(batch,64)
    pred1=torch.matmul(pred0,par)
    dig=torch.argmax(pred1,dim=1).to("cpu").tolist()
    pdig=pdig+dig
  return pdig

def Main():
  global Par,Opt
  Par=Para()
  test_tar,test_one,test_img,train_tar,train_one,train_img=Read_Data()
  Opt=optim.SGD([Par],lr=learning_rate,momentum=0.9)
  fw1=open("Result/accu_case_ori","w")
  fw2=open("Result/accu_sum","w")
  fw2.write("   epoch   train    test\n")
  for i in range (N_Ite):
    Train_Pyt(train_one,train_img)
    if i%10==9:
      pdig_train=Pred_Pyt(train_img,Par.detach())
      pdig_test=Pred_Pyt(test_img,Par.detach())
      acc_tran=Correct(i,"train",fw1,pdig_train,train_tar)
      acc_test=Correct(i,"test",fw1,pdig_test,test_tar)
      OutAcc(i,fw2,acc_tran,acc_test)
  OutPara(Par.detach())
  fw1.close()
  fw2.close()

start=datetime.datetime.now()
Main()
end=datetime.datetime.now()
print("執行時間：",end-start)

