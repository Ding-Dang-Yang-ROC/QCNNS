import os,sys
import torch
import numpy as np
import random
import datetime
import pennylane as qml
from torch import optim
from Global import *
from Read_Data.read_data import *

dev = qml.device("default.qubit", wires=12)

def Para():
  par1=torch.rand((3,N_par),device=dev2,dtype=Dtype,requires_grad=True)
  par2=torch.rand((64*64,10),device=dev2,dtype=Dtype,requires_grad=True)
  return par1,par2

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
    img1=img[num[beg_frm:end_frm]].reshape(batch,2,4,2,4,2,4,2,4).permute(2,4,6,8,1,3,5,7,0).reshape(256,16*batch)
    pred0=torch.matmul(Con_Unitary(Par1[0]),img1).reshape(4,4,4,4,2,2,2,2,batch).permute(4,0,5,1,6,2,7,3,8).reshape(4,2,4,2,4,2,4,2,batch) \
      .permute(0,2,4,6,1,3,5,7,8).reshape(256,16*batch)
    pred1=torch.matmul(Con_Unitary(Par1[1]),pred0).reshape(4,4,4,4,2,2,2,2,batch).permute(0,4,1,5,2,6,3,7,8).reshape(2,2,2,2,2,2,2,2,2,2,2,2,batch) \
      .permute(0,2,3,5,6,8,9,11,1,4,7,10,12).reshape(256,16*batch)
    pred2=torch.matmul(Con_Unitary(Par1[2]),pred1).reshape(2,2,2,2,2,2,2,2,2,2,2,2,batch).permute(12,0,8,1,2,9,3,4,10,5,6,11,7).reshape(batch,64*64)
    pred3=pred2.pow(2)
    err=torch.matmul(pred3,Par2)-one_b
    loss=torch.square(err).sum()
    Opt.zero_grad()
    loss.backward()
    Opt.step()

def Pred_Pyt(img,par1,par2):
  pdig=[]
  sam=img.shape[0]
  n_batch=int(np.ceil(sam/Batch))
  num=np.arange(sam)
  batch=Batch
  cm0=Con_Unitary(par1[0])
  cm1=Con_Unitary(par1[1])
  for i in range (n_batch):
    beg_frm=i*Batch
    end_frm=(i+1)*Batch
    if end_frm>sam:
      end_frm=sam
      batch=end_frm-beg_frm
    img1=img[num[beg_frm:end_frm]].reshape(batch,2,4,2,4,2,4,2,4).permute(2,4,6,8,1,3,5,7,0).reshape(256,16*batch)
    pred0=torch.matmul(Con_Unitary(Par1[0]),img1).reshape(4,4,4,4,2,2,2,2,batch).permute(4,0,5,1,6,2,7,3,8).reshape(4,2,4,2,4,2,4,2,batch) \
      .permute(0,2,4,6,1,3,5,7,8).reshape(256,16*batch)
    pred1=torch.matmul(Con_Unitary(Par1[1]),pred0).reshape(4,4,4,4,2,2,2,2,batch).permute(0,4,1,5,2,6,3,7,8).reshape(2,2,2,2,2,2,2,2,2,2,2,2,batch) \
      .permute(0,2,3,5,6,8,9,11,1,4,7,10,12).reshape(256,16*batch)
    pred2=torch.matmul(Con_Unitary(Par1[2]),pred1).reshape(2,2,2,2,2,2,2,2,2,2,2,2,batch).permute(12,0,8,1,2,9,3,4,10,5,6,11,7).reshape(batch,64*64)
    pred3=pred2.pow(2)
    pred4=torch.matmul(pred3,par2)
    dig=torch.argmax(pred4,dim=1).to("cpu").tolist()
    pdig=pdig+dig
  del cm0,cm1
  return pdig

@qml.qnode(dev,interface="torch")
def quantum_circuit(input_state,cm0,cm1,cm2):
  qml.QubitStateVector(input_state.numpy(),wires=[0,1,2,3,4,5])
  qml.QubitStateVector(input_state.numpy(),wires=[6,7,8,9,10,11])
  qml.QubitUnitary(cm0,wires=[1,2,4,5,7,8,10,11])
  qml.QubitUnitary(cm1,wires=[0,1,3,4,6,7,9,10])
  qml.QubitUnitary(cm2,wires=[0,2,3,5,6,8,9,11])
  return qml.probs(wires=range(12))

def Pred_Pen(img, par1,par2):
  pdig=[]
  cm0=Con_Unitary(par1[0])
  cm1=Con_Unitary(par1[1])
  cm2=Con_Unitary(par1[2])
  for i in range(img.shape[0]):
    pred0=quantum_circuit(img[i],cm0,cm1,cm2)
    pred1 = pred0.clone().to(dtype=Dtype)
    pred2=torch.matmul(pred1,par2)
    dig=torch.argmax(pred2).to("cpu")
    pdig.append(dig)
  del cm0,cm1,cm2
  return pdig

def Con_Unitary(par):
  U,S,VT=torch.linalg.svd(par.reshape(S_Ker,S_Ker))
  Q=torch.matmul(U,VT).to(dev2)
  del U,S,VT
  return Q

def Main():
  global Par1,Par2,Opt
  Par1,Par2=Para()
  test_tar,test_one,test_img_pen,test_img_pyt,train_tar,train_one,train_img_pen,train_img_pyt=Read_Data()
  Opt=optim.SGD([Par1,Par2],lr=learning_rate,momentum=0.9)
  fw1=open("Result/accu_case_ori","w")
  fw2=open("Result/accu_sum","w")
  fw2.write("   epoch   train    test\n")
  for i in range (N_Ite):
    Train_Pyt(train_one,train_img_pyt)
    if i%10==9:
      pdig_train_pyt=Pred_Pyt(train_img_pyt,Par1.detach(),Par2.detach())
      pdig_test_pyt=Pred_Pyt(test_img_pyt,Par1.detach(),Par2.detach())
      acc_tran=Correct(i,"train",fw1,pdig_train_pyt,train_tar)
      acc_test=Correct(i,"test",fw1,pdig_test_pyt,test_tar)
      OutAcc(i,fw2,acc_tran,acc_test)
  pdig_train_pen=Pred_Pen(train_img_pen,Par1.detach(),Par2.detach())
  pdig_test_pen=Pred_Pen(test_img_pen,Par1.detach(),Par2.detach())
  acc_tran_pen=Correct(1000,"train",fw1,pdig_train_pen,train_tar)
  acc_test_pen=Correct(1000,"test",fw1,pdig_test_pen,test_tar)
  OutAcc(1000,fw2,acc_tran_pen,acc_test_pen)
  OutPara(Par1.detach(),Par2.detach())
  fw1.close()
  fw2.close()

start=datetime.datetime.now()
Main()
end=datetime.datetime.now()
print("執行時間：",end-start)

