import os,sys
import torch
import numpy as np
import random
import datetime
import pennylane as qml
from torch import optim
from Global import *
from Read_Data.read_data import *

dev = qml.device("default.mixed", wires=12)

@qml.qnode(dev,interface="torch")
def quantum_circuit(input_state,cm0,cm1):
  qml.QubitStateVector(input_state.numpy(),wires=[0,1,2,3,4,5])
  qml.QubitStateVector(input_state.numpy(),wires=[6,7,8,9,10,11])
  qml.QubitUnitary(cm0,wires=[1,2,4,5,7,8,10,11])
  qml.QubitUnitary(cm1,wires=[0,1,3,4,6,7,9,10])
  noise_level = 0.05
  phase_damping = 0.03
  for i in range(12):
    qml.DepolarizingChannel(noise_level, wires=i)
    qml.PhaseDamping(phase_damping, wires=i)
  return qml.probs(wires=range(12))

def Pred_Pen(img,par1,par2):
  pdig=[]
  cm0=Con_Unitary(par1[0])
  cm1=Con_Unitary(par1[1])
  for i in range(img.shape[0]):
    pred0=quantum_circuit(img[i],cm0,cm1)
    pred1 = pred0.clone().to(dtype=torch.float32)
    pred2=torch.matmul(pred1,par2)
    dig=torch.argmax(pred2).to("cpu")
    pdig.append(dig)
  del cm0,cm1
  return pdig

def Con_Unitary(par):
  U,S,VT=torch.linalg.svd(par.reshape(S_Ker,S_Ker))
  Q=torch.matmul(U,VT)
  del U,S,VT
  return Q

def Read_Par():
  with open('../../../../2_quantum/2_12_qubits/2_con4_2_L/test00/Result/par',"r") as f:
    data=[float(num) for line in f for num in line.strip().split()]
  par1=torch.tensor(data[:2*S_Ker*S_Ker],dtype=torch.float32).reshape(2,S_Ker*S_Ker)
  par2=torch.tensor(data[2*S_Ker*S_Ker:],dtype=torch.float32).reshape(64*64,10)
  return par1,par2

def Main():
  global Par1,Par2,Opt
  Par1,Par2=Read_Par()
  test_tar,test_one,test_img_pen,test_img_pyt=Read_Data()
  fw1=open("Result/accu_case_ori","w")
  fw2=open("Result/accu_sum","w")
  fw2.write("   epoch   test\n")
  pdig_test_pen=Pred_Pen(test_img_pen,Par1,Par2)
  acc_test_pen=Correct(1000,"test",fw1,pdig_test_pen,test_tar)
  OutAcc(1000,fw2,acc_test_pen)
  fw1.close()
  fw2.close()

start=datetime.datetime.now()
Main()
end=datetime.datetime.now()
print("執行時間：",end-start)

