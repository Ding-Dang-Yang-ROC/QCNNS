import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

from torchvision import datasets
from PIL import Image

transform = transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor()])
train_data = datasets.FashionMNIST(root='./data', train=True,download=True, transform=transform)
test_data = datasets.FashionMNIST(root='./data', train=False,download=True, transform=transform)

def data_pre_pro(img,img_size):
  npimg=img.numpy()
  image=np.asarray(npimg[0]*255,np.uint8)
  im=Image.fromarray(image,mode="L")
  im=im.resize((img_size,img_size),Image.BILINEAR)
  trans_to_tensor=transforms.ToTensor()
  return trans_to_tensor(im)

def Resiz(m,n,train_data,test_data):
  img_list=[]
  target_list=[]
  if m==0:
    data=train_data
  elif m==1:
    data=test_data
  for batch_idx,(img,target) in enumerate(data):
    target_list.append(target)
    img_resized=data_pre_pro(img,n).numpy()
    for i in range (n):
      for j in range (n):
        img_list.append(np.float64(img_resized[0][i][j]))
  return target_list,img_list

def Normalize(n,img2):
  img3=np.zeros((img2.shape[0],img2.shape[1],img2.shape[2]))
  for i in range (img2.shape[0]):
    sum2=0
    for j in range (img2.shape[1]):
      for k in range (img2.shape[2]):
        sum2=sum2+np.float64(img2[i][j][k])*np.float64(img2[i][j][k])
    sum=np.sqrt(sum2)
    for j in range (img2.shape[1]):
      for k in range (img2.shape[2]):
        img3[i][j][k]=np.float64(img2[i][j][k])/sum
  return img3

def Output(x,y,m,n):
  if m==0:
    s="train"
  elif m==1:
    s="test"
  fw=open("fmnist_%s_%d_%d"%(s,n,n),"w")
  fw.write("%6d %4d %4d\n"%(x.shape[0],x.shape[1],x.shape[2]))
  for m in range (x.shape[0]):
    fw.write("%s\n"%(y[m]))
    for i in range (x.shape[1]):
      for j in range (x.shape[2]):
        fw.write("%17.14f"%(x[m][i][j]))
      fw.write("\n")
    fw.write("\n")
  fw.close()

def Output_bin(x,y,m,n):
  if m==0:
    s="train"
  elif m==1:
    s="test"
  fw=open("fmnist_%s_%d_%d_bin"%(s,n,n),"w")
  fw.write("%6d %4d %4d\n"%(x.shape[0],x.shape[1],x.shape[2]))
  for m in range (x.shape[0]):
    fw.write("%s\n"%(y[m]))
    for i in range (x.shape[1]):
      for j in range (x.shape[2]):
        if x[m][i][j]>(1.0/n):
          fw.write("1 ")
        else:
          fw.write("0 ")
      fw.write("\n")
    fw.write("\n")
  fw.close()

def Convert(train_data,test_data):
  for m in range (0,2):
    if m==0:
      s="train"
    elif m==1:
      s="test"
    for p in range (2,6,1):
      n=np.power(2,p)
      target,img=Resiz(m,n,train_data,test_data)
      img2=np.reshape(img,(-1,n,n))
      y=np.reshape(target,(-1))
      x=Normalize(n,img2)
      Output(x,y,m,n)
      Output_bin(x,y,m,n)

def Main():
  transform=transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor()])
  train_data=datasets.FashionMNIST(root='./data',train=True,download=True,transform=transform)
  test_data=datasets.FashionMNIST(root='./data',train=False,download=True,transform=transform)
  Convert(train_data,test_data)

Main()

