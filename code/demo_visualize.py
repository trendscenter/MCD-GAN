import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio


###### Origin #########################

data_all = scio.loadmat('result/Origin_Demo_data_category_domain.mat')
data = data_all['data']
data_domain1 = data[:1000, :]
data_domain2 = data[1000:, :]
label_domain1 = data_all['category_label'][:1000]
label_domain2 = data_all['category_label'][1000:]

data_domain1_class1 = data_domain1[np.where(label_domain1>0.5)[0],:]
data_domain1_class2 = data_domain1[np.where(label_domain1<0.5)[0],:]
data_domain2_class1 = data_domain2[np.where(label_domain2>0.5)[0],:]
data_domain2_class2 = data_domain2[np.where(label_domain2<0.5)[0],:]

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
Domain1_Class1 = ax.scatter(data_domain1_class1[:, 0], data_domain1_class1[:, 1], color='red', label='Domain1_Class1')
Domain1_Class2 = ax.scatter(data_domain1_class2[:, 0], data_domain1_class2[:, 1], color='green', label='Domain1_Class2')
Domain2_Class1 = ax.scatter(data_domain2_class1[:, 0], data_domain2_class1[:, 1], color='orange', label='Domain2_Class1')
Domain2_Class2 = ax.scatter(data_domain2_class2[:, 0], data_domain2_class2[:, 1], color='blue', label='Domain2_Class2')
ax.legend()
ax.set_title('Origin')
fig.savefig('result/Origin_Demo.png')
plt.show()


###### ComBat #########################

data_all = scio.loadmat('result/ComBat_Demo_data_category_domain.mat')
data = data_all['data']
data_domain1 = data[:1000,:]
data_domain2 = data[1000:,:]
label_domain1 = data_all['category_label'][:1000]
label_domain2 = data_all['category_label'][1000:]

data_domain1_class1 = data_domain1[np.where(label_domain1>0.5)[0],:]
data_domain1_class2 = data_domain1[np.where(label_domain1<0.5)[0],:]
data_domain2_class1 = data_domain2[np.where(label_domain2>0.5)[0],:]
data_domain2_class2 = data_domain2[np.where(label_domain2<0.5)[0],:]

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
Domain1_Class1 = ax.scatter(data_domain1_class1[:, 0], data_domain1_class1[:, 1], color='red', label='Domain1_Class1')
Domain1_Class2 = ax.scatter(data_domain1_class2[:, 0], data_domain1_class2[:, 1], color='green', label='Domain1_Class2')
Domain2_Class1 = ax.scatter(data_domain2_class1[:, 0], data_domain2_class1[:, 1], color='orange', label='Domain2_Class1')
Domain2_Class2 = ax.scatter(data_domain2_class2[:, 0], data_domain2_class2[:, 1], color='blue', label='Domain2_Class2')
ax.legend()
ax.set_title('ComBat')
fig.savefig('result/ComBat_Demo.png')
plt.show()


###### CycleGAN #########################
data_all = scio.loadmat('result/CycleGAN_Demo_data_category_domain.mat')
data = data_all['data']
data_domain1 = data[:1000,:]
data_domain2 = data[1000:,:]
label_domain1 = data_all['category_label'][:1000]
label_domain2 = data_all['category_label'][1000:]

data_domain1_class1 = data_domain1[np.where(label_domain1>0.5)[0],:]
data_domain1_class2 = data_domain1[np.where(label_domain1<0.5)[0],:]
data_domain2_class1 = data_domain2[np.where(label_domain2>0.5)[0],:]
data_domain2_class2 = data_domain2[np.where(label_domain2<0.5)[0],:]

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
Domain1_Class1 = ax.scatter(data_domain1_class1[:, 0], data_domain1_class1[:, 1], color='red', label='Domain1_Class1')
Domain1_Class2 = ax.scatter(data_domain1_class2[:, 0], data_domain1_class2[:, 1], color='green', label='Domain1_Class2')
Domain2_Class1 = ax.scatter(data_domain2_class1[:, 0], data_domain2_class1[:, 1], color='orange', label='Domain2_Class1')
Domain2_Class2 = ax.scatter(data_domain2_class2[:, 0], data_domain2_class2[:, 1], color='blue', label='Domain2_Class2')
ax.legend()
ax.set_title('CycleGAN')
fig.savefig('result/CycleGAN_Demo.png')
plt.show()


###### MCD-GAN #########################
data_all = scio.loadmat('result/MCDGAN_Demo_discrep_control_3.2_data_category_domain.mat')
data = data_all['data']
data_domain1 = data[:1000,:]
data_domain2 = data[1000:,:]
label_domain1 = data_all['category_label'][:1000]
label_domain2 = data_all['category_label'][1000:]

data_domain1_class1 = data_domain1[np.where(label_domain1>0.5)[0],:]
data_domain1_class2 = data_domain1[np.where(label_domain1<0.5)[0],:]
data_domain2_class1 = data_domain2[np.where(label_domain2>0.5)[0],:]
data_domain2_class2 = data_domain2[np.where(label_domain2<0.5)[0],:]

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
Domain1_Class1 = ax.scatter(data_domain1_class1[:, 0], data_domain1_class1[:, 1], color='red', label='Domain1_Class1')
Domain1_Class2 = ax.scatter(data_domain1_class2[:, 0], data_domain1_class2[:, 1], color='green', label='Domain1_Class2')
Domain2_Class1 = ax.scatter(data_domain2_class1[:, 0], data_domain2_class1[:, 1], color='orange', label='Domain2_Class1')
Domain2_Class2 = ax.scatter(data_domain2_class2[:, 0], data_domain2_class2[:, 1], color='blue', label='Domain2_Class2')
ax.legend()
ax.set_title('MCD-GAN')
fig.savefig('result/MCDGAN_Demo.png')
plt.show()

