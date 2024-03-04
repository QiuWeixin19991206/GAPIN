#main program
#import dependencies
import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras import Model
import numpy as np
import scipy.io
from pyDOE import lhs
import time
from G import Generator
from D import Discriminator
# from D_resnet import d_resnet
# from G_resnet import g_resnet
from tensorflow.keras.callbacks import ModelCheckpoint ,Callback
import os
import Wpenalty
from  tensorflow.keras import optimizers,metrics,layers,losses,datasets
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # 散点图
from scipy.interpolate import griddata
from datetime import datetime

tf.random.set_seed(123)
def Ld(d_real_logits,d_fake_logits):
    return tf.reduce_mean(1-d_real_logits) + d_fake_logits # 或许这里并不是损失函数？

def Lg(d_real_logits,d_fake_logits):
    return 1 + tf.reduce_mean(1-d_fake_logits) # 这里就是添加的物理公式，或许这里并不是损失函数？

def celoss_ones(logits):
    #热编码
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=tf.ones_like(logits))
    return tf.reduce_mean(loss)

def celoss_zeros(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.zeros_like(logits))
    return tf.reduce_mean(loss)

def d_loss_fn(generator, discriminator, x, y):
    # treat real image  as real
    # treat generator image as fake
    fake_lable = generator(x)
    d_fake_logits = discriminator(fake_lable)
    d_real_logits = discriminator(y)
    # d_real_logits = tf.nn.sigmoid(d_real_logits)
    # d_fake_logits = tf.nn.sigmoid(d_fake_logits)
    # ld = Ld(d_real_logits,d_fake_logits)
    # lg = Lg(d_real_logits,d_fake_logits)

    d_real_logits = celoss_ones(d_real_logits)
    d_fake_logits = celoss_zeros(d_fake_logits)
    gp = Wpenalty.Wpena(discriminator,real=y,fake=fake_lable)
    # loss = (-(tf.reduce_mean(d_real_logits) - tf.reduce_mean(d_fake_logits))+ 10 * gp + (0.001 * tf.reduce_mean(d_real_logits) ** 2))
    # loss = - tf.reduce_mean(tf.math.log(1.0 - (d_real_logits) + 1e-8) + tf.math.log((d_fake_logits) + 1e-8)) + 10 * gp
    loss = d_real_logits + d_fake_logits + 10 * gp #惩罚项系数大小按照要求给
    # loss = tf.reduce_mean(1 - d_real_logits) + tf.reduce_mean(d_fake_logits)
    # smooth = 0.9#平滑度
    # d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.zeros_like(d_fake_logits)))
    # d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits,labels=tf.ones_like(d_real_logits) * (1 - smooth)))
    # loss = d_loss_real + d_loss_fake +10 * gp
    return loss , gp #返回的是一个损失函数和一个惩罚项

def g_loss_fn(generator, discriminator, x, y):
    fake_lable = generator(x)
    d_fake_logits = discriminator(fake_lable)
    # loss = (-tf.reduce_mean(d_fake_logits)) + loss_pinn
    # loss = (-tf.reduce_mean(d_fake_logits)) + 10*loss_pinn
    loss = celoss_ones(d_fake_logits)
    # loss = - tf.reduce_mean(tf.math.log(1 - (d_fake_logits) + 1e-8))
    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.ones_like(d_fake_logits)))
    return loss , fake_lable

def standardsize(x,y):
  #对数据的预处理将数据控制在方差为1均值为0的范围中,归一化，这里的y不用编码
  mean = tf.reduce_mean(x)
  std = tf.math.reduce_std(x)
  x = (x - mean) / std
  y = tf.cast(y,dtype=tf.float32)
  # x = 2.0 * (x - (-9)) / ((9) - (-9)) - 1.0
  x = tf.cast(x, dtype=tf.float32)
  #y = tf.squeeze(y,axis=1)
  #y = tf.one_hot(y,depth=1)
  return x,y

def data_chul(x_train,y_train):

    x_train, y_train = standardsize(x_train, y_train) # 将数据(坐标)进行标准化，按照需求使用
    #x_test, y_test = standardsize(x_test, y_test)

    datas_train = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(10000) # 通过这里可以调节批次的大小
    datas_train = datas_train.shuffle(10000) # 打乱顺序 ， 可以不用这个代码

    #datas_test = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(256)
    #datas_test = datas_test.shuffle(10000)

    sample = next(iter(datas_train))
    print("sample",sample[0].shape,sample[1].shape,tf.reduce_min(sample[0]),tf.reduce_max(sample[0]))

    return datas_train #,datas_test
#使用余弦退火降低学习率 ，按照需求使用
class CosineAnnealingSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr_max, lr_min, T):
        super(CosineAnnealingSchedule, self).__init__()
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.T = T

    def __call__(self, step):

        t = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos((step/self.T) * np.pi))
        print("step{},lr;{}".format(step,t))
        return t

#set upper and lower spatial bounds
lb = np.array([-4, -1.0])
ub = np.array([3.96875, 1.0])
#load data from NLS.mat
data = scipy.io.loadmat(r'C:\Users\hcxy\Documents\Tencent Files\1940824907\FileRecv\新建文件夹\tf2\PINN_TF2_NLS-master\PINN_TF2_NLS-master\非简并双孤子\double.mat')

#切片和分配数据：
t = data['tt'].flatten()[:,None]
x = data['z'].flatten()[:,None]
print(x.shape, t.shape)
Exact1 = data['uu1']
Exact2 = data['uu2']
Exact_u1 = np.real(Exact1)
Exact_v1 = np.imag(Exact1)
Exact_h1 = np.sqrt(Exact_u1**2 + Exact_v1**2)

Exact_u2= np.real(Exact2)
Exact_v2 = np.imag(Exact2)
Exact_h2 = np.sqrt(Exact_u2**2 + Exact_v2**2)
#设置初始、边界和配置点数
N0 = 100
N_b = 100
N_f = 20000
#create random indices:
idx_x = np.random.choice(x.shape[0], N0, replace=False)
idx_t = np.random.choice(t.shape[0], N_b, replace=False)

#create initial data:
x0 = x[idx_x, :]
t0 = x0*0.0 - 1
u1 = Exact_u1[idx_x, 0:1]
v1 = Exact_v1[idx_x, 0:1]
u2 = Exact_u2[idx_x, 0:1]
v2 = Exact_v2[idx_x, 0:1]

tb = t[idx_t,:]
u1_lb = Exact_u1[0:1, idx_t]  # 1*100左边界随机去点对应的h实部 # 因为t=t0，所以去取第一列
v1_lb = Exact_v1[0:1, idx_t]
u1_ub = Exact_u1[255:256, idx_t]
v1_ub = Exact_v1[255:256, idx_t]
u2_lb = Exact_u2[0:1, idx_t]  # 1*100左边界随机去点对应的h实部 # 因为t=t0，所以去取第一列
v2_lb = Exact_v2[0:1, idx_t]
u2_ub = Exact_u2[255:256, idx_t]
v2_ub = Exact_v2[255:256, idx_t]

#create colocation points:
X_f = lb + (ub - lb) * lhs(2, N_f)

#convert data to tensors:
x0_t0 = tf.convert_to_tensor(np.concatenate((x0, t0), 1), dtype=tf.float32)
xlb_tlb = tf.convert_to_tensor(np.concatenate((0.0*tb + lb[0], tb), 1), dtype=tf.float32)
xub_tlb = tf.convert_to_tensor(np.concatenate((0.0*tb + ub[0], tb), 1), dtype=tf.float32)
xf_tf = tf.convert_to_tensor(X_f, dtype=tf.float32)
u1_tf = tf.convert_to_tensor(u1, dtype=tf.float32)
v1_tf = tf.convert_to_tensor(v1, dtype=tf.float32)
u2_tf = tf.convert_to_tensor(u2, dtype=tf.float32)
v2_tf = tf.convert_to_tensor(v2, dtype=tf.float32)

u1_lb_tf = tf.convert_to_tensor(u1_lb, dtype=tf.float32)
u1_ub_tf = tf.convert_to_tensor(u1_ub, dtype=tf.float32)
v1_lb_tf = tf.convert_to_tensor(v1_lb, dtype=tf.float32)
v1_ub_tf = tf.convert_to_tensor(v1_ub, dtype=tf.float32)
u2_lb_tf = tf.convert_to_tensor(u2_lb, dtype=tf.float32)
u2_ub_tf = tf.convert_to_tensor(u2_ub, dtype=tf.float32)
v2_lb_tf = tf.convert_to_tensor(v2_lb, dtype=tf.float32)
v2_ub_tf = tf.convert_to_tensor(v2_ub, dtype=tf.float32)

#create test data:
#创建测试数据：
X, T = np.meshgrid(x, t, sparse=False)
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u1_star = Exact_u1.T.flatten()[:,None]
v1_star = Exact_v1.T.flatten()[:,None]
h1_star = Exact_h1.T.flatten()[:,None]
u2_star = Exact_u2.T.flatten()[:,None]
v2_star = Exact_v2.T.flatten()[:,None]
h2_star = Exact_h2.T.flatten()[:,None]

# convert to tensor
X_star_tf = tf.convert_to_tensor(X_star, dtype=tf.float32)
#
# H1 = tf.reshape(Exact_h,[20000,1])
# U1 = tf.reshape(Exact_u,[20000,1])
# V1 = tf.reshape(Exact_v,[20000,1])
# idx = np.argsort(X_f, axis=0)  # 根据 H 中的元素选出对应的行数从 xt 中选出并存入 xt2 中
# idx = idx[:, 0]
# U1_ = []
# V1_ = []
# x_t = []
# for i in range(len(idx)):
#     U1__ = U1[idx[i], :]
#     V1__ = V1[idx[i], :]
#     U1_.append(U1__)
#     V1_.append(V1__)
#     x_t1 = X_star_tf[idx[i], :]
#     x_t.append(x_t1)
# U1_ = np.array(U1_)
# V1_ = np.array(V1_)
# U1_ = U1_.astype(np.float32)
# V1_ = V1_.astype(np.float32)
# UV = np.hstack((U1_ , V1_))
# x_t = np.array(x_t)
# x_t = tf.convert_to_tensor(x_t, dtype=tf.float32)

#create model instance:
g = Generator(lb = lb, ub = ub)

lr = 0.001  # 学习率的设置
epochs = 10000 # 训练的次数

#training of model:
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

# TensorFlow 中用于创建用于写入事件和摘要的文件写入器的函数。它允许您将摘要数据写入磁盘，以便您可以在 TensorBoard 中可视化和分析这些数据。通过 创建的写入器对象可以与等函数一起使用，用于记录模型的训练和评估过程中的相关摘要数据
summary_writer = tf.summary.create_file_writer(r".\log")

# simple = tf.random.normal([200000,2],stddev=1,mean=1) #模拟的是坐标
# lable = tf.random.normal([200000],stddev=1,mean=1) #模拟的是真实值
# lable = tf.expand_dims(lable,axis=1) #真实值进行维度扩展

simple = x0_t0   # 初始坐标
lable = np.concatenate((u1, v1, u2, v2), 1)#初始坐标对应的u v

########################################################################################################################
# U1 = tf.reshape(Exact_u1,[20000,1])
# V1 = tf.reshape(Exact_v1,[20000,1])
# U2 = tf.reshape(Exact_u2,[20000,1])
# V2 = tf.reshape(Exact_v2,[20000,1])
# # # 随机抽取1000个元素
# # new_arr1 = np.random.choice(U1, size=1000, replace=False)
# # new_arr2 = np.random.choice(V1, size=1000, replace=False)
# # new_arr3 = np.random.choice(U2, size=1000, replace=False)
# # new_arr4 = np.random.choice(V2, size=1000, replace=False)
# # U1 = tf.reshape(new_arr1,[1000,1])
# # V1 = tf.reshape(new_arr2,[1000,1])
# # U2 = tf.reshape(new_arr3,[1000,1])
# # V2 = tf.reshape(new_arr4,[1000,1])
# idx = np.argsort(X_f, axis=0)  # 根据 H 中的元素选出对应的行数从 xt 中选出并存入 xt2 中
# idx = idx[:, 0]
# U1_ = []
# V1_ = []
# U2_ = []
# V2_ = []
# x_t = []
# for i in range(len(idx)):
#     U1__ = U1[idx[i], :]
#     V1__ = V1[idx[i], :]
#     U1_.append(U1__)
#     V1_.append(V1__)
#     x_t1 = X_star_tf[idx[i], :]
#     U2__ = U2[idx[i], :]
#     V2__ = V2[idx[i], :]
#     U2_.append(U2__)
#     V2_.append(V2__)
#     x_t.append(x_t1)
# U1_ = np.array(U1_)
# V1_ = np.array(V1_)
# U1_ = U1_.astype(np.float32)
# V1_ = V1_.astype(np.float32)
# U2_ = np.array(U2_)
# V2_ = np.array(V2_)
# U2_ = U2_.astype(np.float32)
# V2_ = V2_.astype(np.float32)
# # UV = np.hstack((U1_ , V1_))
# x_t = np.array(x_t)
# x_t = tf.convert_to_tensor(x_t, dtype=tf.float32)

exact1 = data['q1']
exact2 = data['q2']
exact_u1 = np.real(exact1)
exact_v1 = np.imag(exact1)
exact_h1 = np.sqrt(exact_u1**2 + exact_v1**2)

exact_u2= np.real(exact2)
exact_v2 = np.imag(exact2)
exact_h2 = np.sqrt(exact_u2**2 + exact_v2**2)
H1 = tf.reshape(exact_h1,[51456,1])
H2 = tf.reshape(exact_h2,[51456,1])
U1 = tf.reshape(exact_u1,[51456,1])
V1 = tf.reshape(exact_v1,[51456,1])
U2 = tf.reshape(exact_u2,[51456,1])
V2 = tf.reshape(exact_v2,[51456,1])
xiaoyb = lb + (ub - lb) * lhs(2, 10000)
idx = np.argsort(xiaoyb, axis=0)  # 根据 H 中的元素选出对应的行数从 xt 中选出并存入 xt2 中
idx = idx[:, 0]
U1_ = []
U2_ = []
V1_ = []
V2_ = []
x_t = []
for i in range(len(idx)):
    U1__ = U1[idx[i], :]
    U2__ = U2[idx[i], :]
    V1__ = V1[idx[i], :]
    V2__ = V2[idx[i], :]
    U1_.append(U1__)
    U2_.append(U2__)
    V1_.append(V1__)
    V2_.append(V2__)
    x_t1 = X_star_tf[idx[i], :]
    x_t.append(x_t1)
U1_ = np.array(U1_)
U2_ = np.array(U2_)
V1_ = np.array(V1_)
V2_ = np.array(V2_)
U1_ = U1_.astype(np.float32)
U2_ = U2_.astype(np.float32)
V1_ = V1_.astype(np.float32)
V2_ = V2_.astype(np.float32)
x_t = np.array(x_t)
x_t = tf.convert_to_tensor(x_t, dtype=tf.float32)
#########################################################################################################################
# u1_1 = Exact_u1[58:59, idx_t]
# v1_1 = Exact_v1[58:59, idx_t]
# u2_1 = Exact_u2[58:59, idx_t]
# v2_1 = Exact_v2[58:59, idx_t]
#
# u1_2 = Exact_u1[137:138, idx_t]
# v1_2 = Exact_v1[137:138, idx_t]
# u2_2 = Exact_u2[137:138, idx_t]
# v2_2 = Exact_v2[137:138, idx_t]
# u1_1_tf = tf.convert_to_tensor(u1_1, dtype=tf.float32)
# v1_1_tf = tf.convert_to_tensor(v1_1, dtype=tf.float32)
# u2_1_tf = tf.convert_to_tensor(u2_1, dtype=tf.float32)
# v2_1_tf = tf.convert_to_tensor(v2_1, dtype=tf.float32)
#
# u1_2_tf = tf.convert_to_tensor(u1_2, dtype=tf.float32)
# v1_2_tf = tf.convert_to_tensor(v1_2, dtype=tf.float32)
# u2_2_tf = tf.convert_to_tensor(u2_2, dtype=tf.float32)
# v2_2_tf = tf.convert_to_tensor(v2_2, dtype=tf.float32)
# xxxx1 = np.array([-0.427135678391961])
# xxxx2 = np.array([19.4221105527638])
#
# xian1 = tf.convert_to_tensor(np.concatenate((0.0*tb + xxxx1[0], tb), 1), dtype=tf.float32)
# xian2 = tf.convert_to_tensor(np.concatenate((0.0*tb + xxxx2[0], tb), 1), dtype=tf.float32)
#######################################################################################################################
simple1 = x_t   # 坐标
lable1 = np.concatenate((U1_, V1_, U2_, V2_), 1)#坐标对应的u v
xy_train = data_chul(simple1, lable1)
# g = g_resnet(N=256,L=8)
d = Discriminator(N=100, L=7)

# lr_schedule_d = CosineAnnealingSchedule(lr_max=0.001, lr_min=0.00001, T=epochs) #使用余弦退火，按照要求使用
# lr_schedule_g = CosineAnnealingSchedule(lr_max=0.001, lr_min=0.00001, T=epochs)

# tensoboard 文件保存点
checkpoint_dir = './ckpt'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


#设置权重保存点
best_weights_checkpoint_path_d = os.path.join(checkpoint_dir, 'best_weights_d.h5')
best_weights_checkpoint_path_g = os.path.join(checkpoint_dir, 'best_weights_g.h5')

best_weights_checkpoint_g = ModelCheckpoint(best_weights_checkpoint_path_g,
                                              monitor='loss',
                                              verbose=1,
                                              save_best_only=True,
                                              save_weights_only=True,
                                              mode='min')
best_weights_checkpoint_d = ModelCheckpoint(best_weights_checkpoint_path_d,
                                                monitor='loss',
                                                verbose=1,
                                                save_best_only=True,
                                                save_weights_only=True,
                                                mode='min')


first_start_time = time.time()#当前的时间戳,算总时间
start_time = first_start_time#当前的时间戳，算每次时间
History = []#存loss在列表History中用于画图
# for epoch in range(epochs):
#     for step, (xx, yy) in enumerate(xy_train):
#         with tf.GradientTape() as tape:
#             d_loss, gp = d_loss_fn(generator=g, discriminator=d, x=xx, y=yy)
#         grads = tape.gradient(d_loss, d.trainable_variables)
#         tf.optimizers.Adam().apply_gradients(zip(grads, d.trainable_variables))#learning_rate=lr1, beta_1=0.5
#         # 下面是使用余弦退火算法
#         # tf.optimizers.Adam(learning_rate=lr_schedule_d(epochs), beta_1=0.5).apply_gradients(zip(grads, d.trainable_variables))
#
#         with tf.GradientTape() as tape:
#             loss_pinn = g.loss_fn(x0_t0, xlb_tlb, xub_tlb, xf_tf, u1_tf, v1_tf, u2_tf, v2_tf, u1_lb_tf, u1_ub_tf, v1_lb_tf, v1_ub_tf, u2_lb_tf, u2_ub_tf, v2_lb_tf, v2_ub_tf)
#             g_loss, fake_label = g_loss_fn(generator=g, discriminator=d, x=xx, y=yy)
#             g_Loss = 1 * loss_pinn + 1 * g_loss
#         dW = tape.gradient(g_Loss, g.trainable_variables)
#         optimizer.apply_gradients(zip(dW, g.trainable_variables))
#         # 下面是使用余弦退火算法
#         # tf.optimizers.Adam(learning_rate=lr_schedule_g(epochs), beta_1=0.9).apply_gradients(zip(grads, g.trainable_variables))
#         with summary_writer.as_default():
#             tf.summary.scalar('d_loss', float(d_loss), step=epoch)
#             tf.summary.scalar('g_Loss', float(g_Loss), step=epoch)
#             # tf.summary.scalar('real', float(fake_label), step=epoch)
#             # tf.summary.scalar('fake', float(y), step=epoch)
#         print(epoch, "d_loss:", float(d_loss), "g_loss", float(g_loss), "Pinn_loss", float(loss_pinn),"all_loss:", float(g_loss + d_loss), "gp:", float(gp))
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        d_loss, gp = d_loss_fn(generator=g, discriminator=d, x=simple1, y=lable1)
    grads = tape.gradient(d_loss, d.trainable_variables)
    tf.optimizers.Adam(learning_rate=lr, beta_1=0.5).apply_gradients(zip(grads, d.trainable_variables))  #
    # 下面是使用余弦退火算法
    # tf.optimizers.Adam(learning_rate=lr_schedule_d(epochs), beta_1=0.5).apply_gradients(zip(grads, d.trainable_variables))

    with tf.GradientTape() as tape:
        loss_pinn = g.loss_fn(x0_t0, xlb_tlb, xub_tlb, xf_tf, u1_tf, v1_tf, u2_tf, v2_tf, u1_lb_tf, u1_ub_tf, v1_lb_tf,
                              v1_ub_tf, u2_lb_tf, u2_ub_tf, v2_lb_tf, v2_ub_tf,
                              # u1_1_tf, v1_1_tf, u2_1_tf, v2_1_tf, u1_2_tf, v1_2_tf, u2_2_tf, v2_2_tf, xian1, xian2,
                              U1_, U2_, V1_, V2_, x_t)
        g_loss, fake_label = g_loss_fn(generator=g, discriminator=d, x=simple1, y=lable1)
        g_Loss = 50 * loss_pinn + 0.5 * g_loss
    dW = tape.gradient(g_Loss, g.trainable_variables)
    optimizer.apply_gradients(zip(dW, g.trainable_variables))
    # 下面是使用余弦退火算法
    # tf.optimizers.Adam(learning_rate=lr_schedule_g(epochs), beta_1=0.9).apply_gradients(zip(grads, g.trainable_variables))
    with summary_writer.as_default():
        tf.summary.scalar('d_loss', float(d_loss), step=epoch)
        tf.summary.scalar('g_Loss', float(g_Loss), step=epoch)
        # tf.summary.scalar('real', float(fake_label), step=epoch)
        # tf.summary.scalar('fake', float(y), step=epoch)
    print(epoch, "d_loss:", float(d_loss), "g_loss", float(g_loss), "Pinn_loss", float(loss_pinn), "all_loss:",
          float(g_loss + d_loss), "gp:", float(gp))
    if epoch % 10 == 0:
        # print(epoch, "d_loss:", float(d_loss), "g_loss", float(g_loss), "all_loss:", float(g_loss + d_loss),"gp:",float(gp))
        g.save_weights(best_weights_checkpoint_path_g)
        d.save_weights(best_weights_checkpoint_path_d)

        ###############################################################绘图
        ########存loss在列表History中用于画图
        aaaa = loss_pinn.numpy()  # current_loss由张量转换为np方便存在数组里
        History.append(aaaa)  #
        ######每次用时
        prev_time = start_time
        now = time.time()
        edur = datetime.fromtimestamp(now - prev_time) \
                   .strftime("%S.%f")[:-5]
        prev_time = now
        #####总时间
        Total = datetime.fromtimestamp(time.time() - first_start_time) \
            .strftime("%M:%S")
        #####打印
        print(f"epoch = {epoch} " +  # 训练次数
              f"elapsed = {Total} " +  # 总时间
              f"(+{edur}) "   # 每次时间 +
              )  # 损失函数lossf    "loss = {g_Loss:.4e} "
        start_time = time.time()  # 刷新时间用于下一轮计算每次用时



#calculate outputs:
#计算输出：
predictions = g(X_star_tf)
print(f" Training time: = {Total} ")  ##输出总的训练时间
u1_pred = predictions[:,0:1]
v1_pred = predictions[:,1:2]
h1_pred = tf.sqrt(u1_pred**2 + v1_pred**2)
u1_pred = u1_pred.numpy()
v1_pred = v1_pred.numpy()
h1_pred = h1_pred.numpy()

u2_pred = predictions[:,2:3]
v2_pred = predictions[:,3:4]
h2_pred = tf.sqrt(u2_pred**2 + v2_pred**2)
u2_pred = u2_pred.numpy()
v2_pred = v2_pred.numpy()
h2_pred = h2_pred.numpy()

#calculate errors:
error_u1 = np.linalg.norm(u1_star - u1_pred,2)/np.linalg.norm(u1_star,2)
error_v1 = np.linalg.norm(v1_star - v1_pred,2)/np.linalg.norm(v1_star,2)
error_h1 = np.linalg.norm(h1_star - h1_pred,2)/np.linalg.norm(h1_star,2)

error_u2 = np.linalg.norm(u2_star - u2_pred,2)/np.linalg.norm(u2_star,2)
error_v2 = np.linalg.norm(v2_star - v2_pred,2)/np.linalg.norm(v2_star,2)
error_h2 = np.linalg.norm(h2_star - h2_pred,2)/np.linalg.norm(h2_star,2)

print("u1 error: ", error_u1)
print("v1 error: ", error_v1)
print("h1 error: ", error_h1)

print("u2 error: ", error_u2)
print("v2 error: ", error_v2)
print("h2 error: ", error_h2)
#plot results for 0.75s and 1s:
# index = 75
# plt.plot(X_star[0:256,0], h_star[index*256:index*256+256], 'ro', alpha=0.2, label='0.75s actual')
# plt.plot(X_star[0:256,0], h_pred[index*256:index*256+256], 'k', label='0.75s pred.')
#
# index = 100
# plt.plot(X_star[0:256,0], h_star[index*256:index*256+256], 'bo', alpha=0.2, label='1s actual')
# plt.plot(X_star[0:256,0], h_pred[index*256:index*256+256], 'k--', label='1s pred.')
#
# plt.legend()
# plt.show()

U1_pred = griddata(X_star, u1_pred.flatten(), (X, T), method='cubic')
V1_pred = griddata(X_star, v1_pred.flatten(), (X, T), method='cubic')
H1_pred = griddata(X_star, h1_pred.flatten(), (X, T), method='cubic')

U2_pred = griddata(X_star, u2_pred.flatten(), (X, T), method='cubic')
V2_pred = griddata(X_star, v2_pred.flatten(), (X, T), method='cubic')
H2_pred = griddata(X_star, h2_pred.flatten(), (X, T), method='cubic')
# FU_pred = griddata(X_star, f_u_pred.flatten(), (X, T), method='cubic')
# FV_pred = griddata(X_star, f_v_pred.flatten(), (X, T), method='cubic')
import pandas as pd
test = pd.DataFrame(columns=['1'], data=History)
# 2.数据保存，index表示是否显示行名，sep数据分开符
test.to_csv('耦合1.csv', index=False, sep=',')

fig = plt.figure("loss", dpi=100, facecolor=None, edgecolor=None, frameon=True)
plt.plot(range(1, epochs + 1, 10), History, 'r-', linewidth=1, label='learning rate=0.0001')  # + 1前面的数就是迭代次数
plt.xlabel('$\#$ iterations')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend(loc='upper right')
plt.savefig('loss.png')

X0 = np.concatenate((x0, 0 * x0+ lb[1]), 1)  # (x0, 0)
X_lb = np.concatenate((0 * tb + lb[0], tb), 1)  # (lb[0], tb)
X_ub = np.concatenate((0 * tb + ub[0], tb), 1)  # (ub[0], tb)
X_u_train = np.vstack([X0, X_lb, X_ub])
xxx = X_f[:, 0]  # a为二维列表ls的第1列
ttt = X_f[:, 1]
plt.figure()
plt.scatter(xxx, ttt, c='red', s=1, label='legend')
plt.xticks(range(-20, 40, 10))
plt.yticks(range(0, 4, 1))
plt.xlabel("x", fontdict={'size': 16})
plt.ylabel("t", fontdict={'size': 16})
plt.title("X_f", fontdict={'size': 20})
plt.legend(loc='best')
plt.savefig('X_f.png')

fig = plt.figure(dpi=130)

ax = plt.subplot(3, 1, 1)
plt.plot(x, Exact_h1[:, 25], 'b-', linewidth=2, label='Exact')
plt.plot(x, H1_pred[25, :], 'r:', linewidth=4, label='Prediction')
ax.set_ylabel('$|Q1(t,x)|$')
ax.set_xlabel('$x$')
ax.set_title('$t = %.2f$' % (t[25]), fontsize=10)
plt.legend()

ax1 = plt.subplot(3, 1, 2)
plt.plot(x, Exact_h1[:, 50], 'b-', linewidth=2, label='Exact')
plt.plot(x, H1_pred[50, :], 'r:', linewidth=4, label='Prediction')
ax1.set_ylabel('$|Q1(t,x)|$')
ax1.set_xlabel('$x$')
ax1.set_title('$t = %.2f$' % (t[50]), fontsize=10)
plt.legend()

ax2 = plt.subplot(3, 1, 3)
plt.plot(x, Exact_h1[:, 75], 'b-', linewidth=2, label='Exact')
plt.plot(x, H1_pred[75, :], 'r:', linewidth=4, label='Prediction')
ax2.set_ylabel('$|Q1(t,x)|$')
ax2.set_xlabel('$x$')
ax2.set_title('$t = %.2f$' % (t[75]), fontsize=10)
plt.legend()
fig.tight_layout()
plt.savefig('Q1(t,x).png')

fig = plt.figure(dpi=130)

ax = plt.subplot(3, 1, 2)
plt.plot(x, Exact_h2[:, 50], 'b-', linewidth=2, label='Exact')
plt.plot(x, H2_pred[50, :], 'r:', linewidth=4, label='Prediction')
ax.set_ylabel('$|Q2(t,x)|$')
ax.set_xlabel('$x$')
ax.set_title('$t = %.2f$' % (t[50]), fontsize=10)
plt.legend()

ax1 = plt.subplot(3, 1, 1)
plt.plot(x, Exact_h2[:, 25], 'b-', linewidth=2, label='Exact')
plt.plot(x, H2_pred[25, :], 'r:', linewidth=4, label='Prediction')
ax1.set_ylabel('$|Q2(t,x)|$')
ax1.set_xlabel('$x$')
ax1.set_title('$t = %.2f$' % (t[25]), fontsize=10)
plt.legend()

ax2 = plt.subplot(3, 1, 3)
plt.plot(x, Exact_h2[:, 75], 'b-', linewidth=2, label='Exact')
plt.plot(x, H2_pred[75, :], 'r:', linewidth=4, label='Prediction')
ax2.set_ylabel('$|Q2(t,x)|$')
ax2.set_xlabel('$x$')
ax2.set_title('$t = %.2f$' % (t[75]), fontsize=10)
plt.legend()
fig.tight_layout()
plt.savefig('Q2(t,x).png')


fig = plt.figure('实际h1(t,x)', dpi=130)
ax = fig.add_subplot(1, 1, 1)
ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'rx', label='Data (%d points)' % (X_u_train.shape[0]), markersize=4,
        clip_on=False)
h = ax.imshow(Exact_h1, interpolation='nearest', cmap='YlGnBu',
              extent=[lb[1], ub[1], lb[0], ub[0]],
              origin='lower', aspect='auto')
plt.colorbar(h)
ax.set_ylabel('$x$')
ax.set_xlabel('$t$')
plt.title('Exact Dynamics1')
plt.savefig('实际h1(t,x).png')

fig = plt.figure('实际h2(t,x)', dpi=130)
ax = fig.add_subplot(1, 1, 1)
ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'rx', label='Data (%d points)' % (X_u_train.shape[0]), markersize=4,
        clip_on=False)
h = ax.imshow(Exact_h2, interpolation='nearest', cmap='YlGnBu',
              extent=[lb[1], ub[1], lb[0], ub[0]],
              origin='lower', aspect='auto')
plt.colorbar(h)
ax.set_ylabel('$x$')
ax.set_xlabel('$t$')
plt.title('Exact Dynamics2')
plt.savefig('实际h2(t,x).png')

fig = plt.figure("实际演化图1", dpi=130, facecolor=None, edgecolor=None, frameon=None)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, T, Exact_h1.T, cmap='coolwarm', rstride=1, cstride=1,
                       linewidth=0, antialiased=False)
# ax.grid(False)
ax.set_xlabel('$x$')
ax.set_ylabel('$t$')
ax.set_zlabel('$|Q1(t,x)|$');
plt.savefig('实际演化图1.png')

fig = plt.figure("预测演化图1", dpi=130, facecolor=None, edgecolor=None, frameon=None)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, T, H1_pred, cmap='coolwarm', rstride=1, cstride=1,
                       linewidth=0, antialiased=False)
# ax.grid(False)
ax.set_xlabel('$x$')
ax.set_ylabel('$t$')
ax.set_zlabel('$|Q1(t,x)|$');
plt.savefig('预测演化图1.png')

fig = plt.figure("实际演化图2", dpi=130, facecolor=None, edgecolor=None, frameon=None)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, T, Exact_h2.T, cmap='coolwarm', rstride=1, cstride=1,
                       linewidth=0, antialiased=False)
# ax.grid(False)
ax.set_xlabel('$x$')
ax.set_ylabel('$t$')
ax.set_zlabel('$|Q2(t,x)|$');
plt.savefig('实际演化图2.png')

fig = plt.figure("预测演化图2", dpi=130, facecolor=None, edgecolor=None, frameon=None)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, T, H2_pred, cmap='coolwarm', rstride=1, cstride=1,
                       linewidth=0, antialiased=False)
# ax.grid(False)
ax.set_xlabel('$x$')
ax.set_ylabel('$t$')
ax.set_zlabel('$|Q2(t,x)|$');
plt.savefig('预测演化图2.png')

plt.show()
