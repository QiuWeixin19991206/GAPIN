import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras import Model

#create model class
class Generator(Model):
    def __init__(self, lb, ub):
        super(Generator, self).__init__()
        
        #create layers: [2 100 100 100 100 2]
        self.fc1 = Dense(100, input_shape=(None, 2), activation='tanh', kernel_initializer='glorot_uniform')  # 初始
        # self.bn1 = layers.BatchNormalization(axis=-1,center=True,scale=True,trainable=True)

        self.fc2 = Dense(100, activation='tanh', kernel_initializer='glorot_uniform')  # 中间隐藏层
        # self.bn2 = layers.BatchNormalization(axis=-1,center=True,scale=True,trainable=True)

        self.fc3 = Dense(100, activation='tanh', kernel_initializer='glorot_uniform')
        # self.bn3 = layers.BatchNormalization(axis=-1,center=True,scale=True,trainable=True)

        self.fc4 = Dense(100, activation='tanh', kernel_initializer='glorot_uniform')
        # self.bn4 = layers.BatchNormalization(axis=-1,center=True,scale=True,trainable=True)

        self.fc5 = Dense(100, activation='tanh', kernel_initializer='glorot_uniform')
        # self.bn5 = layers.BatchNormalization(axis=-1,center=True,scale=True,trainable=True)

        self.fc6 = Dense(100, activation='tanh', kernel_initializer='glorot_uniform')
        # self.bn6 = layers.BatchNormalization(axis=-1,center=True,scale=True,trainable=True)

        self.fc7 = Dense(100, activation = 'tanh', kernel_initializer = 'glorot_uniform')
        self.fc8 = Dense(100, activation = 'tanh', kernel_initializer = 'glorot_uniform')
        self.fc9 = Dense(100, activation = 'tanh', kernel_initializer = 'glorot_uniform')
        self.fc10 = Dense(100, activation = 'tanh', kernel_initializer = 'glorot_uniform')
        self.fc11 = Dense(100, activation = 'tanh', kernel_initializer = 'glorot_uniform')
        self.fc12 = Dense(100, activation = 'tanh', kernel_initializer = 'glorot_uniform')
        self.fc13 = Dense(100, activation = 'tanh', kernel_initializer = 'glorot_uniform')
        self.fc14 = Dense(100, activation = 'tanh', kernel_initializer = 'glorot_uniform')
        self.fc15 = Dense(100, activation = 'tanh', kernel_initializer = 'glorot_uniform')
        self.fc16 = Dense(100, activation = 'tanh', kernel_initializer = 'glorot_uniform')
        self.fc17 = Dense(100, activation = 'tanh', kernel_initializer = 'glorot_uniform')
        self.fc18 = Dense(100, activation = 'tanh', kernel_initializer = 'glorot_uniform')
        self.output_layer = Dense(4, activation='linear', kernel_initializer='glorot_uniform')  # 输出层

        # self.fcs1 = Dense(100, input_shape=(None, 2), activation='tanh', kernel_initializer='glorot_uniform')  # 初始
        # # self.bn1 = layers.BatchNormalization(axis=-1,center=True,scale=True,trainable=True)
        #
        # self.fcs2 = Dense(100, activation='tanh', kernel_initializer='glorot_uniform')  # 中间隐藏层
        # # self.bn2 = layers.BatchNormalization(axis=-1,center=True,scale=True,trainable=True)
        #
        # self.fcs3 = Dense(100, activation='tanh', kernel_initializer='glorot_uniform')
        # # self.bn3 = layers.BatchNormalization(axis=-1,center=True,scale=True,trainable=True)
        #
        # self.fcs4 = Dense(100, activation='tanh', kernel_initializer='glorot_uniform')
        # # self.bn4 = layers.BatchNormalization(axis=-1,center=True,scale=True,trainable=True)
        #
        # self.fcs5 = Dense(100, activation='tanh', kernel_initializer='glorot_uniform')
        # # self.bn5 = layers.BatchNormalization(axis=-1,center=True,scale=True,trainable=True)
        #
        # self.fcs6 = Dense(100, activation='tanh', kernel_initializer='glorot_uniform')
        # # self.bn6 = layers.BatchNormalization(axis=-1,center=True,scale=True,trainable=True)
        #
        # self.fcs7 = Dense(100, activation='tanh', kernel_initializer='glorot_uniform')
        # self.output_layer = Dense(2, activation='linear', kernel_initializer='glorot_uniform')  # 输出层

        self.lb = lb
        self.ub = ub

    # call method
    def call(self, x):
        return self.net(x)

    # 返回 u，v 和导数的类方法
    def net_uv1(self, x):
        with tf.GradientTape(persistent=True) as tape_1:
            tape_1.watch(x)
            net_output = self.net(x)
            u1 = net_output[:, 0:1]
            v1 = net_output[:, 1:2]

        u1_x = tape_1.gradient(u1, x)[:, 0:1]
        v1_x = tape_1.gradient(v1, x)[:, 0:1]

        del tape_1
        return u1, v1, u1_x, v1_x

    def net_uv2(self, x):
        with tf.GradientTape(persistent=True) as tape_1:
            tape_1.watch(x)
            net_output = self.net(x)
            u2 = net_output[:, 2:3]
            v2 = net_output[:, 3:4]

        u2_x = tape_1.gradient(u2, x)[:, 0:1]
        v2_x = tape_1.gradient(v2, x)[:, 0:1]

        del tape_1
        return u2, v2, u2_x, v2_x
    def net_f_uv1(self, x):
        with tf.GradientTape(persistent=True) as tape_5:
            tape_5.watch(x)
            output1 = self.net(x)
            u1 = output1[:, 0:1]
            v1 = output1[:, 1:2]
            u2 = output1[:, 2:3]
            v2 = output1[:, 3:4]
        del tape_5
        return u1, v1, u2, v2
    # 返回偏微分方程残差的类方法
    def net_f_uv(self, x):
        with tf.GradientTape(persistent=True) as tape_2:
            tape_2.watch(x)
            with tf.GradientTape(persistent=True) as tape_3:
                tape_3.watch(x)
                output = self.net(x)
                u1 = output[:, 0:1]
                v1 = output[:, 1:2]

                u2 = output[:, 2:3]
                v2 = output[:, 3:4]

            u1_x = tape_3.gradient(u1, x)[:, 0:1]
            v1_x = tape_3.gradient(v1, x)[:, 0:1]
            u1_t = tape_3.gradient(u1, x)[:, 1:2]
            v1_t = tape_3.gradient(v1, x)[:, 1:2]

            u2_x = tape_3.gradient(u2, x)[:, 0:1]
            v2_x = tape_3.gradient(v2, x)[:, 0:1]
            u2_t = tape_3.gradient(u2, x)[:, 1:2]
            v2_t = tape_3.gradient(v2, x)[:, 1:2]

        u1_xx = tape_2.gradient(u1_x, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)[:, 0:1]
        v1_xx = tape_2.gradient(v1_x, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)[:, 0:1]

        u2_xx = tape_2.gradient(u2_x, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)[:, 0:1]
        v2_xx = tape_2.gradient(v2_x, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)[:, 0:1]

        gamma = 1
        mu = 1
        f_v1 = u1_t + v1_xx + (mu) * v1 * (u1 ** 2 + v1 ** 2 + u2 ** 2 + v2 ** 2) + gamma * (
                u1_x * (u1 ** 2 + v1 ** 2 + u2 ** 2 + v2 ** 2) + u1 * 2 * (
                u1 * u1_x + u2 * u2_x + v1 * v1_x + v2 * v2_x))
        f_v2 = u2_t + v2_xx + (mu) * v2 * (u1 ** 2 + v1 ** 2 + u2 ** 2 + v2 ** 2) + gamma * (
                u2_x * (u1 ** 2 + v1 ** 2 + u2 ** 2 + v2 ** 2) + u2 * 2 * (
                u1 * u1_x + u2 * u2_x + v1 * v1_x + v2 * v2_x))
        f_u1 = v1_t - u1_xx - (mu) * u1 * (u1 ** 2 + v1 ** 2 + u2 ** 2 + v2 ** 2) + gamma * (
                v1_x * (u1 ** 2 + v1 ** 2 + u2 ** 2 + v2 ** 2) + v1 * 2 * (
                u1 * u1_x + u2 * u2_x + v1 * v1_x + v2 * v2_x))  # 实部
        f_u2 = v2_t - u2_xx - (mu) * u2 * (u1 ** 2 + v1 ** 2 + u2 ** 2 + v2 ** 2) + gamma * (
                v2_x * (u1 ** 2 + v1 ** 2 + u2 ** 2 + v2 ** 2) + v2 * 2 * (
                u1 * u1_x + u2 * u2_x + v1 * v1_x + v2 * v2_x))  # 实部

        del tape_2, tape_3
        return f_u1, f_v1, f_u2, f_v2

    # 网络前向传播方法
    def net(self, x):
        x1 = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        x1= self.fc1(x1)
        x1 = self.fc2(x1)
        x1 = self.fc3(x1)
        x1 = self.fc4(x1)
        x1 = self.fc5(x1)
        x1 = self.fc6(x1)
        x1 = self.fc7(x1)
        # x2 = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        # x2 = self.fcs1(x2)
        # # x = self.bn1(x)
        # x2 = self.fcs2(x2)
        # # x = self.bn2(x)
        # x2 = self.fcs3(x2)
        # # residual_block2 = x2+ residual_block2
        # # x = self.bn3(x)
        # x2 = self.fcs4(x2)
        # # x = self.bn4(x)
        # x2 = self.fcs5(x2)
        # x = self.bn5(x)
        # x2 = self.fcs6(x2)
        # x = self.bn6(x)

        return self.output_layer(x1)

    # def loss_fn(self, x0_t0, xlb_tlb, xub_tub, xf_tf, u1, v1, u2, v2,
    #             u1_lb_tf, u1_ub_tf, v1_lb_tf, v1_ub_tf, u2_lb_tf, u2_ub_tf, v2_lb_tf, v2_ub_tf,
    #             u1_1_tf, v1_1_tf, u2_1_tf,v2_1_tf, xian1,
    #             u1_2_tf, v1_2_tf, u2_2_tf, v2_2_tf,xian2):
    def loss_fn(self, x0_t0, xlb_tlb, xub_tub, xf_tf, u1_tf, v1_tf, u2_tf, v2_tf, u1_lb_tf, u1_ub_tf, v1_lb_tf, v1_ub_tf, u2_lb_tf, u2_ub_tf, v2_lb_tf, v2_ub_tf,
        #u1_1_tf, v1_1_tf, u2_1_tf, v2_1_tf, u1_2_tf, v1_2_tf, u2_2_tf, v2_2_tf, xian1, xian2,
                U1_, U2_, V1_, V2_, x_t):
        #         ,u1_3_tf, v1_3_tf, u2_3_tf, v2_3_tf,u1_4_tf, v1_4_tf, u2_4_tf, v2_4_tf, , xian3, xian4):
        # u1_1_pred, v1_1_pred, _, _ = self.net_uv1(xian1)
        # u1_2_pred, v1_2_pred, _, _ = self.net_uv1(xian2)
        # # u1_3_pred, v1_3_pred, _, _ = self.net_uv1(xian3)
        # # u1_4_pred, v1_4_pred, _, _ = self.net_uv1(xian4)
        # u2_1_pred, v2_1_pred, _, _ = self.net_uv2(xian1)
        # u2_2_pred, v2_2_pred, _, _ = self.net_uv2(xian2)
        # # u2_3_pred, v2_3_pred, _, _ = self.net_uv2(xian3)
        # # u2_4_pred, v2_4_pred, _, _ = self.net_uv2(xian4)
        u1_pred, v1_pred, _, _ = self.net_uv1(x0_t0)
        u2_pred, v2_pred, _, _ = self.net_uv2(x0_t0)
        u1_lb_pred, v1_lb_pred, u1_x_lb_pred, v1_x_lb_pred = self.net_uv1(xlb_tlb)
        u2_lb_pred, v2_lb_pred, u2_x_lb_pred, v2_x_lb_pred = self.net_uv2(xlb_tlb)
        u1_ub_pred, v1_ub_pred, u1_x_ub_pred, v1_x_ub_pred = self.net_uv1(xub_tub)
        u2_ub_pred, v2_ub_pred, u2_x_ub_pred, v2_x_ub_pred = self.net_uv2(xub_tub)
        f_u1_pred, f_v1_pred, f_u2_pred, f_v2_pred= self.net_f_uv(xf_tf)

        loss_point2 = 0
        # loss = tf.reduce_mean(tf.square(u2_lb_tf - u2_lb_pred)) +\
        #        tf.reduce_mean(tf.square(v2_lb_tf - v2_lb_pred)) +\
        #        tf.reduce_mean(tf.square(u2_ub_tf - u2_ub_pred)) +\
        #        tf.reduce_mean(tf.square(v2_ub_tf - v2_ub_pred)) +\
        #        tf.reduce_mean(tf.square(u1_lb_tf - u1_lb_pred)) +\
        #        tf.reduce_mean(tf.square(v1_lb_tf - v1_lb_pred)) +\
        #        tf.reduce_mean(tf.square(u1_ub_tf - u1_ub_pred)) +\
        #        tf.reduce_mean(tf.square(v1_ub_tf - v1_ub_pred)) + \
        #        tf.reduce_mean(tf.square(u1_pred - u1_tf)) + \
        #        tf.reduce_mean(tf.square(v1_pred - v1_tf)) + \
        #        tf.reduce_mean(tf.square(u2_pred - u2_tf)) + \
        #        tf.reduce_mean(tf.square(v2_pred - v2_tf)) + \
        #        tf.reduce_mean(tf.square(f_u1_pred)) + \
        #        tf.reduce_mean(tf.square(f_v1_pred)) + \
        #        tf.reduce_mean(tf.square(f_u2_pred)) + \
        #        tf.reduce_mean(tf.square(f_v2_pred))
        # loss = loss_point2 + \
        #        20*tf.reduce_mean(tf.square(u1_pred - u1_tf)) + \
        #        20*tf.reduce_mean(tf.square(v1_pred - v1_tf)) + \
        #        20*tf.reduce_mean(tf.square(u2_pred - u2_tf)) + \
        #        20*tf.reduce_mean(tf.square(v2_pred - v2_tf)) + \
        #        20*tf.reduce_mean(tf.square(u1_1_pred - u1_1_tf)) + \
        #        20*tf.reduce_mean(tf.square(v1_1_pred - v1_1_tf)) + \
        #        20*tf.reduce_mean(tf.square(u2_1_pred - u2_1_tf)) + \
        #        20*tf.reduce_mean(tf.square(v2_1_pred - v2_1_tf)) + \
        #        20*tf.reduce_mean(tf.square(u1_2_pred - u1_2_tf)) + \
        #        20*tf.reduce_mean(tf.square(v1_2_pred - v1_2_tf)) + \
        #        20*tf.reduce_mean(tf.square(u2_2_pred - u2_2_tf)) + \
        #        20*tf.reduce_mean(tf.square(v2_2_pred - v2_2_tf)) + \
        #        tf.reduce_mean(tf.square(u2_lb_tf - u2_lb_pred)) +\
        #        tf.reduce_mean(tf.square(v2_lb_tf - v2_lb_pred)) +\
        #        tf.reduce_mean(tf.square(u2_ub_tf - u2_ub_pred)) +\
        #        tf.reduce_mean(tf.square(v2_ub_tf - v2_ub_pred)) +\
        #        tf.reduce_mean(tf.square(u1_lb_tf - u1_lb_pred)) +\
        #        tf.reduce_mean(tf.square(v1_lb_tf - v1_lb_pred)) +\
        #        tf.reduce_mean(tf.square(u1_ub_tf - u1_ub_pred)) +\
        #        tf.reduce_mean(tf.square(v1_ub_tf - v1_ub_pred)) +\
        #        tf.reduce_mean(tf.square(f_u1_pred)) + \
        #        tf.reduce_mean(tf.square(f_v1_pred)) + \
        #        tf.reduce_mean(tf.square(f_u2_pred)) + \
        #        tf.reduce_mean(tf.square(f_v2_pred))
        u1_, v1_, u2_, v2_ = self.net_f_uv1(x_t)
        # LOSS = 1+1
        loss = 1*(tf.reduce_mean(tf.square(u1_ - U1_)) + \
               tf.reduce_mean(tf.square(u2_ - U2_)) + \
               tf.reduce_mean(tf.square(v1_ - V1_)) + \
               tf.reduce_mean(tf.square(v2_ - V2_)))+ \
               0.5* (tf.reduce_mean(tf.square(u2_lb_tf - u2_lb_pred)) + \
               tf.reduce_mean(tf.square(v2_lb_tf - v2_lb_pred)) +\
               tf.reduce_mean(tf.square(u2_ub_tf - u2_ub_pred)) +\
               tf.reduce_mean(tf.square(v2_ub_tf - v2_ub_pred)) +\
               tf.reduce_mean(tf.square(u1_lb_tf - u1_lb_pred)) +\
               tf.reduce_mean(tf.square(v1_lb_tf - v1_lb_pred)) +\
               tf.reduce_mean(tf.square(u1_ub_tf - u1_ub_pred)) +\
               tf.reduce_mean(tf.square(v1_ub_tf - v1_ub_pred)) + \
               tf.reduce_mean(tf.square(u1_pred - u1_tf)) + \
               tf.reduce_mean(tf.square(v1_pred - v1_tf)) + \
               tf.reduce_mean(tf.square(u2_pred - u2_tf)) + \
               tf.reduce_mean(tf.square(v2_pred - v2_tf)) + \
               tf.reduce_mean(tf.square(f_u1_pred)) + \
               tf.reduce_mean(tf.square(f_v1_pred)) + \
               tf.reduce_mean(tf.square(f_u2_pred)) + \
               tf.reduce_mean(tf.square(f_v2_pred)))
        return loss