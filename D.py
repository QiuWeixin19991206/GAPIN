import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model,Sequential
# 生成器和鉴别器的网络模型的深度和宽度不应该相差很大
class Discriminator(Model): # This is Discriminator
    def __init__(self,N,L):
        super(Discriminator, self).__init__()
        self.backbone = self.Base_block(N=N,L=L)
        self.out_layer = Dense(1 )#,activation=tf.sigmoid

    def call(self, inputs, training=None, mask=None):
        x = self.backbone(inputs)
        out = self.out_layer(x)
        return out

    def Base_block(self,N,L):
        block = []
        for _ in range(L):
            block.append(Dense(N,activation=tf.nn.leaky_relu))
        return Sequential(block)
# if __name__ == "__main__":
#     x = tf.random.normal([200,1],stddev=1,mean=0)
#     model = Discriminator(N=128,L=8)
#     out = model(x)
#     model.summary()
#     print(out.numpy().shape)





