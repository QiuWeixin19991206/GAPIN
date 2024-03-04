import D
import tensorflow as tf
#使用earth movers distance 距离公式作为惩罚项
def Wpena(discriminator, real, fake):
    t = tf.random.uniform(real.shape, minval=0, maxval=1)
    interplate =  tf.multiply(t, real) +  (1. - t) * fake
    with tf.GradientTape() as tape:
        tape.watch([interplate])
        d_interplate_logits = discriminator(interplate)
    grads = tape.gradient(d_interplate_logits, interplate)
    # grads[b,h,w,c]
    grads = tf.reshape(grads, [grads.shape[0], -1])  # 来进行一个打平的操作
    gp = tf.norm(grads, axis=1)
    gp = tf.reduce_mean((gp - 1) ** 2)
    return gp
