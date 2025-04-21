
from tensorflow.keras.layers import Layer
import tensorflow as tf

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        latent_dim = input_shape[0][-1]
        self.W_a = self.add_weight(name='W_a', shape=(latent_dim, latent_dim), initializer='random_normal', trainable=True)
        self.U_a = self.add_weight(name='U_a', shape=(latent_dim, latent_dim), initializer='random_normal', trainable=True)
        self.V_a = self.add_weight(name='V_a', shape=(latent_dim, 1), initializer='random_normal', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        encoder_output, decoder_output = inputs
        decoder_steps = tf.shape(decoder_output)[1]
        encoder_steps = tf.shape(encoder_output)[1]

        dec_exp = tf.expand_dims(decoder_output, 2)
        dec_rep = tf.tile(dec_exp, [1, 1, encoder_steps, 1])

        enc_exp = tf.expand_dims(encoder_output, 1)
        enc_rep = tf.tile(enc_exp, [1, decoder_steps, 1, 1])

        W_enc = tf.tensordot(enc_rep, self.W_a, axes=[[3], [0]])
        U_dec = tf.tensordot(dec_rep, self.U_a, axes=[[3], [0]])

        e = tf.nn.tanh(W_enc + U_dec)
        score = tf.tensordot(e, self.V_a, axes=[[3], [0]])
        score = tf.squeeze(score, axis=-1)

        attention_weights = tf.nn.softmax(score, axis=-1)
        context_vector = tf.matmul(attention_weights, encoder_output)
        return context_vector, attention_weights
