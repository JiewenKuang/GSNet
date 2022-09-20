import tensorflow as tf
import tensorflow_graphics as tfg


class MaxPool:
    def __call__(self, X, indices):
        _X = []
        for i in range(1, len(indices)):
            s, e = indices[i - 1], indices[i]
            _X += [tf.reduce_max(X[s:e], axis=0)]
        return tf.stack(_X, axis=0)


class GraphConvolution(tf.keras.layers.Layer):
    def __init__(self, input_size, hidden_size, act, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.hidden_size = hidden_size
        self.act = act
        self.build((1, input_size))

    def build(self, input_shape):
        self.w1 = self.add_weight(
            name=self.name + "_w1",
            shape=(input_shape[-1], self.hidden_size),
            initializer='random_normal',
            trainable=True)
        self.w2 = self.add_weight(
            name=self.name + "_w2",
            shape=(input_shape[-1], self.hidden_size),
            initializer='random_normal',
            trainable=True)
        self.b = self.add_weight(
            name=self.name + "_b",
            shape=(self.hidden_size,),
            initializer='random_normal',
            trainable=True)
        self.built = True

    def call(self, X, L):
        X0 = tf.einsum('ab,bc->ac', X, self.w1)  # Node
        X1 = tf.einsum('ab,bc->ac', X, self.w2)  # Neigh
        # Neigh conv
        X1 = tfg.geometry.convolution.graph_convolution.edge_convolution_template(
            X1, L, sizes=None,
            edge_function=lambda x, y: y,
            reduction='weighted',
            edge_function_kwargs={}
        )
        X = X0 + X1 + self.b
        X = self.act(X)
        return X


class PhysicsConvolution(tf.keras.layers.Layer):
    def __init__(self, input_size, hidden_size, act, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.hidden_size = hidden_size
        self.act = act
        self.build((1, input_size))

    def build(self, input_shape):
        self.w = self.add_weight(
            name=self.name + "_w",
            shape=(input_shape[-1], self.hidden_size),
            initializer='random_normal',
            trainable=True)
        self.b = self.add_weight(
            name=self.name + "_b",
            shape=(self.hidden_size,),
            initializer='random_normal',
            trainable=True)
        self.built = True

    def call(self, notes, edge_tensor, garment_size):
        X0 = tf.einsum('ab,bc->ac', notes, self.w)  # Node
        X1 = tfg.geometry.convolution.graph_convolution.edge_convolution_template(
            X0, edge_tensor, sizes=None,
            edge_function=lambda x, y: y,
            reduction='weighted',
            edge_function_kwargs={}
        )
        X = X1 + self.b
        X = tf.concat([self.act(X), X0[garment_size:]], axis=0)
        return X


class BodyConvClothGraphConvolution(tf.keras.layers.Layer):
    def __init__(self, input_size, hidden_size, act, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.hidden_size = hidden_size
        self.act = act
        self.build((1, input_size))

    def build(self, input_shape):
        self.w = self.add_weight(
            name=self.name + "_w",
            shape=(input_shape[-1], self.hidden_size),
            initializer='random_normal',
            trainable=True)
        self.b = self.add_weight(
            name=self.name + "_b",
            shape=(self.hidden_size,),
            initializer='random_normal',
            trainable=True)
        self.built = True

    def call(self, notes, weight):
        # 构建边
        diag_index = tf.tile(tf.expand_dims(tf.constant([i for i in range(weight.shape[0])]), axis=1), (1, 2))
        diag_value = tf.ones(shape=[weight.shape[0], ])
        tmp_indices = tf.where(tf.not_equal(weight, tf.constant(0, dtype=tf.float32)))
        data = tf.gather_nd(weight, tmp_indices)
        expand = tf.tile(tf.constant([0, weight.shape[0]], shape=(1, 2), dtype=tf.int64),
                         (tmp_indices.shape[0], 1))
        tmp_indices = tmp_indices + expand
        diag_index = tf.concat([tf.cast(diag_index, dtype=tf.int64), tmp_indices], axis=0)
        diag_value = tf.concat([diag_value, data], axis=0)

        edge_tensor = tf.SparseTensor(diag_index, diag_value, dense_shape=(
            weight.shape[0] + weight.shape[1], weight.shape[0] + weight.shape[1]))

        X0 = tf.einsum('ab,bc->ac', notes, self.w)  # Node
        # Neigh conv
        X1 = tfg.geometry.convolution.graph_convolution.edge_convolution_template(
            X0, edge_tensor, sizes=None,
            edge_function=lambda x, y: y,
            reduction='weighted',
            edge_function_kwargs={}
        )
        X = X1 + self.b
        X = tf.concat([self.act(X), X0[weight.shape[0]:]], axis=0)
        return X

