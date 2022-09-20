import tensorflow as tf

from data.data_loader import Data
from model.layers import GraphConvolution, MaxPool, BodyConvClothGraphConvolution
from model.smpl.smpl_np import SMPLModel
from values import rest_pose
import numpy as np
import os


class GSNet(tf.keras.Model):
    def __init__(self, psd_dim, checkpoint=None, rest_pose=rest_pose, train_type='supervised', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.psd_dim = psd_dim
        self.train_type = train_type
        self._best = float('inf')  # best metric obtained with this model

        smpl_path = os.path.dirname(os.path.abspath(__file__)) + '/smpl/'
        self.SMPL = {
            0: SMPLModel(smpl_path + 'model_f.pkl', rest_pose),
            1: SMPLModel(smpl_path + 'model_m.pkl', rest_pose)
        }

        self.phi = [
            GraphConvolution(hidden_size=32, act=tf.nn.relu, name='phi0', input_size=6),
            GraphConvolution(hidden_size=64, act=tf.nn.relu, name='phi1', input_size=32),
            GraphConvolution(hidden_size=128, act=tf.nn.relu, name='phi2', input_size=64),
            GraphConvolution(hidden_size=256, act=tf.nn.relu, name='phi3', input_size=128)
        ]

        self.glb0 = tf.keras.Sequential(
            [tf.keras.layers.Dense(256, activation=tf.keras.activations.relu, use_bias=True, name='glb0')])
        self.glb0.build(input_shape=(1, 256))
        self.glb1 = MaxPool()

        self.omega = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.keras.activations.relu, use_bias=True, name='omega0'),
            tf.keras.layers.Dense(64, activation=tf.keras.activations.relu, use_bias=True, name='omega1'),
            tf.keras.layers.Dense(32, activation=tf.keras.activations.relu, use_bias=True, name='omega2'),
            tf.keras.layers.Dense(24, activation=tf.keras.activations.relu, use_bias=True, name='omega3')
        ])
        self.omega.build(input_shape=(1, 518))

        self.psi = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation=tf.keras.activations.relu, use_bias=True, name='psi0'),
            tf.keras.layers.Dense(256, activation=tf.keras.activations.relu, use_bias=True, name='psi1'),
            tf.keras.layers.Dense(256, activation=tf.keras.activations.relu, use_bias=True, name='psi2'),
            tf.keras.layers.Dense(self.psd_dim * 3, use_bias=True, name='psi3')
        ])
        self.psi.build(input_shape=(1, 518))

        self.chi = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation=tf.keras.activations.relu, use_bias=True, name='chi0'),
            tf.keras.layers.Dense(256, activation=tf.keras.activations.relu, use_bias=True, name='chi1'),
            tf.keras.layers.Dense(256, activation=tf.keras.activations.relu, use_bias=True, name='chi2'),
            tf.keras.layers.Dense(self.psd_dim * 3, use_bias=True, name='chi3')
        ])
        self.chi.build(input_shape=(1, 518))

        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation=tf.keras.activations.relu, use_bias=True, name='mlp0'),
            tf.keras.layers.Dense(256, activation=tf.keras.activations.relu, use_bias=True, name='mlp1'),
            tf.keras.layers.Dense(256, activation=tf.keras.activations.relu, use_bias=True, name='mlp2'),
            tf.keras.layers.Dense(self.psd_dim, activation=tf.keras.activations.relu, use_bias=True, name='mlp3')
        ])
        self.mlp.build(input_shape=(1, 72))

        self.template_fc = tf.keras.Sequential([
            tf.keras.layers.Dense(8, use_bias=True, name='template_fc0'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.LayerNormalization(name='template_fc1', epsilon=1e-6),
            tf.keras.layers.Dense(16, use_bias=True, name='template_fc2'),
            tf.keras.layers.LeakyReLU()
        ])
        self.template_fc.build(input_shape=(1, 3))

        self.edge_conv = [
            BodyConvClothGraphConvolution(hidden_size=32, act=tf.nn.leaky_relu, name='edge_conv0', input_size=16),
            BodyConvClothGraphConvolution(hidden_size=64, act=tf.nn.leaky_relu, name='edge_conv1', input_size=32),
            BodyConvClothGraphConvolution(hidden_size=128, act=tf.nn.leaky_relu, name='edge_conv2', input_size=64),
            BodyConvClothGraphConvolution(hidden_size=256, act=tf.nn.tanh, name='edge_conv3', input_size=128)
        ]

        self.deformation_fc = tf.keras.Sequential([
            tf.keras.layers.Dense(256, use_bias=True, name='deformation_fc0'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.LayerNormalization(name='deformation_fc1', epsilon=1e-6),
            tf.keras.layers.Dense(128, use_bias=True, name='deformation_fc2'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(64, use_bias=True, name='deformation_fc3'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(32, use_bias=True, name='deformation_fc4'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(16, use_bias=True, activation=tf.keras.activations.tanh, name='deformation_fc5'),
        ])
        self.deformation_fc.build(input_shape=(1, 256))

        # load pre-trained
        if checkpoint is not None:
            print("Loading pre-trained model: " + checkpoint)
            self.load(checkpoint)

    def gather(self):
        var = []
        if self.train_type == 'supervised':
            for gcn in self.phi:
                var += gcn.weights
            var = var + self.glb0.weights + self.omega.weights + self.psi.weights + self.mlp.weights
        elif self.train_type == 'skinning2':
            for gcn in self.phi:
                var += gcn.weights
            var = var + self.glb0.weights + self.omega.weights + self.psi.weights + self.mlp.weights + self.template_fc.weights + self.deformation_fc.weights
            for gcn in self.edge_conv:
                var += gcn.weights
        elif self.train_type == 'only_skinning2':
            var = var + self.template_fc.weights + self.deformation_fc.weights
            for gcn in self.edge_conv:
                var += gcn.weights
        elif self.train_type == 'unsupervised':
            var = self.chi.weights
        return var

    def gather_all(self):
        var = []
        for gcn in self.phi:
            var += gcn.weights
        var = var + self.glb0.weights + self.omega.weights + self.psi.weights + self.mlp.weights + self.chi.weights + self.template_fc.weights + self.deformation_fc.weights
        for gcn in self.edge_conv:
            var += gcn.weights
        return var

    def load(self, checkpoint):
        # checkpoint: path to pre-trained model
        # list vars
        vars = self.gather_all()
        # load vars values
        if not checkpoint.endswith('.npy'): checkpoint += '.npy'
        values = np.load(checkpoint, allow_pickle=True)[()]
        # assign
        _vars = set([v.name for v in vars])
        _vars_chck = set(values.keys()) - {'best'}
        _diff = sorted(list(_vars_chck - _vars))
        if len(_diff):
            print("Model missing vars:")
            for v in _diff: print("\t" + v)
        _diff = sorted(list(_vars - _vars_chck))
        if len(_diff):
            print("Checkpoint missing vars:")
            for v in _diff: print("\t" + v)
        for v in vars:
            try:
                v.assign(values[v.name])
            except:
                if v.name not in values:
                    continue
                else:
                    print("Mismatch in variable shape:")
                    print("\t" + v.name)
        if 'best' in values: self._best = values['best']

    def save_model(self, checkpoint):
        # checkpoint: path to save the pre-trained model
        print("\tSaving checkpoint: " + checkpoint)
        # get vars values
        values = {v.name: v.numpy() for v in self.gather_all()}
        if self._best is not float('inf'): values['best'] = self._best
        # save weights
        if not checkpoint.endswith('.npy'): checkpoint += '.npy'
        np.save(checkpoint, values)

    def _concat_descriptors(self, X, F, indices):
        F_tile = []
        for i in range(1, len(indices)):
            n = indices[i] - indices[i - 1]
            F_tile += [tf.tile(tf.expand_dims(F[i - 1], 0), [n, 1])]
        F_tile = tf.concat(F_tile, axis=0)
        return tf.concat((X, F_tile), axis=-1)

    # combines pose embedding and PSD matrices to obtain final deformations
    def _deformations(self, X, PSD, indices):
        D = []
        for i in range(1, len(indices)):
            s, e = indices[i - 1], indices[i]
            _X = X[i - 1]
            _PSD = PSD[s:e]
            _D = tf.einsum('a,bac->bc', _X, _PSD)
            D += [_D]
        return tf.concat(D, axis=0)

    def _ones(self, T_shape):
        return tf.ones((T_shape[0], 1), tf.float32)

    # Computes the skinning for each outfit/pose
    def _skinning(self, T, W, G, indices):
        V = []
        for i in range(1, len(indices)):
            s, e = indices[i - 1], indices[i]
            _T = T[s:e]
            _G = G[i - 1]
            _weights = W[s:e]
            _G = tf.einsum('ab,bcd->acd', _weights, _G)
            _T = tf.concat((_T, self._ones(tf.shape(_T))), axis=-1)
            _T = tf.linalg.matmul(_G, _T[:, :, None])[:, :3, 0]
            V += [_T]
        return tf.concat(V, axis=0)

    def _skinning2(self, T, W, G, PGs, indices):
        V = []
        for i in range(1, len(indices)):
            s, e = indices[i - 1], indices[i]
            _T = T[s:e]
            _TT = self.template_fc(_T)
            _G = tf.constant(np.reshape(G[i - 1], (24, 16)), dtype=tf.float32)
            _PG = tf.constant(np.reshape(PGs[i - 1], (24, 16)), dtype=tf.float32)
            _TT = tf.concat([_TT, (_G - _PG)], axis=0)
            _weights = W[s:e]
            _G = self.template_conv(_TT, _weights)
            _G = _G[:_T.shape[0]]
            _G = self.deformation_fc(_G)
            _G = tf.reshape(_G, [_G.shape[0], 4, 4])
            _T = tf.concat((_T, self._ones(tf.shape(_T))), axis=-1)
            _T = tf.linalg.matmul(_G, _T[:, :, None])[:, :3, 0]
            V += [_T]
        return tf.concat(V, axis=0)

    def template_conv(self, X, L):
        # X: template outfit verts
        # L: template outfit laplacian
        for l in self.edge_conv:
            X = l(X, L)
        return X

    # Computes the transformation matrices of each joint of the skeleton for each pose
    def _transforms(self, poses, shapes, genders, with_body):
        G = []
        B = []
        for p, s, g in zip(poses, shapes, genders):
            _G, _B = self.SMPL[g].set_params(pose=p, beta=s, with_body=with_body)
            G += [_G]
            B += [_B]
        return np.stack(G), np.stack(B)

    # Computes geometric descriptors for each vertex of the template outfit
    def _descriptors(self, X, L):
        # X: template outfit verts
        # L: template outfit laplacian
        for l in self.phi: X = l(X, L)
        return X

    def call(self, T, L, P, S, G, fabric, tightness, indices, with_body=False):
        Gs, B1 = self._transforms(P, S, G, True)
        PGs, PB1 = self._transforms(Data.pre_pose_buffer, S, G, True)
        P = tf.cast(P, dtype=tf.float32)
        X = self._descriptors(T, L)
        GLB = self.glb0(X)
        GLB = self.glb1(GLB, indices)
        X = self._concat_descriptors(X, GLB, indices)
        X = tf.concat((X, fabric), axis=-1)
        X = self._concat_descriptors(X, tightness, indices)
        self.W = self.omega(X)
        self.W = self.W / (tf.reduce_sum(self.W, axis=-1, keepdims=True) + 1e-7)

        PSD = self.psi(X)
        PSD = tf.reshape(PSD, (-1, self.psd_dim, 3))

        """ DYNAMIC """
        X = self.mlp(P)
        X /= tf.reduce_sum(X, axis=-1, keepdims=True)

        # deformations
        self.D = self._deformations(X, PSD, indices)
        if 'skinning2' == self.train_type:
            V = self._skinning2(tf.concat(Data.pre_gt_buffer, axis=0) + self.D, self.W, Gs, PGs, indices)
        elif 'only_skinning2' == self.train_type:
            V = self._skinning2(tf.concat(Data.pre_gt_buffer, axis=0) + tf.stop_gradient(self.D),
                                tf.stop_gradient(self.W), Gs, PGs, indices)
        else:
            V = self._skinning(tf.concat(Data.pre_gt_buffer, axis=0) + self.D, self.W, Gs, indices)
        return V, B1
