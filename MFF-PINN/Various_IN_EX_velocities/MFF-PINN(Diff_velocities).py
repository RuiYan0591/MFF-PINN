"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
from scipy.io import loadmat
import numpy as np
from deepxde.backend import tf
import matplotlib.pyplot as plt
o = tf.math.abs(tf.Variable(1.0))
X_min=0
X_max=100
T_min=0
T_max=10

def gen_traindata():
    data = loadmat("GFDM_u1_v0.5.mat")#Change according to flow velocity combination
    t = data["t"]
    x = data["x"]
    y = data["y"]
    q = data["q"]
    X, T = np.meshgrid(x, t)
    X = np.reshape(X, (-1, 1))
    T = np.reshape(T, (-1, 1))
    Y = np.reshape(y, (-1, 1))
    Q = np.reshape(q, (-1, 1))
    return np.hstack((X, T)), Y, Q


def get_initial_loss(model):
    model.compile("adam", lr=0.001)
    losshistory, train_state = model.train(0)
    return losshistory.loss_train[0]

def pde(x, y):
    ...
    return x,y

def boundary_yl(x, on_boundary):
    return on_boundary and np.isclose(x[0], X_min)
def boundary_yr(x, on_boundary):
    return on_boundary and np.isclose(x[0], X_max)

def boundary_ql(x, on_boundary):
    return on_boundary and np.isclose(x[1], X_min)
def boundary_qr(x, on_boundary):
    return on_boundary and np.isclose(x[1], X_max)

def func_yBC(x,y):
    dy_x = tf.gradients(y, x)[0][:, 0:1]
    dy_xx = tf.gradients(dy_x, x)[0][:, 0:1]
    return dy_xx - 0

def func_qBC(x,q):
    dq_x = tf.gradients(q, x)[0][:, 0:1]
    dq_xx = tf.gradients(dq_x, x)[0][:, 0:1]
    return dq_xx - 0

def boundary_initial(x, _):
    return np.isclose(x[1], 0)

geom = dde.geometry.Interval(X_min, X_max)
timedomain = dde.geometry.TimeDomain(T_min, T_max)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc_1 = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_yl)
bc_2 = dde.icbc.OperatorBC(geomtime, func_yBC, boundary_yl)
bc_3 = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_yr)
bc_4 = dde.icbc.OperatorBC(geomtime, func_yBC, boundary_yr)
bc_5 = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_ql)
bc_6 = dde.icbc.OperatorBC(geomtime, func_qBC, boundary_ql)
bc_7 = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_qr)
bc_8 = dde.icbc.OperatorBC(geomtime, func_qBC, boundary_qr)
ic_1 = dde.icbc.IC(geomtime, lambda x: 0, lambda _, on_initial: on_initial)

observe_x, Y, Q = gen_traindata()
observe_y = dde.icbc.PointSetBC(observe_x, Y, component=0)

data = dde.data.TimePDE(geomtime,
    pde,
    [observe_y],
    num_domain=5000,
    num_boundary=100,
    num_initial=100,
    anchors=observe_x,
    num_test=10000)

layer_size = [2] + [50] * 3 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.STMsFFN(layer_size, activation, initializer, sigmas_x=[1], sigmas_t=[1,21,33])#Change according to flow velocity combination
model = dde.Model(data, net)
initial_losses = get_initial_loss(model)
loss_weights = 1 / initial_losses
model.compile("adam",lr=0.001,loss_weights=loss_weights,decay=("inverse time", 2000, 0.9),)
pde_residual_resampler = dde.callbacks.PDEResidualResampler(period=1)
losshistory, train_state = model.train(epochs = 0, display_every = 1, callbacks = [pde_residual_resampler],model_restore_path = r"Diff_u1_v0.5")#Change according to flow velocity combination

# The vibration data
T_min = 0
T_max = 10
N_nn = 5000
T_nn = np.linspace(T_min, T_max, N_nn)
T_nn = np.reshape(T_nn, (len(T_nn), 1))
X_nn = np.ones((N_nn, 1)) * 50
Q_nn = np.hstack((X_nn, T_nn))
W_nn = model.predict(Q_nn)
R_nn = np.hstack((Q_nn, W_nn))
np.savetxt("VIV_Diff_u1_v0.5_ξ0.5.txt", R_nn)

# The variation cloud diagram data
X, y_ture, Q = gen_traindata()
W = model.predict(X)
R = np.hstack((X, W))
R1 = np.hstack((X, y_ture))
np.savetxt("Yuntu_Diff_u1_v0.5.txt", R)
np.savetxt("Yuntu_Diff_u1_v0.5_true.txt", R1)


X, Y, Q = gen_traindata()
y_true = np.hstack((Y,Q))
YY_true = y_true[:, 0:1]
y_pred = model.predict(X)
YY_pred = y_pred[:, 0:1]
f = model.predict(X, operator=pde)
print("L2 relative error y:", dde.metrics.l2_relative_error(YY_true, YY_pred))
print("RMSE y:", np.sqrt(dde.metrics.mean_squared_error(YY_true, YY_pred)))

