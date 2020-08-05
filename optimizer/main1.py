import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plt_cl

# ------------------定义目标函数beale、目标函数的偏导函数dbeale_dx，并画出目标函数---------------------
# 定义函数beale
def beale(x1, x2):
    return (1.5 - x1 + x1 * x2) ** 2 + (2.25 - x1 + x1 * x2 ** 2) ** 2 + (2.625 - x1 + x1 * x2 ** 3) ** 2

# 定义函数beale的偏导
def dbeale_dx(x1, x2):
    dfdx1 = 2 * (1.5 - x1 + x1 * x2) * (x2 - 1) + 2 * (2.25 - x1 + x1 * x2 ** 2) * (x2 ** 2 - 1) + 2 * (
            2.625 - x1 + x1 * x2 ** 3) * (x2 ** 3 - 1)
    dfdx2 = 2 * (1.5 - x1 + x1 * x2) * x1 + 2 * (2.25 - x1 + x1 * x2 ** 2) * (2 * x1 * x2) + 2 * (
            2.625 - x1 + x1 * x2 ** 3) * (3 * x1 * x2 ** 2)
    return dfdx1, dfdx2
step_x1, step_x2 = 0.2, 0.2
X1, X2 = np.meshgrid(np.arange(-5, 5 + step_x1, step_x1),
                     np.arange(-5, 5 + step_x2, step_x2))
Y = beale(X1, X2)
print("目标结果 (x_1, x_2) = (3, 0.5)")

# 定义画图函数
def gd_plot(x_traj):
    plt.rcParams['figure.figsize'] = [6, 6]
    plt.contour(X1, X2, Y, levels=np.logspace(0, 6, 30),
                norm=plt_cl.LogNorm(), cmap=plt.cm.jet)
    plt.title('2D Contour Plot of Beale function(Momentum)')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.axis('equal')
    plt.plot(3, 0.5, 'k*', markersize=10)
    if x_traj is not None:
        x_traj = np.array(x_traj)
        plt.plot(x_traj[:, 0], x_traj[:, 1], 'k-')
    plt.show()

gd_plot(None)

# ------------------------------------------------------------无优化器-------------------------------------------
def gd_no(df_dx, x0, conf_para=None):
    if conf_para is None:
        conf_para = {}
    conf_para.setdefault('n_iter', 1000)  # 迭代次数
    conf_para.setdefault('learning_rate', 0.001)  # 设置学习率
    x_traj = []
    x_traj.append(x0)
    v = np.zeros_like(x0)
    for iter in range(1, conf_para['n_iter'] + 1):
        x_traj.append(x_traj[-1])
    return x_traj

x0 = np.array([1.0, 1.5])
conf_para_no = {'n_iter': 2000, 'learning_rate': 0.005}
x_traj_no = gd_no(dbeale_dx, x0, conf_para_no)
print("无优化器求得极值点 (x_1, x_2) = (%s, %s)" % (x_traj_no[-1][0], x_traj_no[-1][1]))
gd_plot(x_traj_no)

# ------------------------------------------------------------SGD-------------------------------------------
def gd_sgd(df_dx, x0, conf_para=None):
    if conf_para is None:
        conf_para = {}
    conf_para.setdefault('n_iter', 1000)  # 迭代次数
    conf_para.setdefault('learning_rate', 0.001)  # 设置学习率
    x_traj = []
    x_traj.append(x0)
    v = np.zeros_like(x0)
    for iter in range(1, conf_para['n_iter'] + 1):
        dfdx = np.array(df_dx(x_traj[-1][0], x_traj[-1][1]))
        v = - conf_para['learning_rate'] * dfdx
        x_traj.append(x_traj[-1] + v)
    return x_traj

x0 = np.array([1.0, 1.5])
conf_para_sgd = {'n_iter': 2000, 'learning_rate': 0.005}
x_traj_sgd = gd_sgd(dbeale_dx, x0, conf_para_sgd)
print("SGD求得极值点 (x_1, x_2) = (%s, %s)" % (x_traj_sgd[-1][0], x_traj_sgd[-1][1]))
gd_plot(x_traj_sgd)

# -------------------------------------------------------Momentum---------------------------------
def gd_momentum(df_dx, x0, conf_para=None):
    if conf_para is None:
        conf_para = {}
    conf_para.setdefault('n_iter', 1000)  # 迭代次数
    conf_para.setdefault('learning_rate', 0.001)  # 设置学习率
    conf_para.setdefault('momentum', 0.9)  # 设置动量参数
    x_traj = []
    x_traj.append(x0)
    v = np.zeros_like(x0)
    for iter in range(1, conf_para['n_iter'] + 1):
        dfdx = np.array(df_dx(x_traj[-1][0], x_traj[-1][1]))
        v = conf_para['momentum'] * v - conf_para['learning_rate'] * dfdx
        x_traj.append(x_traj[-1] + v)
    return x_traj

x0 = np.array([1.0, 1.5])
conf_para_momentum = {'n_iter': 500, 'learning_rate': 0.005}
x_traj_momentum = gd_momentum(dbeale_dx, x0, conf_para_momentum)
print("Momentum求得极值点 (x_1, x_2) = (%s, %s)" % (x_traj_momentum[-1][0], x_traj_momentum[-1][1]))
gd_plot(x_traj_momentum)

# ----------------------------------------------------adagrad-----------------------------
def gd_adagrad(df_dx, x0, conf_para=None):
    if conf_para is None:
        conf_para = {}
    conf_para.setdefault('n_iter', 1000)  # 迭代次数
    conf_para.setdefault('learning_rate', 0.001)  # 学习率
    conf_para.setdefault('epsilon', 1e-7)
    x_traj = []
    x_traj.append(x0)
    r = np.zeros_like(x0)
    for iter in range(1, conf_para['n_iter'] + 1):
        dfdx = np.array(df_dx(x_traj[-1][0], x_traj[-1][1]))
        r += dfdx ** 2
        x_traj.append(x_traj[-1] - conf_para['learning_rate'] / (np.sqrt(r) + conf_para['epsilon']) * dfdx)
    return x_traj

x0 = np.array([1.0, 1.5])
conf_para_adag = {'n_iter': 500, 'learning_rate': 2}
x_traj_adag = gd_adagrad(dbeale_dx, x0, conf_para_adag)
print("Adagrad求得极值点 (x_1, x_2) = (%s, %s)" % (x_traj_adag[-1][0], x_traj_adag[-1][1]))
gd_plot(x_traj_adag)
