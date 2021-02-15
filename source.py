import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
def f(x):
    return np.array([[1], [x], [x ** 2], [x ** 3]])

#вычисление p плана по задан. q
def get_p(q):
    return np.array([q, q * 250.0 / 107.0, 1 - q * 714.0 / 107.0,
                     q * 250.0 / 107.0, q])

#выч-е информац.  матрицы
def calc_mtrx_M(P,X):
    mtrx_M = np.zeros((4,4))
    for x, p in zip(X,P):
        _f = f(x)
        mtrx_M += p * np.dot(_f, np.transpose(_f))
    return mtrx_M

#выч-е дисперсион. матрицы и ее собств. чисел
def calc_mtrx_D(mtrx_M):
    mtrx_D = np.linalg.inv(mtrx_M)
    return mtrx_D, np.linalg.eigvals(mtrx_D)

#дисперсия оценки функции отклика
def calc_d(x, mtrx_D):
    _f = f(x)
    result = np.dot(np.transpose(_f), mtrx_D)
    result = np.dot(result, _f)
    return result[0][0]

#вычисление критерия lambda
def calc_Lambda(eig_D):
    return np.sum(np.square(eig_D - np.mean(eig_D)))

#вычисление критерия G
def calc_G(X,mtrx_D):
    G = np.NINF
    for x in X:
        temp = calc_d(x, mtrx_D)
        if temp > G: G = temp
    return G

#загрузка и выч-е исход.  данных
#path - папка с планами
def data_load(path):
    data = pd.DataFrame([], columns = ['plan','D','A','E',
                                       'Phi2','Lambda','MV','G'])
    os.chdir(path)
    #[номер плана].txt
    files = os.listdir()
    for i in range(len(files)):
        #отбрасывание .txt из номеров планов
        n_plan = files[i].replace('.txt','')
        #plan[0] - x, plan[1] - p
        plan = np.loadtxt(files[i])
        mtrx_M = calc_mtrx_M(plan[1], plan[0])
        mtrx_D, eig_D = calc_mtrx_D(mtrx_M)
        #D - критерий
        D = np.linalg.det(mtrx_D)
        #A - критерий
        A = np.trace(mtrx_D)
        #E - критерий
        E = np.max(eig_D)
        #Phi2 - критерий
        Phi2 = 0.5 * np.trace(np.dot(mtrx_D, mtrx_D))
        #Lambda - критерий
        Lambda = calc_Lambda(eig_D)
        #MV - критерий
        MV = np.max(np.diag(mtrx_D))
        #G - критерий
        G = calc_G(plan[0], mtrx_D)
        
        data.loc[n_plan] = [plan, D, A, E, Phi2, Lambda, MV, G]
    return data

#вычисление lambda - критерия при заданных зн-х q
def get_q_lambda(n_plan, q_min, q_max, h):
    list_q = []
    list_lambda = []
    plan = data['plan'][n_plan]
    X = plan[0]
    P = plan[1]
    q = q_min
    q_opt = P[0]
    step = (q_opt - q_min) / h
    while q < q_max:
        P = get_p(q)
        mtrx_M = calc_mtrx_M(P, X)
        mtrx_D, eig_D = calc_mtrx_D(mtrx_M)
        Lambda = calc_Lambda(eig_D)
        list_q.append(q)
        list_lambda.append(Lambda)
        q += step
    return list_q, list_lambda
data = data_load('plans')
q, Lambda = get_q_lambda('10', 0.01, 107.0 / 714.0, 10)
a_Lambda = np.array(Lambda)
data['D_opt'] = data['D'].rank()
data['A_opt'] = data['A'].rank()
data['E_opt'] = data['E'].rank()
data['Phi2_opt'] = data['Phi2'].rank()
data['Lambda_opt'] = data['Lambda'].rank()
data['MV_opt'] = data['MV'].rank()
data['G_opt'] = data['G'].rank()
opt_plans = data[['D_opt','A_opt','E_opt','Phi2_opt','Lambda_opt','MV_opt','G_opt','Lambda']]
plt.plot(q, Lambda,'.-')
plt.ylabel('Критерий Lambda')
plt.xlabel('q')
plt.grid()
plt.show()
min_index = a_Lambda.argmin()
print(min_index)
print(len(a_Lambda))
print(q[min_index])
print(Lambda[min_index])
print(get_p(q[min_index]))