import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

print('Carregando Arquivo de teste')
arquivo = np.load('teste5.npy')
x = arquivo[0]
y = np.ravel(arquivo[1])

list_min = []

menor_erro = 0
menor_i = 0
menor_y_est = []

for i in range(5):

    regr = MLPRegressor(hidden_layer_sizes=(100),
                        max_iter=100000,
                        activation='logistic', #{'identity', 'logistic', 'tanh', 'relu'},
                        solver='adam',
                        learning_rate = 'adaptive',
                        n_iter_no_change=50)
    
    #print('Treinando RNA')
    regr = regr.fit(x,y)

    #print('Preditor')
    y_est = regr.predict(x)

    plt.figure(figsize=[14,7])

    #plot curso original
    plt.subplot(1,3,1)
    plt.plot(x,y)

    #plot aprendizagem
    plt.subplot(1,3,2)
    plt.plot(regr.loss_curve_)
    list_min.append(regr.best_loss_)
    print("Gráfico {}".format(i))
    print("Menor loss:" + str(round(regr.best_loss_, 2)))

    #plot regressor
    plt.subplot(1,3,3)
    plt.plot(x,y,linewidth=1,color='yellow')
    plt.plot(x,y_est,linewidth=2)

    if (i == 0):
        menor_erro = regr.best_loss_
        menor_loss_curve = regr.loss_curve_
        menor_y_est = y_est.copy()
        menor_i = i
    elif (menor_erro > regr.best_loss_):
        menor_i = i
        menor_erro = regr.best_loss_
        menor_loss_curve = regr.loss_curve_.copy()
        menor_y_est = y_est.copy()

    plt.show()

print("\nMelhor gráfico: {}".format(menor_i))
print("Média:" + str(round(np.average(list_min),2)))
print("Desvio padrão:" + str(round(np.std(list_min),2)))

plt.figure(figsize=[14,7])

#plot curso original
plt.subplot(1,3,1)
plt.plot(x,y)

#plot aprendizagem
plt.subplot(1,3,2)
plt.plot(menor_loss_curve)

#plot regressor
plt.subplot(1,3,3)
plt.plot(x,y,linewidth=1,color='yellow')
plt.plot(x,menor_y_est,linewidth=2)

plt.show()
