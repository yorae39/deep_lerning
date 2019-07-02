import pandas as pd
"CLASSE QUE CLASSIFICA DE MODO SEQUENCIAL"
from keras.models import Sequential
"CLASSE QUE É USADA PARA CAMADAS DESNSAS"
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv("entradas-breast.csv")
classe =  pd.read_csv("saidas-breast.csv")

def criarRede(optimizer, loss, kernel_initializer, activation, neurons):
    
    classificador = Sequential()
    
    "ADICIONANDO DADOS A REDE"
    # parametros: 
    # neuronios da camada oculta - Total de entradas + total da saida / 2 (ponto de partida)
    # função de ativação -> relu (melhores resultados)
    # inicializador
    # elementos da camada de entrada
    classificador.add(Dense(units = neurons, activation = activation, 
                            kernel_initializer = kernel_initializer, input_dim = 30))
    classificador.add(Dropout(0.2))    
    
    "ACRESCENTANDO UMA CAMADA - SEM ESSA CAMADA O RESULTADO FICOU RUIM 0,64% --> COM 0,94%"
    classificador.add(Dense(units=neurons, activation=activation,
                           kernel_initializer=kernel_initializer))   
    
    "ELIMINAÇÃO ALEATÓRIA DE ENTRADANPARA MELHORIA DO RESULTADO"
    classificador.add(Dropout(0.2))    
    
    "PARA O CASO DE UMA CLASSIFICAÇÃO BINÁRIA - SIGMOID"
    classificador.add(Dense(units = 1, activation = 'sigmoid'))  
    
    "COMPILANDO A REDE NEURAL - GRADIENTE STOCHATISCO"
    # parametros:
    # optimizer = adam (padrão)
    # loss - função de perda = para casos binarios (binary_crossentropy)
    # metricas - acurácia =  para casos binarios (binary_accuracy)
    classificador.compile(optimizer = optimizer, loss = loss,
                          metrics = ['binary_accuracy'])
    return classificador


classificador = KerasClassifier(build_fn=criarRede)
parametros = {'batch_size': [10, 30],
              'epochs': [50, 100],
              'optimizer': ['adam', 'sgd'],
              'loss': ['binary_crossentropy', 'hinge'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tanh'],
              'neurons': [16, 8]}

grid_search = GridSearchCV(estimator = classificador,
                           param_grid = parametros,
                           scoring = 'accuracy',
                           cv = 5)

grid_search = grid_search.fit(previsores, classe)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_







