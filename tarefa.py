import pandas as pd
import keras
"CLASSE QUE CLASSIFICA DE MODO SEQUENCIAL"
from keras.models import Sequential
"CLASSE QUE É USADA PARA CAMADAS DESNSAS"
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


previsores = pd.read_csv("entradas-breast.csv")
classe =  pd.read_csv("saidas-breast.csv")


def criarRede():
    
    classificador = Sequential()
    
    "ADICIONANDO DADOS A REDE"
    # parametros: 
    # neuronios da camada oculta - Total de entradas + total da saida / 2 (ponto de partida)
    # função de ativação -> relu (melhores resultados)
    # inicializador
    # elementos da camada de entrada
    classificador.add(Dense(units = 16, activation = 'relu', 
                            kernel_initializer = 'normal', input_dim = 30))
    classificador.add(Dropout(0.2))    
    
    "ACRESCENTANDO UMA CAMADA - SEM ESSA CAMADA O RESULTADO FICOU RUIM 0,64% --> COM 0,94%"
    classificador.add(Dense(units = 8, activation='relu',
                           kernel_initializer='random_uniform'))   
    
    "ELIMINAÇÃO ALEATÓRIA DE ENTRADA PARA MELHORIA DO RESULTADO"
    classificador.add(Dropout(0.2))    
    
    "ACRESCENTANDO UMA CAMADA - SEM ESSA CAMADA O RESULTADO FICOU RUIM 0,64% --> COM 0,94%"
    classificador.add(Dense(units = 8, activation='relu',
                           kernel_initializer='random_uniform'))   
    classificador.add(Dropout(0.2)) 
    
    "ACRESCENTANDO UMA CAMADA - SEM ESSA CAMADA O RESULTADO FICOU RUIM 0,64% --> COM 0,94%"
    classificador.add(Dense(units = 8, activation='relu',
                           kernel_initializer='random_uniform'))   
    classificador.add(Dropout(0.2))    
    
    "PARA O CASO DE UMA CLASSIFICAÇÃO BINÁRIA - SIGMOID"
    classificador.add(Dense(units = 1, activation = 'sigmoid'))  
    
    "COMPILANDO A REDE NEURAL - GRADIENTE STOCHATISCO"
    # parametros:
    # optimizer = adam (padrão)
    # loss - função de perda = para casos binarios (binary_crossentropy)
    # metricas - acurácia =  para casos binarios (binary_accuracy)
    #otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
    classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                          metrics = ['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn=criarRede, epochs = 100, batch_size = 10)


resultados = cross_val_score(estimator=classificador,
                             X=previsores, y=classe,
                             cv = 10, scoring = 'accuracy')
media = resultados.mean()
desvio = resultados.std()






