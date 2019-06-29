"AULA 16 - UDEMY DL - 15/05/2019"

import pandas as pd

previsores = pd.read_csv("entradas-breast.csv")
classe =  pd.read_csv("saidas-breast.csv")


from sklearn.model_selection import train_test_split
"DIVISÃO ENTRE TREINO E TESTE"
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)


"CLASSE QUE CLASSIFICA DE MODO SEQUENCIAL"
from keras.models import Sequential
"CLASSE QUE É USADA PARA CAMDAS DESNSAS"
from keras.layers import Dense


"CRIANDO A REDE NEURAL"
classificador = Sequential()


"ADICIONANDO DADOS A REDE"
# parametros: 
# neuronios da camada oculta - Total de entradas + total da saida / 2 (ponto de partida)
# função de ativação -> relu (melhores resultados)
# inicializador
# elementos da camada de entrada
classificador.add(Dense(units = 16, activation = 'relu', 
                        kernel_initializer = 'random_uniform', input_dim = 30))

"ACRESCENTANDO UMA CAMADA - SEM ESSA CAMADA O RESULTADO FICOU RUIM 0,64% --> COM 0,94%"
classificador.add(Dense(units=16, activation='relu',
                       kernel_initializer='random_uniform'))


"PARA O CASO DE UMA CLASSIFICAÇÃO BINÁRIA - SIGMOID"
classificador.add(Dense(units = 1, activation = 'sigmoid'))


"COMPILANDO A REDE NEURAL - GRADIENTE STOCHATISCO"
# parametros:
# optimizer = adam (padrão)
# loss - função de perda = para casos binarios (binary_crossentropy)
# metricas - acurácia =  para casos binarios (binary_accuracy)
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])

"ENCAIXAR PREVISORES COM CLASSES"
# parametros:
# previsores x classes
# batch_size = calcula o erro e depois atualiza os pesos, neste caso de dez em dez
# epochs - quantas vezes vou ajustar os pesos
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs = 100)


"VISUALIZAÇÃO DOS PESOS DA REDE NEURAL"
pesos0 = classificador.layers[0].get_weights()

print(type(pesos0))

print(len(pesos0))

print(pesos0)


pesos1 = classificador.layers[1].get_weights()

pesos2 = classificador.layers[2].get_weights()

print(pesos2)



"TESTES"
previsoes = classificador.predict(previsores_teste)

previsoes = (previsoes > 0.5) #TRUE OU FALSE

from sklearn.metrics import confusion_matrix, accuracy_score

precisao = accuracy_score(classe_teste, previsoes)

print(precisao)

matriz = confusion_matrix(classe_teste, previsoes)

print(matriz)

resultado = classificador.evaluate(previsores_teste, classe_teste)

print(resultado)
"FIM TESTES"

