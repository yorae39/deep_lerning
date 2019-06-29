import numpy as np

#Trasnfer function
def stepFunction(soma):
    if(soma >= 1):
        return 1
    return 0

#CLASSIICAÇÃO BINÁRIA (0 E 1)
def sigmodeFunction(soma):
    return 1 / (1 + np.exp(-soma))

#CLASSIFICAÇÃO (-1 E 1)
def hiperTangFunction(soma):
    return (np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))

#REDES CONVULUCIONAIS
def reLUFunction(soma):
    if(soma >= 0):
     return soma
    return 0

#USADA PARA REGRESSÃO
def linearFunction(soma):
    return soma

#CLASSIFICAÇÃO COM MAIS DE DUAS CLASSES
def softMaxFunction(x):
    ex = np.exp(x)
    return ex / ex.sum()
    
teste = stepFunction(2.1)
teste = sigmodeFunction(2.1)
teste = hiperTangFunction(2.1)
teste = reLUFunction(30)
teste = linearFunction(-0.358)

valores = [5.0, 2.0, 1.3]
print(softMaxFunction(valores))