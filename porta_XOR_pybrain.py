#Implantação de porta lógica XOR com redes neurais usando pybrain

#Foi necessário instalar o pybrain pelo github e no arquivo "C:\ProgramData\Anaconda3\lib\site-packages\pybrain\tools\functions.py" alterar o "expm2" por "expm" no "from scipy.linalg" devido a alteração de nome da biblioteca na versão
#!pip install https://github.com/pybrain/pybrain/archive/0.3.3.zip

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.structure.modules import SigmoidLayer


'''
rede = buildNetwork(2, 3, 1)
#a rede neural é criada com 2 neurônios de entrada, 3 neurônios na camada oculta e 1 de saída
#é possível setar manualmente os parâmetros como por exemplo:

rede = buildNetwork(2, 3, 1, outclass = SoftmaxLayer,
                    hiddenClass = SigmoidLayer, bias = False)

#é possível verificar os tipos/função de ativação das camadas com um print chamando assim:

print(rede['in'])
#camada de entrada

print(rede['hidden0'])
#camada oculta

print(rede['out'])
#camada de saída

print(rede['bias'])
#bias

'''

rede = buildNetwork(2, 3, 1)
#a rede neural é criada com 2 neurônios de entrada, 3 neurônios na camada oculta e 1 de saída

base = SupervisedDataSet(2, 1)
#teremos 2 atributos previsores e uma classe, ou seja: podemos dizer que a combinação 00 vale 0, 01 vale 1, etc.

base.addSample((0, 0), (0, ))
#adicionando elementos: a combinação de 0 e 0 na entrada va resultar em 0 na saída

base.addSample((0, 1), (1, ))
#passando 0 e 1 o resultado vai ser 1

base.addSample((1, 0), (1, ))
#1 e 0 vai resultar em 1

base.addSample((1, 1), (0, ))
#1 e 1 vai resultar em 0

#Agora vou iniciar o treino:
treinamento = BackpropTrainer(rede, dataset = base, learningrate = 0.01,
                              momentum = 0.06)

#Definindo número de épocas em um for:
for i in range(1, 30000):
    erro = treinamento.train()
    #printar a porcentagem do erro a cada 1000 loops do for para ter noção
    if i % 1000 == 0:
        print("Erro %s" % erro)

#E por fim pode-se verificar os resultados do treinamento, vendo as respostas/saídas de acordo com as entradas    
print(rede.activate([0, 0]))
print(rede.activate([0, 1]))
print(rede.activate([1, 0]))
print(rede.activate([1, 1]))

#Os valores muito baixos são mostrados no formato de notação científica




