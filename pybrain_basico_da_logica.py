from pybrain.sctructure import FeedForwardNetwork
from pybrain.sctructure import LinearLayer, SigmoidLayer, BiasUnity
from pybrain.sctructure import FullConnection

rede = FeedForwardNetwork()

camadaEntrada = LinearLayer(2)
camadaOculta = SigmoidLayer(3)
camadaSaida = SigmoidLayer(1)
bias1 = BiasUnity()
bias2 = BiasUnity()

rede.addModule(camadaEntrada)
rede.addModule(camadaOculta)
rede.addModule(camadaSaida)
rede.addModule(bias1)
rede.addModule(bias2)

entradaOculta = FullConnection(camadaEntrada, camadaOculta)
ocultaSaida = FullConnection(camadaOculta, camadaSaida)
biasOculta = FullConnection(bias1, camadaEntrada)
biasSaida = FullConnection(bias2, camadaSaida)

rede.sortModules()

print(rede)
print(entradaOculta.params)
print(ocultaSaida.params)
print(biasOculta.params)
print(biasSaida.params)