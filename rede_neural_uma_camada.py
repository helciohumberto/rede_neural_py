import numpy as np

# Função sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada da sigmoide
def sigmoid_derivative(x):
    return x * (1 - x)

# Propagação para frente
def forward(X, pesos, bias):
    z = np.dot(X, pesos) + bias
    return sigmoid(z)

# Cálculo do erro
def calcular_erro(saida_predita, y_real):
    erro = y_real - saida_predita
    return erro

# Backpropagation
def backpropagation(X, y, saida_predita, pesos, bias, taxa_aprendizado):
    erro = calcular_erro(saida_predita, y)
    
    ajuste = erro * sigmoid_derivative(saida_predita)
    pesos += np.dot(X.T, ajuste) * taxa_aprendizado
    bias += np.sum(ajuste) * taxa_aprendizado
    
    return pesos, bias

# Função para treinar a rede
def treinar_rede(X, y, pesos, bias, epocas, taxa_aprendizado):
    for i in range(epocas):
        saida_predita = forward(X, pesos, bias)
        pesos, bias = backpropagation(X, y, saida_predita, pesos, bias, taxa_aprendizado)
        
        if i % 1000 == 0:
            erro_total = np.mean(np.square(y - saida_predita))
            print(f"Erro na época {i}: {erro_total}")
    
    return pesos, bias

# Inicialização dos dados e parâmetros
X = np.array([[0, 0, 1], 
              [0, 1, 1], 
              [1, 0, 1], 
              [1, 1, 1]])

y = np.array([[0], [1], [1], [0]])

# Inicializando pesos e bias
np.random.seed(1)
pesos = np.random.rand(3, 1)
bias = np.random.rand(1)

# Definição de hiperparâmetros
epocas = 10000
taxa_aprendizado = 0.1

# Treinando a rede
pesos_treinados, bias_treinado = treinar_rede(X, y, pesos, bias, epocas, taxa_aprendizado)

# Testando a rede após o treinamento
saida_predita = forward(X, pesos_treinados, bias_treinado)
print("Saída prevista após o treinamento:")
print(saida_predita)
