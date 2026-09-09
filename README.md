# Projeto 1 – Rede MLP

Projeto desenvolvido para a Unidade Curricular de **Redes Neurais**, com o objetivo de estudar o comportamento de **Redes Neurais Multicamadas (MLP – Multilayer Perceptron)** em problemas de **classificação** e **regressão**, utilizando **Python** e **PyTorch**.

## Objetivos

O projeto contempla dois problemas distintos:

### 1. Classificação

Para o problema de classificação será utilizado o dataset **Fashion-MNIST**, composto por imagens de peças de vestuário distribuídas em diferentes classes.

O objetivo é desenvolver uma rede MLP capaz de classificar corretamente as imagens do conjunto de dados.

### 2. Regressão

Para o problema de regressão será criada uma função sintética não trivial de uma variável independente:

$$
y = f(x)
$$

A função deverá apresentar características que tornem o problema de aproximação mais complexo, podendo incluir:

* Não linearidades;
* Descontinuidades;
* Ruído;
* Diferentes comportamentos ao longo do domínio.

A rede MLP será utilizada para aproximar essa função a partir dos dados gerados.

---

## Divisão dos Dados

Para ambos os problemas, os dados serão separados em três conjuntos:

* **Treinamento (Train)** – utilizado para ajustar os pesos da rede;
* **Validação (Validation)** – utilizado para acompanhar o desempenho durante o desenvolvimento do modelo;
* **Teste (Test)** – utilizado somente para a avaliação final do modelo.

---

## Modelo Baseline

Para cada problema será desenvolvido empiricamente um modelo **baseline (vanilla)**, correspondente à melhor arquitetura MLP básica encontrada durante os experimentos.

O baseline deverá utilizar exclusivamente:

* Rede neural **MLP (Multilayer Perceptron)**;
* Otimizador **SGD (Stochastic Gradient Descent)**.

O modelo baseline **não deverá utilizar**:

* Momentum;
* Regularização L1;
* Regularização L2;
* Dropout;
* Otimizadores adaptativos, como Adam, AdamW, RMSprop etc.

Durante o desenvolvimento do baseline serão avaliadas diferentes configurações, como:

* Número de camadas escondidas;
* Número de neurônios por camada;
* Funções de ativação;
* Learning rate;
* Número de épocas;
* Batch size.

A arquitetura selecionada como baseline será aquela que apresentar o melhor desempenho nos experimentos realizados.

---

## Estudos de Ablação

Após a definição do baseline, serão realizados experimentos adicionando individualmente diferentes técnicas ao modelo:

* **Regularização L1**;
* **Regularização L2**;
* **Dropout**;
* **Momentum**.

A arquitetura da rede definida no baseline deverá permanecer **inalterada** durante esses experimentos.

O objetivo é avaliar o impacto de cada técnica sobre o treinamento, capacidade de generalização e desempenho final da rede.

---

## Avaliação dos Modelos

Durante o treinamento serão gerados gráficos mostrando a evolução das métricas nos conjuntos de **treinamento e validação**.

### Métricas de Classificação

Entre as métricas consideradas estão:

* Acurácia (Accuracy);
* Precisão (Precision);
* Recall;
* F1-score;
* Matriz de confusão.

### Métricas de Regressão

Para o problema de regressão serão consideradas métricas como:

* MAE – Mean Absolute Error;
* MSE – Mean Squared Error;
* RMSE – Root Mean Squared Error;
* R² – Coeficiente de determinação.

Também serão analisadas graficamente as predições da rede em comparação com a função original.

---

## Análises

Todos os experimentos realizados durante o desenvolvimento do baseline serão documentados, incluindo:

* Configurações utilizadas;
* Arquiteturas testadas;
* Hiperparâmetros;
* Evolução do treinamento;
* Comparação entre treino e validação;
* Resultados no conjunto de teste;
* Comparação entre baseline e modelos com L1, L2, Dropout e Momentum;
* Discussão dos resultados obtidos.

---

## Tecnologias

* Python
* PyTorch
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Torchvision

---

## Entregáveis

O projeto será composto por:

* **Código-fonte em Python utilizando PyTorch**;
* **Relatório** contendo a metodologia, experimentos, resultados e análises;
* Gráficos de treinamento e validação;
* Comparação entre os modelos desenvolvidos.
