Vamos começar com o Multinomial Naive Bayes (MultinomialNB), um algoritmo de Machine Learning bastante utilizado para classificação de texto.

A ideia é simples:
1. Recebe as palavras transformadas em números.
2. Recebe a classificação correta de cada frase.
3. Aprende a relação entre as palavras e as classificações.
4. Consegue prever a classificação de frases novas.

Por exemplo:
Texto	                                Classificação
Adoro Python	                            positivo
Python é fantástico	                        positivo
Odeio este programa                     	negativo
Este projeto é frustrante               	negativo

O modelo vai aprender padrões a partir destes exemplos.

## O train usa o MultinomialNB para com base no training.csv que treina o modelo, o test.csv para testar se o modelo gera as os positivo/negativo esperado.

////


## O que é o Naive Bayes?
O Naive Bayes é um algoritmo de Machine Learning utilizado para classificação.
o Naive Bayes, que é o algoritmo responsável por classificar as frases como positivas ou negativas.
Vai aprender a identificar padrões nas palavras associadas a sentimentos positivos e negativos.

- Dataset
Dados de treino e teste em CSV.
Concluído

- Preprocessing
Limpeza de texto e normalização.
Concluído


- TF-IDF
Conversão de texto em valores numéricos.
Concluído

- Naive Bayes
Treino e classificação de texto.
Concluído

- Predict Proba
Probabilidades estimadas das classificações.

## N-grams
O conceito chama-se N-grams e é bastante utilizado em Natural Language Processing (NLP). Bigrams — combinações de duas palavras

## F1-score:

Precision: das previsões positivas, quantas estavam corretas?
Recall: dos exemplos realmente positivos, quantos conseguimos encontrar?
F1-score: equilíbrio entre Precision e Recall.


## metricas iniciais caltuladas:

Métrica	    Resultado	Interpretação
Accuracy	68,17%	    Acerta aproximadamente 68 em cada 100 exemplos.
Precision	69,99%	    As previsões têm uma precisão média de cerca de 70%.
Recall	    58,03%	    Identifica aproximadamente 58% dos exemplos de cada classe.
F1-score	55,53%	    O equilíbrio entre Precision e Recall ainda é relativamente baixo.

======================================

   ## RafiAI - Grid Search

Results:
Alpha: 0.001 | F1-score: 61.24% | Standard Deviation: 7.44%
Alpha: 0.005 | F1-score: 61.72% | Standard Deviation: 7.36%
Alpha: 0.01 | F1-score: 62.72% | Standard Deviation: 8.43%
Alpha: 0.02 | F1-score: 63.09% | Standard Deviation: 7.05%
Alpha: 0.03 | F1-score: 62.27% | Standard Deviation: 5.90%
Alpha: 0.05 | F1-score: 61.40% | Standard Deviation: 4.99%

Best configuration:
Alpha: 0.02
Best F1-score: 63.09%

### O que é um modelo treinado (trained model)?

É o resultado de um algoritmo que aprendeu padrões a partir dos dados de treino. No teu caso, aprendeu associações entre palavras e sentimentos positivos ou negativos.









========================================================================

´´´´´´´´´´´´´´´´´´´´´´
================================
          RafiAI
================================

Select a mode:
1. Development Mode
2. Usage Mode
0. Exit

Option:
´´´´´´´´´´´´´´´´´´´´´´

Escolhe 1 para treinar e avaliar o modelo.
Quando terminar, volta ao menu.
Escolhe 2 para carregar o modelo e testar frases.
Escreve I love Python.
Escreve exit para sair do modo de utilização.


-----------------------------------


Antes:
Criar Naive Bayes
      |
Treinar modelo
      |
Guardar modelo
      |
Executar Grid Search


Agora:
Carregar dados
      |
Executar avaliações
      |
Grid Search
      |
Selecionar melhor configuração
      |
Treinar melhor modelo com todos os dados
      |
Guardar modelo e Vectorizer
      |
Avaliar modelo