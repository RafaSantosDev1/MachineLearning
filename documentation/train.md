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

