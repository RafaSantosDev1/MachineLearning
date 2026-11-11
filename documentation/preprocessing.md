O CountVectorizer do Scikit-learn organiza as palavras do vocabulário por ordem alfabética, não pela ordem em que aparecem nas frases.

No teu exemplo:

texts = [
    "Adoro Python",
    "Adoro Java",
    "Python olas"
]

ou seja, por ordem alfabetica: ['adoro', 'java', 'olas', 'python']


['adoro', 'java', 'olas', 'python']

Frase	      adoro	java   olas	 python
Adoro Python	1	 0	    0	   1
Adoro Java	    1	 1	    0	   0
Python olas	    0	 0      1      1

Assim verifica se cada palavra existe em cada uma das frases.

Training dataset (palavras):
Number of examples: 16

Testing dataset (palavras):
Number of examples: 6

- CountVectorizer: conta quantas vezes aparece cada palavra.
- TfidfVectorizer: calcula um peso para cada palavra, tendo em conta a sua frequência e a sua distribuição nos documentos.

Mas atenção a três coisas:
- 0.000: a palavra não aparece na frase.
- Valor baixo: a palavra tem menor peso relativo.
- Valor elevado: a palavra tem maior peso relativo.

O valor TF-IDF aumenta ou diminui principalmente devido a dois fatores: a frequência da palavra na frase e a frequência dessa palavra no conjunto de frases.

No TF-IDF:
- Aparece em muitas frases diferentes: o IDF diminui e a palavra tende a ter um peso menor.
- Aparece em poucas frases diferentes: o IDF aumenta e a palavra tende a ter um peso maior.
- Aparece muitas vezes na mesma frase: o TF aumenta.


F-IDF é uma técnica utilizada em Machine Learning e Processamento de Linguagem Natural (NLP) para transformar palavras em valores numéricos, atribuindo-lhes um peso que representa a sua importância num determinado texto.

TF-IDF significa:

Term Frequency – Inverse Document Frequency

É composto por duas partes:
1. TF — Term Frequency
Mede a frequência com que uma palavra aparece num determinado texto.

Cross-Validation (Validação Cruzada):Cross-Validation é uma técnica que permite avaliar um modelo através de várias divisões dos dados.
StratifiedKFold do Scikit-learn. Este método mantém aproximadamente a mesma proporção de exemplos positivos e negativos em cada grupo.
