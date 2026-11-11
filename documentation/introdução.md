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