'''
Main app
Benodigdheden: NLTK
'''
from nltk.tokenize import sent_tokenize, word_tokenize

example_input_string = """Dit is een example input string die bedoeld is om
te testen of de werking van de tokenization methode werkt. Dit werkt op basis
van zinnen, maar ook individuele worden.
"""
print(sent_tokenize(example_input_string))
print(word_tokenize(example_input_string))



