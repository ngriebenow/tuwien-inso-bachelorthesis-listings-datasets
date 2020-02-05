from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer("german")
stemmer.stem("Dienstleistungen")

from nltk.tokenize import word_tokenize
word_tokenize("Was tun, sprach Zeus, die Götter sind besoffen und bekotzen den Olymp.")


