import spacy

from spacy.lang.de.stop_words import STOP_WORDS
stopwords = list(STOP_WORDS)
print(stopwords[:10])

nlp = spacy.load('de')
doc = nlp("Konzernchefs lehnen den Milliardär als US-Präsidenten ab.")

for token in doc:
    print(f"{token.text}\t{token.lemma_}\t{token.pos_}\t{token.tag_} \
        \t{token.dep_}\t{token.is_stop}")


doc = nlp("Der Präsident der Vereinigten Staaten, Richard Nixon, unterschrieb in Washington D. C. seine Rücktrittserklärung.")

doc = nlp("Lehman Brothers, eine ehemalige Investmentbank der Vereinigten Staaten, beantragte am 15. September 2008 die Insolvenz in New York.")

for chunk in doc.noun_chunks:
    print(f"{chunk.text}\t{chunk.root.text}\t{chunk.root.dep_}\t{chunk.root.head.text}")

for ent in doc.ents:
    print(f"{ent.text}\t{ent.start_char}\t{ent.end_char}\t{ent.label_}")