import random
import spacy
from spacy.util import minibatch, compounding

TRAIN_DATA = [
    ("Wer ist Sahar Ghar?", {"entities": [(8, 18, "PERSON")]}),
    ("Ich bin gerne in London und Berlin unterwegs.", {"entities": [(17, 23, "LOC"), (28, 34, "LOC")]}),
    ("Apple und Microsoft sind zwei IT-Firmen.", {"entities": [(0, 5, "ORG"), (10, 19, "ORG")]}),
    ("Der Eiffelturm steht in Paris.", {"entities": [(4, 14, "OBJ"), (24, 29, "LOC")]})
]
 
nlp = spacy.blank("de")
ner = nlp.create_pipe("ner")
nlp.add_pipe(ner, last=True)

for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):

    nlp.begin_training()
    
    for itn in range(200):
        random.shuffle(TRAIN_DATA)
        losses = {}

        batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(
                texts,
                annotations,
                drop=0.5,
                losses=losses,
            )

doc = nlp("Sahar Welora arbeitet bei Micosoft in Berlin, wohnt aber in Hamburg.")
print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
print("Tokens", [(t.text, t.ent_type_) for t in doc])