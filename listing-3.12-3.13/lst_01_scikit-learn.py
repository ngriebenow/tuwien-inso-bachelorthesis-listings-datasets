from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np

categories = ['comp.graphics', 'rec.sport.baseball', 'sci.med', 'talk.politics.guns']

twenty_train = fetch_20newsgroups(subset='train',
     categories=categories, shuffle=True, random_state=42)

# transformers
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# estimator
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

demo_docs = ['Man shot dead', 'DirectX on the GPU is fast', 'Season of MLB started', 'no more blood donations']
demo_counts = count_vect.transform(demo_docs)
demo_tfidf = tfidf_transformer.transform(demo_counts)

# predictor
predicted = clf.predict(demo_tfidf)

for doc, category in zip(demo_docs, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))






pipe = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
    ])

pipe.fit(twenty_train.data, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test',
    categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = pipe.predict(docs_test)
print(np.mean(predicted == twenty_test.target))
