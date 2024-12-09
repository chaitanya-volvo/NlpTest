from nltk.corpus import wordnet as wn
import nltk
from nltk.corpus import sentiwordnet as swn

from nltk import WordNetLemmatizer

import spacy
import pandas as pd

print(wn.synsets('Motorcar'))
print(wn.synset('car.n.01').lemma_names())
print(wn.synset('car.n.01').definition())

cat = (swn.senti_synset('cat.n.03'))
print(cat.pos_score())

print(cat.neg_score())
print(cat.obj_score())

#print(cat.unicode_repr())

nlp = spacy.load('en_core_web_sm')

review = "Columbia University was founded in 1754 as King's College by royal charter of King George II of England. It is the oldest institution of higher learning in the state of New York and the fifth oldest in the United States. Controversy preceded the founding of the College, with various groups competing to determine its location and religious affiliation. Advocates of New York City met with success on the first point, while the Anglicans prevailed on the latter. However, all constituencies agreed to commit themselves to principles of religious liberty in establishing the policies of the College. In July 1754, Samuel Johnson held the first classes in a new schoolhouse adjoining Trinity Church, located on what is now lower Broadway in Manhattan. There were eight students in the class. At King's College, the future leaders of colonial society could receive an education designed to 'enlarge the Mind, improve the Understanding, polish the whole Man, and qualify them to support the brightest Characters in all the elevated stations in life.'' One early manifestation of the institution's lofty goals was the establishment in 1767 of the first American medical school to grant the M.D. degree."

doc = nlp(review)

sentences = [sentence.orth_ for sentence in doc.sents]

print("There were {} Sentences found.".format(len(sentences)))

nounphrases = [[np.orth_, np.root.head.orth_] for np in doc.noun_chunks]
print("There were {} noun phrases found.".format(len(nounphrases)))

entities = list(doc.ents)
print("There were {} entities found.".format(len(entities)))

orgs_and_people = [entity.orth_ for entity in entities if entity.label_ in ['ORG','PERSON']]

df = pd.DataFrame(orgs_and_people)
print(df.head())
