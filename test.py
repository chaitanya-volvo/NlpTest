import nltk

def format_sentance(sent):
    return ({word: True for word in nltk.word_tokenize(sent)})

result = format_sentance ("Life is beautiful so Enjoy everymoment you have.")
print(result)

#print(nltk.word_tokenize("Life is beautiful so Enjoy everymoment you have."))
#dict = {}
#print({word: True for word in nltk.word_tokenize(text)})
    #dict.items. .items([{word} : True])

text = nltk.word_tokenize("Life is beautiful so Enjoy everymoment you have.")


print(nltk.pos_tag(text))


raw = "OMG, Natural Language Processing is SO cool and I'm really enjoying this workshop!"

tokens = [i.lower() for i in nltk.word_tokenize(raw)]

print(tokens)

porter = nltk.PorterStemmer()
stem = [porter.stem(i) for i in tokens]
print(stem)

text = "Women in technology are amazing at coding"

#print(text.split())

print("***********")

lemma = nltk.WordNetLemmatizer()
 
ex = [word.lower() for word in text.split()]

print([lemma.lemmatize(i) for i in ex])


