# https://www.nltk.org/api/nltk.lm.html
# https://towardsdatascience.com/understanding-word-n-grams-and-n-gram-probability-in-natural-language-processing-9d9eef0fa058
# https://towardsdatascience.com/text-generation-using-n-gram-model-8d12d9802aa0

import nltk
from nltk import ngrams
from nltk.lm.preprocessing import pad_both_ends
from nltk.util import bigrams
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE

nltk.download('punkt')

text = [
    ['there', 'is', 'a', 'big', 'house'],
    ['i', 'buy', 'a', 'house'],
    ['they', 'buy', 'the', 'new', 'house'],
]

text_padded = list()
for t in text:
    text_padded.append(list(pad_both_ends(t, n=2)))

# text2 = list(bigrams(pad_both_ends(text, n=4)))
print(text_padded)

train, vocab = padded_everygram_pipeline(2, text)

# train_list = [x for x in train]
# vocab_list = [x for x in vocab]

lm = MLE(2)
lm.fit(train, vocab)
# print(lm.vocab.lookup(text[0]))
# print(lm.vocab.lookup(["aliens", "from", "Mars"]))

# print(lm.counts)
# print(lm.counts['a'])
# print(lm.counts[['a']]['buy'])
# print(lm.score("a"))
# print(lm.score("a", ["buy"])) # 0.5 as in presentation

test = [
    ['they', 'buy', 'a', 'big', 'house'],
]

test_padded = list()
for t in test:
    test_padded.append(list(pad_both_ends(t, n=2)))

bigrams_test = list(bigrams(test[0], pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
ngrams_test = list(ngrams(test[0], n=2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
print(ngrams_test)

P_final = 1
for b in ngrams_test:
    a = lm.score(b[1], [b[0]])
    print(a, b)
    P_final = P_final * round(a, 2)

print(f'P(S)={P_final}')
