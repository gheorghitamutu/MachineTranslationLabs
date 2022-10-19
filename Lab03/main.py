# https://web.stanford.edu/~jurafsky/slp3/slides/LM_4.pdf
# https://www.geeksforgeeks.org/n-gram-language-modelling-with-nltk

import nltk
from nltk import ngrams
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE, Laplace, Lidstone, StupidBackoff, WittenBellInterpolated, AbsoluteDiscountingInterpolated, \
    KneserNeyInterpolated
from nltk.corpus import stopwords

import string
from math import prod


class RomanianTrainingModel:
    def __init__(self, context, order, model):
        nltk.download('punkt')

        self.context = context
        self.order = order
        self.lm = model(order=self.order)
        self.train = None
        self.vocab = None

    def train_model(self):
        self.train, self.vocab = padded_everygram_pipeline(self.order, self.context)
        self.lm.fit(self.train, self.vocab)

    def data_test(self, data):
        ngrams_test = list(ngrams(data[0], n=self.order))
        print(f'NGrams from test data:\n{ngrams_test}')

        print('Score for each combination:')
        for b in ngrams_test:
            left, right = b[-1], b[0:-1]

            a = self.lm.score(left, right)
            print(f'P{tuple(reversed(b))} = {a}')

        final = prod([self.lm.score(b[1], [b[0]]) for b in ngrams_test])
        print(f'P(S)={final}')

    def lookup_words_in_sentence(self, sentence):
        print(f'Lookup for sentence: {sentence}')
        for word in sentence:
            lookup = self.lm.vocab.lookup(word)
            print(f'{word} => {lookup}')

    @staticmethod
    def process_input_from_file(path):
        nltk.download('stopwords')

        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        stop_words = set(stopwords.words('romanian'))
        string.punctuation = string.punctuation + '"' + '"' + '-' + '''+''' + '—'

        tokenized_sentences = nltk.sent_tokenize(content)
        filtered_sentences = list()
        for each_sentence in tokenized_sentences:
            words = nltk.tokenize.word_tokenize(each_sentence)
            words_filtered = list()
            for word in words:
                if word not in string.punctuation:
                    if word[0] in string.punctuation:
                        word = word[1:]
                    words_filtered.append(word)

            filtered_sentences.append(words_filtered)

        return filtered_sentences


def example_01():
    order = 2
    context = [
        ['there', 'is', 'a', 'big', 'house'],
        ['i', 'buy', 'a', 'house'],
        ['they', 'buy', 'the', 'new', 'house'],
    ]

    test_data = [
        ['<s>', 'they', 'buy', 'a', 'red', 'house', '</s>'],
    ]

    rtm = RomanianTrainingModel(context, order, Laplace)
    rtm.train_model()
    rtm.lookup_words_in_sentence(test_data[0])
    rtm.data_test(test_data)

    '''
        Lookup for sentence: ['<s>', 'they', 'buy', 'a', 'red', 'house', '</s>']
        <s> => <s>
        they => they
        buy => buy
        a => a
        red => <UNK>
        house => house
        </s> => </s>
        NGrams from test data:
        [('<s>', 'they'), ('they', 'buy'), ('buy', 'a'), ('a', 'red'), ('red', 'house'), ('house', '</s>')]
        Score for each combination:
        P('they', '<s>') = 0.125
        P('buy', 'they') = 0.14285714285714285
        P('a', 'buy') = 0.13333333333333333
        P('red', 'a') = 0.06666666666666667
        P('house', 'red') = 0.07692307692307693
        P('</s>', 'house') = 0.25
        P(S)=3.052503052503052e-06
    '''


def example_02():
    order = 2
    context = [
        ['I', 'am', 'Sam'],
        ['Sam', 'I', 'am'],
        ['I', 'do', 'not', 'like', 'green', 'eggs', 'and', 'ham'],
    ]

    test_data = [
        ['<s>', 'I', 'am', 'Sam', '</s>'],
    ]

    rtm = RomanianTrainingModel(context, order, MLE)
    rtm.train_model()
    rtm.lookup_words_in_sentence(test_data[0])
    rtm.data_test(test_data)

    '''
        Lookup for sentence: ['<s>', 'I', 'am', 'Sam', '</s>']
        <s> => <s>
        I => I
        am => am
        Sam => Sam
        </s> => </s>
        NGrams from test data:
        [('<s>', 'I'), ('I', 'am'), ('am', 'Sam'), ('Sam', '</s>')]
        Score for each combination:
        P('I', '<s>') = 0.6666666666666666
        P('am', 'I') = 0.6666666666666666
        P('Sam', 'am') = 0.5
        P('</s>', 'Sam') = 0.5
        P(S)=0.1111111111111111
    '''


def example_03():
    order = 3
    context = RomanianTrainingModel.process_input_from_file('input.txt')

    test_data = [
        ['<s>', 'Peștele', 'maro', 'este', 'un', 'animal', 'vertebrat', '</s>'],
    ]

    rtm = RomanianTrainingModel(context, order, Laplace)
    rtm.train_model()
    rtm.lookup_words_in_sentence(test_data[0])
    rtm.data_test(test_data)

    '''
        Lookup for sentence: ['<s>', 'Peștele', 'maro', 'este', 'un', 'animal', 'vertebrat', '</s>']
        <s> => <s>
        Peștele => Peștele
        maro => <UNK>
        este => este
        un => un
        animal => animal
        vertebrat => vertebrat
        </s> => </s>
        NGrams from test data:
        [('<s>', 'Peștele', 'maro'), ('Peștele', 'maro', 'este'), ('maro', 'este', 'un'), ('este', 'un', 'animal'), ('un', 'animal', 'vertebrat'), ('animal', 'vertebrat', '</s>')]
        Score for each combination:
        P('maro', 'Peștele', '<s>') = 0.0013679890560875513
        P('este', 'maro', 'Peștele') = 0.0013698630136986301
        P('un', 'este', 'maro') = 0.0013698630136986301
        P('animal', 'un', 'este') = 0.0027359781121751026
        P('vertebrat', 'animal', 'un') = 0.0027359781121751026
        P('</s>', 'vertebrat', 'animal') = 0.001366120218579235
        P(S)=1.19743185081498e-16
    '''


if __name__ == '__main__':
    example_01()
    # example_02()
    # example_03()

