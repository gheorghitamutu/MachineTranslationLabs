# https://spacy.io/
# https://github.com/shayanhooshmand/shakespeare_generator
# https://shayanhooshmand.medium.com/can-we-use-part-of-speech-tags-to-improve-the-n-gram-language-model-3ef7e0b465d3

import nltk
from collections import defaultdict, Counter
import math
import random
import spacy
from spacy.lang.ro.examples import sentences


def pre_process(corpus):
    corpus_tagged = list()
    for s in corpus:
        tt = [(t.text, t.pos_) for t in s]
        corpus_tagged.append(tt)

    return corpus_tagged


def get_ngrams(line, n):
    """
    returns list of ngrams given a list of tokens and ngram size n
    """
    bookended = [('START', 'START')] * max(1, n - 1) + line + [('STOP', 'STOP')]
    return [tuple(bookended[i:i + n]) for i in range(len(bookended) - (n - 1))]


def count_words(corpus):
    """
    returns a dictionary of dictionaries that word from tag emissions
    """
    word_counts = defaultdict(lambda: defaultdict(int))
    for line in corpus:
        for word, tag in line:
            word_counts[tag][word] += 1

    return word_counts


class Ngram_model:

    def __init__(self, corpus, n=3, smoother='interpolate'):

        if smoother != 'interpolate' and smoother != 'backoff':
            print("ERR: invalid smoothing method ", smoother)
            exit(1)

        self.n = n
        self.smoother = smoother

        # initialize for backoff calculations
        self.alpha = {}
        self.normalizer = {}
        self.tag_lexicon = set()

        self.corpus = pre_process(corpus)

        # initialize total number of tokens
        self.M = 0
        self.ngram_counts = [self.count_ngrams(i) for i in range(1, self.n + 1)]

        # count up word probabilites
        self.word_counts = count_words(self.corpus)

        # initialize beta for backoff prob
        self.beta = 0.2

        return

    def get_ngrams(self, line, n):
        """
        specifically gets tags or words for ngrams
        depending on the model instance
        """
        ngrams = []
        for tuple_ngram in get_ngrams(line, n):
            ngrams.append(tuple([pair[1] for pair in tuple_ngram]))
        return ngrams

    def count_ngrams(self, n):
        """
        counts up the ngrams for a given n in the model's corpus
        returns a dictionary of the counts
        """
        ngram_counts = defaultdict(int)
        START = tuple(['START'] * n)
        for line in self.corpus:
            for ngram in self.get_ngrams(line, n):
                ngram_counts[ngram] += 1
                # add ngram full of starts for each beginning of sentence
                # for when we calculate likelihoods later
                if n > 1:
                    if ngram[0] == 'START' and ngram[1] != ngram[0]:
                        ngram_counts[START] += 1
                # if we are counting unigrams then also use this loop to calculate
                # count up total number of tokens
                elif n == 1:
                    self.M += 1
                    self.tag_lexicon.add(ngram[0])

        return ngram_counts

    def count(self, ng):
        """
        returns the raw count of an ngram
        """
        if len(ng) > self.n:
            print("ERR: ngram length out of range for model")
            exit(1)

        return self.ngram_counts[len(ng) - 1][ng]

    def raw_ngram_prob(self, ng):
        """
        gives raw mle probability of an ngram
        """

        n = len(ng)
        ng_count = self.ngram_counts[n - 1][ng]
        if ng_count == 0: return 0

        given = ng[:-1]
        normalizer = self.ngram_counts[len(given) - 1][given] if len(given) > 0 else self.M
        return ng_count / normalizer

    def interpolated_prob(self, ng):
        """
        computationally cheap smoothed probability
        using linear interpolation
        """
        n = len(ng)
        likelihood = 0
        for i in range(n):
            likelihood += self.raw_ngram_prob(ng[:i + 1])

        return likelihood / n

    def get_alpha(self, ng, beta):
        """
        helper for backoff probability
        alpha is leftover probability mass for an
        ngram
        """
        if ng in self.alpha:
            return self.alpha[ng]

        # if this context is unseen
        if self.count(ng) == 0: return 0

        # count how many tokens copmlete this n+1_gram
        num_grams = 0
        curr_lexicon = self.tag_lexicon
        for token in curr_lexicon:
            ng_plus1 = ng + (token,)
            if self.count(ng_plus1) > 0:
                num_grams += 1

        return (num_grams * beta) / self.count(ng)

    def backoff_prob(self, ngram, beta=-1):
        """
        returns the katz' backoff probability of a given
        ngram using beta as the discount
        """

        # if we are not passed a beta
        if beta == -1: beta = self.beta

        def get_normalizer(ng):
            # avoid recalculating
            if ng in self.normalizer:
                return self.normalizer[ng]

            normalizer = 0
            # check to see if the larger n+1_gram has count of 0
            # if so, then add the probability of the ngram starting
            # from the first index of n+1_gram to the end to the normalizer
            curr_lexicon = self.tag_lexicon
            for token in curr_lexicon:
                if token == 'START': continue
                ng_plus1 = ng + (token,)
                if self.count(ng_plus1) == 0:
                    normalizer += self.backoff_prob(ng_plus1[1:], beta)

            self.normalizer[ng] = normalizer
            return self.normalizer[ng]

        # base case 1 : unigram
        # simple MLE
        if len(ngram) == 1:
            return self.count(ngram) / self.M

        ngram_count = self.count(ngram)
        # base case 2 : seen ngram
        # discounted MLE
        if ngram_count > 0:
            return (ngram_count - beta) / self.count(ngram[:-1])

        # otherwise, we have to recurse
        alpha = self.get_alpha(ngram[:-1], beta)
        # in case this context is totally unseen
        if alpha == 0:
            return self.backoff_prob(ngram[1:], beta)

        normalizer = get_normalizer(ngram[:-1])

        return alpha * self.backoff_prob(ngram[1:], beta) / normalizer

    def word_prob(self, word, tag):
        """
        probability of a word given a tag using
        MLE for instances of the model that use tags for ngrams
        """
        if word == 'START' or word == 'STOP':
            return 1

        # do laplacian smoothing
        alpha = 0.1
        tag_count = self.ngram_counts[0][(tag,)]
        voc_size = len(self.word_counts[tag])
        return (self.word_counts[tag][word] + alpha) / (tag_count + voc_size * alpha)

    def sentence_logprob(self, sentence, beta):
        """
        returns the log prob of a given sentence
        uses linear interpolation or smoothing as per the model instance
        """
        # sum ngram probs
        total = 0
        for ngram in self.get_ngrams(sentence, self.n):
            if self.smoother == 'interpolate':
                total += math.log2(self.interpolated_prob(ngram))
            else:
                total += math.log2(self.backoff_prob(ngram, beta))

        # sum word probs, too, if we use POS tags for ngrams
        total += sum(math.log2(self.word_prob(word, tag)) for word, tag in sentence)
        return total

    def perplexity(self, corpus, beta=-1):
        """
        calculate the log probability of an entire corpus
        assumes corpus is passed formatted or as line by line
        strings
        """
        corpus = pre_process(corpus)

        # if we are not passed a beta
        if beta == -1: beta = self.beta

        l, M = 0, 0
        for sentence in corpus:
            # every word and stop token counted once, and
            # if unigram model, then start token also counted
            # individually
            M += len(sentence) + 1 + (self.n == 1)
            M += len(sentence)
            l += self.sentence_logprob(sentence, beta)

        return 2 ** -(l / M)


nlp = spacy.load("ro_core_news_lg")

doc_sentences = list()
for sentence in sentences:
    doc_sentences.append(nlp(sentence))

doc_test_sentences = [nlp('Peștele maro este un animal vertebrat')]

print(doc_test_sentences[0].text)
print(f"{'text':{10}} {'POS':{6}} {'TAG':{10}} {'Dep':{6}} {'POS explained':{20}}")
print('-------------------------------------------------')
for token in doc_test_sentences[0]:
    print(f'{token.text:{10}} {token.pos_:{6}} {token.tag_:{10}} {token.dep_:{6}} {spacy.explain(token.pos_):{20}}')
print()

train = doc_sentences
test = doc_test_sentences
order = 3

model = Ngram_model(train, order)
perplexity = model.perplexity(test)
print('NGram #{}\nPerplexity: {}\n'.format(order, perplexity))

ss = pre_process(test)

ngrams = model.get_ngrams(ss[0], order)
for ngram in ngrams:
    p = model.backoff_prob(ngram)
    print('{} => {}'.format(ngram, p))
print()

for token in doc_test_sentences[0]:
    p = model.word_prob(token.text, token.pos_)
    print('{} => {} => {}'.format(token.text, token.pos_, p))
print()
