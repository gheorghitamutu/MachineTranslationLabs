# http://emjotde.github.io/publications/pdf/mjd2011siis.pdf


import warnings
from collections import defaultdict

from nltk import IBMModel1, IBMModel2, IBMModel3, IBMModel4, IBMModel5
from nltk.translate import AlignedSent, Alignment, IBMModel
from nltk.translate.ibm_model import Counts
from nltk.translate.phrase_based import phrase_extraction
from nltk.translate.ibm_model import AlignmentInfo

import numpy
from itertools import product


class MyIBMModel(IBMModel1):

    @staticmethod
    def alignment_to_info(aligned_sent):
        source_sent = tuple([None] + aligned_sent.mots)
        target_sent = tuple([None] + aligned_sent.words)
        alignment = [0 for _ in range(len(target_sent))]
        for t, s in dict(aligned_sent.alignment).items():
            alignment[t + 1] = s + 1
        return AlignmentInfo(tuple(alignment), source_sent, target_sent, None)

    @staticmethod
    def create_inverse_table(model):
        d = dict()
        for t, sd in model.translation_table.items():
            for s, p in sd.items():
                d.setdefault(s, {})[t] = p
        model.source_target_table = d

    def most_probable_words(self, word, source=True, k=5):
        if source:
            translations = self.source_target_table.get(word, {})
        else:
            translations = self.translation_table.get(word, {})
        translations = {k: v for k, v in translations.items() if k and v}
        return sorted(translations.items(), key=lambda kv: (-kv[1], kv[0]))[:k]

    def real_prob_t_a_given_s(self, alignment_info):
        return self.prob_t_a_given_s(alignment_info) / pow(
            len(alignment_info.src_sentence), len(alignment_info.trg_sentence) - 1)

    def prob_t_best_a_given_s(self, aligned_sent):
        return self.real_prob_t_a_given_s(self.alignment_to_info(aligned_sent))

    def prob_t_given_s(self, aligned_sent):
        ali = self.alignment_to_info(aligned_sent)
        p = 0
        for align in product(range(len(ali.src_sentence)), repeat=len(ali.trg_sentence) - 1):
            ali.alignment = tuple([0] + list(align))
            p += self.real_prob_t_a_given_s(ali)
        return p

    @staticmethod
    def get_consistent_phrases(aligned_sentences):
        aligned_sentences_filtered = list()
        for s in aligned_sentences:
            alignment = s.alignment
            found_none = False
            for a in alignment:
                if a[0] is None or a[1] is None:
                    found_none = True
                    break

            if found_none is not True:
                aligned_sentences_filtered.append(s)

        consistent_phrases = list()
        for s in aligned_sentences_filtered:
            phrases = phrase_extraction(srctext=' '.join(s.words), trgtext=' '.join(s.mots), alignment=s.alignment)
            consistent_phrases.extend(phrases)

        return consistent_phrases


testext = [
    # AlignedSent(['klein', 'ist', 'das', 'haus'], ['the', 'house', 'is', 'small']),
    # AlignedSent(['das', 'haus', 'ist', 'ja', 'groß'], ['the', 'house', 'is', 'big']),
    # AlignedSent(['das', 'buch', 'ist', 'ja', 'klein'], ['the', 'book', 'is', 'small']),
    AlignedSent(['das', 'haus'], ['the', 'house']),
    AlignedSent(['das', 'buch'], ['the', 'book']),
    AlignedSent(['ein', 'buch'], ['a', 'book'])
]

# ibm1 = MyIBMModel(testext, 5)
#
# # Tests for Exercise 2.4
# testalis = [ibm2.alignment_to_info(s) for s in testext]
#
# assert testalis[2].alignment == (0, 1, 2, 3, 3, 4)
# assert testalis[5].alignment == (0, 1, 2)
#
# # Tests for Exercise 2.5
# assert numpy.allclose(ibm2.real_prob_t_a_given_s(testalis[5]), 0.08283000979778607)
# assert numpy.allclose(ibm2.real_prob_t_a_given_s(testalis[0]), 0.00018256804431244556)
#
# # Tests for Exercise 2.6
# assert numpy.allclose(ibm2.prob_t_best_a_given_s(testext[4]), 0.059443309368677)
# assert numpy.allclose(ibm2.prob_t_best_a_given_s(testext[2]), 1.3593610057711997e-05)

# # Tests for Exercise 2.7
# assert numpy.allclose(ibm2.prob_t_given_s(testext[4]), 0.13718805082588842)
# assert numpy.allclose(ibm2.prob_t_given_s(testext[2]), 0.0001809283308942621)


english_tokenized_sentences = list()
french_tokenized_sentences = list()

with open('fr-en/europarl-v7.fr-en.en', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        tokens = line.split()
        english_tokenized_sentences.append(tokens)

with open('fr-en/europarl-v7.fr-en.fr', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        tokens = line.split()
        french_tokenized_sentences.append(tokens)

aligned_sentences = list()

for (a, b) in zip(english_tokenized_sentences, french_tokenized_sentences):
    aligned_sentence = AlignedSent(a, b)
    aligned_sentences.append(aligned_sentence)

ibm_model = MyIBMModel(aligned_sentences, 5)
# ibm1.train(aligned_sentences)
# ibm1.create_inverse_table(ibm1)

consistent_phrases = ibm_model.get_consistent_phrases(aligned_sentences)
print(consistent_phrases)

# aligned_sentences_info = [ibm1.sample(x) for x in testext]
# aligned_sentences_info = [ibm1.alignment_to_info(x) for x in aligned_sentences]

# alignment = list()
# consistent_phrases = phrase_extraction(english_tokenized_sentences, french_tokenized_sentences, alignment)

# a = ibm1.alignment_table

# print(aligned_sentences_info[0])
# for s in aligned_sentences_info[0]:
#     for ss in s:
#         c = phrase_extraction(srctext=ss.src_sentence, trgtext=ss.trg_sentence, alignment=ss.alignment)
#         print(c)

bitext = [AlignedSent(['le', 'chien'], ['the', 'dog']), AlignedSent(['le', 'chat'], ['the', 'cat'])]
ibm1 = IBMModel1(bitext, 5)
ibm1.train(bitext)

print('le    -> the => {}'.format(ibm1.translation_table['le']['the']))
print('le    -> dog => {}'.format(ibm1.translation_table['le']['dog']))
print('le    -> cat => {}'.format(ibm1.translation_table['le']['cat']))

print('chien -> the => {}'.format(ibm1.translation_table['chien']['the']))
print('chien -> dog => {}'.format(ibm1.translation_table['chien']['dog']))
print('chien -> cat => {}'.format(ibm1.translation_table['chien']['cat']))

print('chat  -> the => {}'.format(ibm1.translation_table['chat']['the']))
print('chat  -> dog => {}'.format(ibm1.translation_table['chat']['dog']))
print('chat  -> cat => {}'.format(ibm1.translation_table['chat']['cat']))

print()

bitext2 = [AlignedSent(['le', 'chien'], ['the', 'dog']), AlignedSent(['le', 'chat'], ['the', 'cat'])]
ibm1.align_all(bitext2)
print(bitext)
print(bitext2)

