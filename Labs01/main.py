"""
Project files:
    - main.py => the script that processes data and does the translation
    - lexicon.txt => words mapping
    - rules.txt => rules mapping
    - input.txt => test data

The script assumes the following flow:
    parse_lexicon()    => maps lexicon from files to data structure
    parse_rules()      => maps rules from file to data structure
    input_to_tokens()  => english input mapped to tokens
    tokens_to_output() => tokens resulted from english sentences to French sentences

Functions can be divided into 2 parts:
    - creating the context/domain
        parse_lexicon()
        parse_rules()
    - applying constraints and rules on input (generating output)
        input_to_tokens()
        tokens_to_output()

No dependencies required. Tested using python 3.9.
"""


# Masc N (nouns)
# Fem N (nouns)
# V (verbs)
# DET (determiners)
# ADJ (adjectives)
# CONJ (conjunctions)
# PREP (prepositions)
# PNOUN (proper noun)

# data from lexicon.txt
lexicon = {
    'Masc N': {},
    'Fem N': {},
    'V': {},
    'DET': {},
    'ADJ': {},
    'CONJ': {},
    'PREP': {},
    'PNOUN': {},
}

# rules from rules.txt
rules = {
    'Rewritting rules': {},
    'POS identification rules': {}
}

# lists of lists containing either words from initial sentences or their tokens
input_words = list()
input_tokens = list()


def parse_lexicon():
    """
    Parsing lexicon file with specific syntax.
    It ignores empty lines and handles PNOUN special cases.
    Maps the data into 'lexicon' dictionary declared global above.
    :return:
    """

    with open('lexicon.txt', 'r') as l:
        chosen_key = None
        for line in l.readlines():

            line = line.strip()
            if line == '':
                continue

            skip_line = False
            for key in lexicon.keys():
                if line.startswith(key):
                    chosen_key = key
                    skip_line = True
                    break
            if skip_line or chosen_key is None:
                continue

            if '->' in line:  # map
                left, right = line.split('->')
                left = left.strip().lower()
                right = right.strip().lower()
                lexicon[chosen_key][left] = right
            else:  # proper noun
                lexicon[chosen_key][line] = line

    print(lexicon)


def parse_rules():
    """
    Parsing rules file with specific syntax.
    It ignores empty lines.
    Maps the data into 'rules' dictionary declared global above.
    It reads a line, splits it in half by '->' then each half is split into tokens using '+' as delimiter.
    :return:
    """

    rule_number = 0
    with open('rules.txt', 'r') as r:
        chosen_key = None
        for line in r.readlines():

            line = line.strip()
            if line == '':
                continue

            skip_line = False
            for key in rules.keys():
                if line.startswith(key):
                    chosen_key = key
                    skip_line = True
                    rule_number = 0
                    break
            if skip_line or chosen_key is None:
                continue

            left, right = line.split('->')

            tokens_left = left.split('+')
            tokens_left = [x.strip() for x in tokens_left]

            tokens_right = right.split('+')
            tokens_right = [x.strip() for x in tokens_right]

            rules[chosen_key][str(rule_number)] = [tokens_left, tokens_right]
            rule_number += 1

    print(rules)


def input_to_tokens():
    """
    Tokenizes input file 'input.txt'.
    It parses line by line and adds the data into 2 containers defined globally above: 'input_words' and 'input_tokens'.
    Does not apply the rules but it could if refactored.
    :return:
    """
    with open('input.txt', 'r') as inp:
        for line in inp.readlines():
            line = line.strip()[:-1]
            words = line.split(' ')

            tokens = list()
            for word in words:
                word = word.lower()
                found = False
                for k, v in lexicon.items():
                    for k1, v1 in v.items():
                        if word == k1.lower():
                            tokens.append(k)
                            found = True
                            break
                        if found:
                            break

            print(words)
            print(tokens)

            assert(len(words) == len(tokens))
            input_words.append(words)
            input_tokens.append(tokens)

    print(input_tokens)


def find_special_word(word):
    """
        Helper function in order to find words with special rules:
        DET + saw -> DET + Fem N
        saw + DET -> V + DET
        DET + cane -> DET + Fem N
        N + cane -> N + ADJ
    """

    for k, v in rules.items():
        for k1, v1 in v.items():
            if word in v1[0]:
                return v1
    return None


def tokens_to_output():
    """
    Gets all the tokens and words generated/mapped by `input_to_tokens` function from 'input_tokens' and 'input_words'
    containers and it maps them to French words while applying the rules.
    :return:
    """

    for tokens, words in zip(input_tokens, input_words):
        translated_sentence_words = list()

        i = 0
        for token, word in zip(tokens, words):

            w = words[i]
            rule_found = find_special_word(w)
            if rule_found is not None:
                left_side, right_side = rule_found
                word_index = left_side.index(w)
                token_replacement = right_side[word_index]
                if token_replacement not in lexicon.keys():
                    translated_sentence_words.append(token_replacement.lower())
                else:
                    translated_sentence_words.append(lexicon[token_replacement][w])
            else:
                options = lexicon[token]
                for k, v in options.items():
                    if word == k:
                        translated_sentence_words.append(v)

            i += 1

        sentence = ' '.join(translated_sentence_words)
        sentence = sentence.capitalize()
        sentence = f'{sentence}.'
        print(sentence)


if __name__ == '__main__':
    parse_lexicon()
    parse_rules()
    input_to_tokens()
    tokens_to_output()
