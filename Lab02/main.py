# https://www.geeksforgeeks.org/jaro-and-jaro-winkler-similarity
# https://deep-translator.readthedocs.io/en/latest/README.html
# https://github.com/maxbachmann/Levenshtein

"""
Jaro-Winkler takes into account only matching characters and any required transpositions (swapping of characters) and
it gives more priority to prefix similarity.
Levenshtein counts the number of edits to convert one string to another.

https://srinivas-kulkarni.medium.com/jaro-winkler-vs-levenshtein-distance-2eab21832fd6#:~:text=Jaro%2DWinkler%20takes%20into%20account,convert%20one%20string%20to%20another.
https://stackoverflow.com/questions/25540581/difference-between-jaro-winkler-and-levenshtein-distance
"""

from math import floor, ceil
from deep_translator import GoogleTranslator
import Levenshtein


# Function to calculate the Jaro Similarity of two strings
def jaro_distance(s1, s2):
    if s1 == s2:
        return 1.0

    len1 = len(s1)
    len2 = len(s2)

    # Maximum distance up to which matching is allowed
    max_dist = floor(max(len1, len2) / 2) - 1

    # Count of matches
    match = 0

    # Hash for matches
    hash_s1 = [0] * len(s1)
    hash_s2 = [0] * len(s2)

    # Traverse through the first
    for i in range(len1):

        # Check if there is any matches
        for j in range(max(0, i - max_dist),
                       min(len2, i + max_dist + 1)):

            # If there is a match
            if s1[i] == s2[j] and hash_s2[j] == 0:
                hash_s1[i] = 1
                hash_s2[j] = 1
                match += 1
                break

    # If there is no match
    if match == 0:
        return 0.0

    # Number of transpositions
    t = 0
    point = 0

    # Count number of occurrences where two characters match but
    # there is a third matched character in between the indices
    for i in range(len1):
        if hash_s1[i]:

            # Find the next matched character in second
            while hash_s2[point] == 0:
                point += 1

            if s1[i] != s2[point]:
                t += 1
            point += 1
    t = t // 2

    # Return the Jaro Similarity
    return (match / len1 + match / len2 +
            (match - t) / match) / 3.0


# Jaro Winkler Similarity
def jaro_winkler(s1, s2):
    jaro_dist = jaro_distance(s1, s2);

    # If the jaro Similarity is above a threshold
    if jaro_dist > 0.7:

        # Find the length of common prefix
        prefix = 0

        for i in range(min(len(s1), len(s2))):
            # If the characters match
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break

        # Maximum of 4 characters are allowed in prefix
        prefix = min(4, prefix)

        # Calculate jaro winkler Similarity
        jaro_dist += 0.1 * prefix * (1 - jaro_dist)

    return jaro_dist


translator = GoogleTranslator(source='auto')
langs_dict = translator.get_supported_languages(as_dict=True)


def process_text(text):
    source_language = 'ro'
    for k, v in langs_dict.items():
        translator.source = source_language
        translator.target = v
        text = translator.translate(text)

        source_language = v

    last_target_language = 'ro'
    translator.source = source_language
    translator.target = last_target_language
    text = translator.translate(text)

    return text


def process_data(input_data):
    output_data = process_text(input_data)
    jaro = jaro_distance(input_data, output_data)
    jaro_winkler_data = jaro_winkler(input_data, output_data)
    levenshtein = Levenshtein.distance(input_data, output_data)
    # print(f'Input: {input_data}')
    # print(f'Output: {output_data}')
    print(f'Translation count: {len(langs_dict.keys()) + 1}')
    print(f'Jaro Similarity: {jaro}')
    print(f'Jaro Winkler Similarity: {jaro_winkler_data}')
    print(f'Levenshtein Similarity: {levenshtein}')


if __name__ == "__main__":
    input_01 = 'O simplă propoziție.'
    print('-------- Input 01 --------')
    process_data(input_01)

    input_02 = '''
    Cațelus cu părul creț,\n
    Fură rața din coteț.\n
    El se jură că nu fură,\n
    Și l-am prins cu rața-n gura,\n
    Și cu ou-n buzunar,\n
    Hai la Sfatul Popular.\n        
    Nu mă duc c-am fost o dată,\n
    Și am căzut cu nasu’-n baltă.\n'''

    print('-------- Input 02 --------')
    process_data(input_02)




