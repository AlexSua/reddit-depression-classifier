#!/usr/bin/env python3
import os
import json
import nltk
from nltk import tokenize, word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

stop_words_list = set(stopwords.words('english'))
submissions_path = "./submissions_dataset/"
depression_submissions_path = "./depression_submissions_dataset/"
depression_results_path = "./results/"

# Transform the obtained line into a list of phrases.
def _process_line(line):
    sentences = []
    submission = json.loads(line)
    if submission['title'] != "[removed]" and submission['title'] != "[deleted]":
        sentences += tokenize.sent_tokenize(submission['title'], 'english')
    if submission['selftext'] != "[removed]" and submission['selftext'] != "[deleted]":
        sentences += tokenize.sent_tokenize(submission['selftext'], 'english')
    return _filter_sentences(sentences)

# Filter the phrases obtained by eliminating the stop words, common words in English.
def _filter_sentences(sentences):
    filtered_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        word_tokens = word_tokenize(sentence, 'english')
        filtered_sentence = []
        for word in word_tokens:
            if word not in stop_words_list:
                # filtered_word = re.sub("[^\w]", " ", word.lower())
                if word.isalpha():
                    filtered_sentence.append(word)
        filtered_sentences.append(filtered_sentence)
    return filtered_sentences


# From the set of phrases obtained in a post, the co-elements are created, which will be each of the bidirectional relations of the final graph.
# From this function all the words presented in the previous sentences and a set of co-elements will be obtained.
# ws is the window size (Window Size)
# Cooelements will initially be stored in a set due to their performance in checking occurrences.
def _get_vocab_and_cooelements(sentences, ws, word_vocab, cooelements):
    for s in sentences:
        sentence_len = len(s)
        for i, word in enumerate(s):
            if word not in word_vocab:
                word_vocab[word] = 1
            for j in range(i + 1, i + ws):
                if j >= sentence_len:
                    break
                cooelement = (word, s[j])
                if cooelement not in cooelements:
                    cooelements.add(cooelement)
    return word_vocab, cooelements


# def stop_words():
#     stop_word_list = []
#     with open("./stop.txt") as file:
#         for line in file:
#             line = line.replace(" ", "").replace("\n", "")
#             lineArray = line.split("|")
#             word = ""
#             if len(lineArray) > 1:
#                 word = lineArray[0]
#             else:
#                 word = line
#             if word:
#                 stop_word_list.append(word)
#
#     return stop_word_list


# Transforms the set of elements into a dictionary. In this dictionary an element will be stored as a key
# and as a value a list of elements that are co-elements of the previous element. This is done because as a result we will obtain
# a dictionary that contains for each element all those elements that will add value to it. This value for each of them subsequently
# can be calculated by accessing the element inside the dictionary and checking its length.
def _get_cooelements_dict(cooelements):
    cooelementsDict = dict()
    for cooelement in cooelements:
        if cooelement[0] not in cooelementsDict:
            cooelementsDict[cooelement[0]] = [cooelement[1]]
        else:
            cooelementsDict[cooelement[0]] += [cooelement[1]]
        if cooelement[1] not in cooelementsDict:
            cooelementsDict[cooelement[1]] = [cooelement[0]]
        else:
            cooelementsDict[cooelement[1]] += [cooelement[0]]
    return cooelementsDict

# Calculate the weight for each element. To do this, the dictionary of cooelements is traversed, within each element the list of its cooelements is traversed.
# once a cooelement is obtained, this element is accessed in the dictionary and its length is obtained in order to calculate the value it contributes to the main element.
# The sum of the value of all its co-elements will be its final value.
def _calculate_weight(cooelements_dict, damping_factor, word_vocab):
    weight_result = dict()
    for key in cooelements_dict:
        val = 0
        for elem in cooelements_dict[key]:
            val += word_vocab[elem] / float(len(cooelements_dict[elem]))
        weight_result[key] = (1 - float(damping_factor)) * (1 / float(len(word_vocab))) + damping_factor * val
    return weight_result


# Write the computer result to a file
def _write_result(dictionary, path):
    with open(path, 'w') as results_file:
        for key in sorted(dictionary, key=dictionary.get, reverse=True):
            results_file.write(key + "\t" + str(dictionary[key]) + "\n")
            # print(key, ":", depression_words[0][key])

# Initialize the value of each word to 1/number of vocabulary words.
def _initialize_weight(vocab):
    vocab_len = float(len(vocab))
    for key in vocab:
        vocab[key] = 1 / vocab_len
    return vocab

# Function that will be in charge of obtaining all the depression files of the directory that is assigned to it and will execute the functions explained above
# in order to obtain the pagerank value for each token.
# A total of 50 iterations will be applied.
def _analyzing_files(path, d, iter, ws):
    vocab = dict()
    cooelements = set()

    for file in os.listdir(path):
        print(path + file)
        counter = 0
        with open(depression_submissions_path + file) as depression_file:
            for line in depression_file:
                counter += 1
                sentences = _process_line(line)
                vocab, cooelements = _get_vocab_and_cooelements(sentences, ws, vocab, cooelements)

    codict = _get_cooelements_dict(cooelements)
    vocab = _initialize_weight(vocab)
    print("Page rank iteration:")
    for i in range(iter):
        print("- Iteration [" + str(i) + "]")
        vocab = _calculate_weight(codict, d, vocab)
    return vocab


def generate_page_rank_list(submisions_path=depression_submissions_path, results=depression_results_path):
    print("Generating word importance list using page rank...")
    depression_results_path = submisions_path
    depression_results_path = results
    damping = 0.85
    iterations = 50
    window_size = 3
    result = _analyzing_files(depression_submissions_path, damping, iterations, window_size)
    _write_result(result, depression_results_path + "results_page_rank.txt")
    print("Finished: "+  depression_results_path + "results_page_rank.txt")


