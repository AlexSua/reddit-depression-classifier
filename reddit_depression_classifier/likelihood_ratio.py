import os
import re
import json
from math import log, sqrt

from reddit_depression_classifier.reddit_posts_extractor import extract_post_from_subreddit

# submissions_path = "./submissions_dataset/"

# Path where all monthly depression posts files are located.These monthly files have been previously extracted
# with the posts_extractor script.py.This script filters all monthly posts and saves a file with the posts of a specified subbredit.
depression_submissions_path = "./depression_submissions_dataset/"
depression_results_path = "./results/"


# Function that processes each line. Gets the selftext and title of reddit posts and concatenates them. Splits the obtained text
# into a list of words by discarding all symbols. It also cleans up all those posts that are deleted.
def _process_line(line):
    submission = json.loads(line)
    return re.sub("[^\w]", " ", ((submission['title'].lower() if submission['title'] != "[removed]" and submission[
        'title'] != "[deleted]" else "") + " " + (
        submission['selftext'].lower() if submission['selftext'] != "[removed]" and
        submission[
            'selftext'] != "[deleted]" else ""))).split()

# Function that retrieves all words from the absolute frequency file of English words.


def _get_absolute_frequency(absolute_frequency_file="./assets/count_1w.txt"):
    result = dict()
    totalNumber = 0
    with open(absolute_frequency_file) as file:
        for line in file:
            lineArray = line.replace("\n", "").split("\t")
            wordNumber = int(lineArray[1])
            result[lineArray[0]] = wordNumber
            totalNumber += wordNumber
    return result, totalNumber

# Function that gets the number of times a word is repeated and stores it in a dictionary. It also counts the total number of words.


def _get_depression_frequency(words):
    depressionDict = dict()
    observationsNumber = 0
    for file in os.listdir(depression_submissions_path):
        print(file)
        with open(depression_submissions_path + file) as depression_file:
            for line in depression_file:
                lineArray = _process_line(line)
                for word in lineArray:
                    if word in depressionDict:
                        depressionDict[word] = depressionDict.get(word) + 1
                        observationsNumber += 1
                    else:
                        if word in words:
                            depressionDict[word] = 1
                            observationsNumber += 1
    return depressionDict, observationsNumber


# Function that applies the likelihood ratio of a word based on the absolute frequency list and the number of observations.
def _root_log_likelihood_ratio(a, b, c, d):
    E1 = c * (a + b) / (c + d)
    E2 = d * (a + b) / (c + d)
    result = 2 * (a * log(a / E1 + (1 if a == 0 else 0)) +
                  b * log(b / E2 + (1 if b == 0 else 0)))
    result = sqrt(result)
    if (a / c) < (b / d):
        result = -result
    return result


# Create the sorted list according to the ratio of the words.
def _create_llr_list_file(words, depression_words):
    for key in depression_words[0].keys():
        depression_words[0][key] = (_root_log_likelihood_ratio(depression_words[0][key], words[0][key], depression_words[1],
                                                               words[1]), depression_words[0][key])
    path_exists = os.path.exists(depression_results_path)
    if not path_exists:
        os.makedirs(depression_results_path)
    
    with open(depression_results_path + "results_llr_list.txt", 'w+') as results_file:
        for key in sorted(depression_words[0], key=depression_words[0].get, reverse=True):
            if depression_words[0][key][0] >= 0:
                results_file.write(
                    key + "\t" + str(depression_words[0][key][0]) + "\n")
            # print(key, ":", depression_words[0][key])
    return depression_words[0]


def generate_llr_list(submisions_path=depression_submissions_path, results=depression_results_path):
    print("Generating llr list...")
    depression_results_path = submisions_path
    depression_results_path = results
    extract_post_from_subreddit("depression","./submissions_dataset/","./depression_submissions_dataset/")
    words = _get_absolute_frequency()
    depression_words = _get_depression_frequency(words[0])
    _create_llr_list_file(words, depression_words)
    print("Finished: "+depression_results_path)
