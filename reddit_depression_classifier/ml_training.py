import json
import os
import re

from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from reddit_depression_classifier.reddit_posts_extractor import extract_post_from_subreddit, extract_random_posts, get_subreddits_list_from_file

# All the posts of training_data are obtained.
def _get_posts_from_files(files):
    posts = dict()

    for file in files:
        with open(file) as f:
            for line in f:
                obj = json.loads(line)
                posts[obj["id"]] = obj
    return posts

# Function to process line for alcoholism posts stop playing stop smoking...
def _process_line(line):
    submission = json.loads(line)
    if submission['selftext'] != "[removed]" and submission['selftext'] != "[deleted]" and submission['selftext']:
        return (
            submission,
            re.sub("[^\w]", " ", (submission['title'].lower() if submission['title'] != "[removed]" and submission[
                'title'] != "[deleted]" else "") + " " + submission['selftext'].lower()).split())
    else:
        return ("", [])

# Prepare the training list of the positive posts obtained in training_data.py. Get a list of 1s and a list of post texts.
def _prepare_positive_data_for_training(posts):
    post_text_list = []
    positiveness_list = []
    for key in posts:
        new_content_processed = []
        for token in posts[key]["content_processed"]:
            if "depres" not in token:
                new_content_processed.append(token)
        if bool(posts[key]["positive"]):
            post_text_list.append(' '.join(new_content_processed))
            positiveness_list.append(1)
    return post_text_list, positiveness_list

# Prepare the training list of random negative posts obtained from the random posts script. Get a list of 1s and a list of post texts.
def _prepare_negative_data_for_training(file):
    negative_list = []
    positiveness_list = []
    counter = 0
    with open(file) as f:
        for line in f:
            submission = json.loads(line)
            appendable = True
            for word in submission["content_processed"]:
                if "depres" in word:
                    appendable = False
                    break
            if appendable:
                negative_list.append(' '.join(submission["content_processed"]))
                positiveness_list.append(0)
                counter += 1
            if counter >= 100:
                break
    return negative_list, positiveness_list

# Obtain a dictionary with all the posts obtained from alcoholism, quitting smoking.... These posts are obtained with the posts_extractor script.
def _get_ad_al_stop_submissions(path):
    result = dict()
    for file in os.listdir(path):
        with open(path + file) as f:
            for line in f:
                line_processed = _process_line(line)
                if len(line_processed[1]) > 5:
                    d = {}
                    d["content"] = line_processed[0]
                    d["content_processed"] = line_processed[1]
                    result[line_processed[0]["id"]] = d

    return result

# Get a list of a dictionary of posts to make a prediction. It also returns a list of ids in order to get the original post from the dictionary
# since the list of posts obtained will only have a text already processed.
def _get_list_from_dict(dictionary):
    result = []
    id_list = []
    for key in dictionary:
        result.append(' '.join(dictionary[key]["content_processed"]))
        id_list.append(key)
    return result, id_list

def _extract_data():
    extract_post_from_subreddit("addiction","./submissions_dataset/","./predict_submissions_dataset/")
    extract_post_from_subreddit("alcoholism","./submissions_dataset/","./to_predict_submissions_dataset/")
    extract_random_posts("./submissions_dataset/", "./results/results_random_posts", get_subreddits_list_from_file("./assets/mental_health_posts"))

def ml_training_and_prediction():
    results_dir = "./results/"

    _extract_data()
    posts = _get_posts_from_files([results_dir + "results_best_posts", results_dir + "results_worst_posts"])

    #Get positive training data
    Depression, true_array = _prepare_positive_data_for_training(posts)

    # Get the negatives
    NotDepression, false_array = _prepare_negative_data_for_training(results_dir + "results_random_posts")

    x = Depression + NotDepression
    y = true_array + false_array

    # vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    # X = vectorizer.fit_transform(x).toarray()

    #A vectorizer is instantiated that will transform the text into numbers
    vectorizer = TfidfVectorizer(binary=True, ngram_range=(1, 3), stop_words=stopwords.words('english'))

    #Gets array of numbers transformed from the text and saves it as a vocabulary in the vectorizer, to make later transformations.
    X = vectorizer.fit_transform(x).toarray()

    #The data is divided into training set 80% and testing set 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    #se instantiate classifier
    classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

    #classifier is trained
    classifier.fit(X_train, y_train)

    #Dictionary of alcoholism posts is obtained...
    submissions = _get_ad_al_stop_submissions("./to_predict_submissions_dataset/")

    #List of the previously obtained dictionary is obtained
    submissions_list, id_list = _get_list_from_dict(submissions)

    #It is predicted and the prediction result is obtained
    prediction_result = classifier.predict(vectorizer.transform(submissions_list))

    #It will be used to store depressed people.
    depressed_people = set()
    positive_posts = list()
    for i, x in enumerate(prediction_result):
        if x == 1:
            print(submissions[id_list[i]]["content"])
            if "deleted" not in submissions[id_list[i]]["content"]["author"]:
                depressed_people.add(submissions[id_list[i]]["content"]["author"])
            positive_posts.append(submissions[id_list[i]]["content"])

    print("\n"
            "Depressed persons:")
    for person in depressed_people:
        print(person)

    print("\n")

    #Precision is calculated, the rest is in the result.
    accuracy_test =accuracy_score(y_test, classifier.predict(X_test))
    accuracy_train =accuracy_score(y_train, classifier.predict(X_train))

    with open("./results/results_depressed_people_posts_alc_ad_stop",'w') as f:
        f.write("*********CLASSIFIER ACCURACY************"+"\n")
        f.write("Accuracy test set:" + str(accuracy_test)+ "\n")
        f.write("Precision training set:" + str(accuracy_train) + "\n")
        f.write("\n")
        f.write("********DEPRESSED PEOPLE***********"+"\n")
        for person in depressed_people:
            f.write(person+"\n")
        f.write("\n")
        f.write("********POSTS ON ALCOHOLISM, ADDICTION AND STOPPING GAMBLING AND SMOKING THAT HAVE GIVEN POSITIVE IN THE CLASSIFIER (The list of depressed people has been obtained from these posts)***********\n")
        for positive_post in positive_posts:
            json.dump(positive_post,f)
            f.write("\n")

    print("Accuracy test %s"
            % (accuracy_test) )
    print("Accuracy training %s"
            % (accuracy_train))
