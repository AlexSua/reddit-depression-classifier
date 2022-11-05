import os
import json
import os
import re


def _process_line(line):
    submission = json.loads(line)
    if submission['selftext'] != "[removed]" and submission['selftext'] != "[deleted]" and submission['selftext']:
        return (
            submission,
            re.sub("[^\w]", " ", (submission['title'].lower() if submission['title'] != "[removed]" and submission[
                'title'] != "[deleted]" else "") + " " + submission['selftext'].lower()).split())
    else:
        return ("", [])


#Get all mental health posts to filter in the extractor.
def get_subreddits_list_from_file(file):
    mhsub = set()
    with open(file) as f:
        for line in f:
            mhsub.add(line.replace("\n", ""))
    return mhsub


#This extractor will be in charge of obtaining random posts. What it does is go through the monthly archive, get all those posts
# with more than 100 words, discard all those that contain words related to depression.
#Each time a post is obtained, it is done modulo 100 to obtain the hundredth post that shares these criteria. This will be the one added to the list
# of random posts.
#Also filtered by previous subreddits.
def extract_random_posts(submissions_path, result_path, exception_list):
    for file in os.listdir(submissions_path):
        print("Extracting submissions from dataset: " + file + "...")
        depression_file = open(result_path , 'w')
        with open(submissions_path + file) as fp:
            counter = 0
            totalCounter = 0
            for line in fp:
                if totalCounter % 100 == 0:
                    submission = json.loads(line)
                    appendable = True
                    if "subreddit" in  submission and submission["subreddit"] not in exception_list:
                        word_list = _process_line(line)[1]
                        if len(word_list) > 100:
                            for word in word_list:
                                if "depres" in word:
                                    appendable = False
                                    break
                            if appendable:
                                obj = {}
                                obj["id"] = submission["id"]
                                obj["content"] = submission
                                obj["content_processed"] = word_list
                                json.dump(obj,depression_file)
                                depression_file.write("\n")
                                counter += 1
                    if counter > 150:
                        break
                totalCounter+=1

        depression_file.close()


# Get posts from a given subreddit.
def extract_post_from_subreddit(subreddit_name,submissions_path,result_path):
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    subreddit_name_len = len('"' + subreddit_name + '"')
    for file in os.listdir(submissions_path):
        depression_file_path = result_path + subreddit_name + '-' + file
        if not os.path.isfile(depression_file_path):
            print("Extracting " + subreddit_name + " submissions from dataset: " + file + "...")
            depression_file = open(depression_file_path, 'w')
            with open(submissions_path + file, "r+") as fp:
                counter = 0
                for line in fp:
                    num = line.find('"subreddit":"') + 12
                    if line[num:num + subreddit_name_len].lower() == '"' + subreddit_name.lower() + '"':
                        depression_file.write(line)
                        counter += 1
                print("\t" + str(counter) + " posts found about " + subreddit_name)
            depression_file.close()






# extract_post_from_subreddit("addiction","./submissions_dataset/","./predict_submissions_dataset/")
# extract_post_from_subreddit("offmychest","./submissions_dataset/","./offmychest_submissions_dataset/")
# extract_post_from_subreddit("alcoholism","./submissions_dataset/","./to_predict_submissions_dataset/")
# extract_random_posts("./submissions_dataset/", "./results/results_random_posts", get_mental_health_subreddits("./mental_health_posts"))

