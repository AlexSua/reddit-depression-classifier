import json
import os
import re

from reddit_depression_classifier.reddit_posts_extractor import extract_post_from_subreddit

submissions_path = "./submissions_dataset/"
depression_submissions_path = "./depression_submissions_dataset/"
depression_results_path = "./results/"

# Process a line of the file obtaining a list of words clean of symbols, as well as return
# the original post.
def _process_line(line):
    submission = json.loads(line)
    if submission['selftext'] != "[removed]" and submission['selftext'] != "[deleted]" and submission['selftext']:
        return (
            submission,
            re.sub("[^\w]", " ", (submission['title'].lower() if submission['title'] != "[removed]" and submission[
                'title'] != "[deleted]" else "") + " " + submission['selftext'].lower()).split())
    else:
        return ("", [])

# Gets the weight of each post to see which one contain more words. Those in likelihood_ratio or page_rank solution.
def _calculate_weight_posts(posts_path, keywords):
    posts = dict()
    for file in os.listdir(posts_path):
        print(file)
        with open(posts_path + file) as depression_file:
            for line in depression_file:
                line_tuple = _process_line(line)
                if len(line_tuple[1]) <= 1:
                    continue
                value = 0
                for word in line_tuple[1]:
                    if word in keywords:
                        value += keywords[word]
                posts[line_tuple[0]["id"]] = (value, line_tuple[0], line_tuple[1])
    return posts

# Get the weights of the words in the likelihood_ratio and combine both lists to get a combined dictionary of the result
# likelihood_ratio and page_rank with the values ​​of likelihood_ratio. Words containing "depres" are filtered.
def _combine_results(file1, file2):
    file2dict = dict()
    result = dict()

    with open(file2) as f2:
        for line in f2:
            lineArray = line.replace("\n", "").split("\t")
            file2dict[lineArray[0]] = lineArray[1]

    with open(file1) as f1:
        for line in f1:
            lineArray = line.replace("\n", "").split("\t")
            if lineArray[0] in file2dict:
                if "depres" not in lineArray[0]:
                    result[lineArray[0]] = float(lineArray[1])
                    print(lineArray[0], result[lineArray[0]])

    return result

# Create a post object that will be saved as a result, marking it as positive or negative if it has the substring 'depres'
def _create_post_object(actual_post):
    positive = False
    for word in actual_post[2]:
        if "depres" in word:
            positive = True

    post = {}
    post["id"] = actual_post[1]["id"]
    post["positive"] = positive
    post["weight"] = actual_post[0]
    post["content"] = actual_post[1]
    post["content_processed"] = actual_post[2]
    return post

# Get the best and worst posts obtained from calculate_weight_posts. Being the best with the highest accumulated value of words related to depression obtained
# of the result of likelihood_ratio
def _get_best_worst_posts(posts):
    counter = 0
    best_posts = []
    worst_posts = []
    for key in sorted(posts, key=lambda x: posts.get(x)[0], reverse=True):
        print(posts[key][0])
        if counter < 100:
            best_posts.append(_create_post_object(posts[key]))
            counter += 1
        else:
            break

    counter = 0

    for key in sorted(posts, key=lambda x: posts.get(x)[0], reverse=False):
        print(posts[key][0])
        if counter < 100:
            worst_posts.append(_create_post_object(posts[key]))
            counter += 1
        else:
            break
    return best_posts, worst_posts


# Write a list to a file. It is used to save the list of best posts and worst posts.
def _write_list_in_file(list,file):
    with open(file,"w") as f:
        for element in list:
            json.dump(element, f)
            f.write("\n")



def process_and_obtain_training_data():
    results_dir = "./results/"
    keywords = _combine_results(results_dir+ "results_llr_list.txt",
                                results_dir+ "results_page_rank.txt")
    
    extract_post_from_subreddit("offmychest","./submissions_dataset/","./offmychest_submissions_dataset/")
    posts = _calculate_weight_posts("./offmychest_submissions_dataset/", keywords)

    best_posts, worst_posts = _get_best_worst_posts(posts)
    _write_list_in_file(best_posts,results_dir+"results_best_posts")
    _write_list_in_file(worst_posts, results_dir + "results_worst_posts")