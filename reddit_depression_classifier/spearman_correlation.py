
import scipy.stats

depression_results_path = "./results/"

# Function that calculates the spearman correlation between two lists of values. In this case the result of the weights of the words
# result of likelihood ratio and the result of page_rank. The result obtained is 0.8. This means that both lists
# have an 80% correlation.
def _spearman_correlation_calculation(file1, file2):
    file2dict = dict()
    correlationlist1 = []
    correlationlist2 = []

    with open(file2) as f2:
        for line in f2:
            lineArray = line.replace("\n", "").split("\t")
            file2dict[lineArray[0]] = lineArray[1]

    with open(file1) as f1:
        for line in f1:
            lineArray = line.replace("\n", "").split("\t")
            if lineArray[0] in file2dict:
                correlationlist1.append(float(lineArray[1]))
                correlationlist2.append(float(file2dict[lineArray[0]]))

    spear = scipy.stats.stats.spearmanr(correlationlist1, correlationlist2)
    return spear



def get_spearman_correlation(results_file1=depression_results_path + "results_llr_list.txt", results_file_2=depression_results_path + "results_page_rank.txt"):
    correlation = _spearman_correlation_calculation(results_file1,
                                        results_file_2)
    print("Spearman correlation between llr_list and page_rank_list",correlation)
    return correlation
