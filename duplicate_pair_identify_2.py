import pandas as pd
import numpy as np
from nltk.corpus import stopwords # Natural Language Toolkit which provides a suit of text processing libraries for tokenization and other purposes.

#The stop words from the question is extracted and stored as a list.
st_words = set(stopwords.words("english"))

# Read the training dataset train.csv
input_frame = pd.read_csv('train.csv')

# The similarity is measured using Jaccard score, which is a statistic used for comparing the similiarity and diversity of sample sets.
def compute_score(row):
    ques1_list = []
    ques2_list = []
    #removing the stop words from the input questions because sometimes they cause a retrieving zero hits. Append them to alist.
    for word in str(row['question1']).lower().split():
        if word not in st_words:
            ques1_list.append(word)
    for word in str(row['question2']).lower().split():
        if word not in st_words:
            ques2_list.append(word)
    number = len(set(ques1_list) & set(ques2_list)) #The intersection operation similar to a set is performed.
    deno = len(set(ques1_list) | set(ques2_list)) #The set union operation is performed and thus get the toatal size.
    if len(ques1_list) == 0 or len(ques2_list) == 0:
        return 0
    score = float(number)/float(deno) #The similarity score is calculated for every pair of questions
    return score

score_list = []
for index, row in input_frame.iterrows():
    score = compute_score(row)
    score_list.append(score)

input_frame['jaccard_score'] = score_list
A = np.array([score_list, np.ones(len(score_list))])
w = np.linalg.lstsq(A.T,input_frame['is_duplicate'])[0]
#The training part ends here.

#The testing starts from here onwards.
testdata_frame = pd.read_csv('test.csv')
jaccard_score_list = [] 

#Now compute the similarity score for the test dataset.
for index, row in testdata_frame.iterrows():
    score = compute_score(row)
    jaccard_score_list.append(score)

#creating a frame for writing the output file
testdata_frame['jaccard_score'] = jaccard_score_list
testdata_frame['duplicate_index'] = testdata_frame['jaccard_score']*w[0]+w[1]

#The result is written to the file output.csv
sub = pd.DataFrame()
sub['test_id'] = testdata_frame['test_id']
sub['duplicate_index'] = testdata_frame['duplicate_index']
sub.to_csv('output.csv', index=False)




