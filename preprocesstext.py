import json
import re
import nltk
from nltk.corpus import stopwords
from copy import deepcopy
import random
import torch
import numpy as np
from unidecode import unidecode
import nltk
from tqdm import tqdm

stop_words = stopwords.words('english')
def remove_special_characters(s):
    return re.sub('[^A-Za-z0-9# %]+', ' ', s)

def remove_unicode(text):
    # Define a regular expression pattern to match Unicode characters
    unicode_pattern = re.compile(r'[\u0080-\uffff]')  # Matches any Unicode character

    # Replace Unicode characters with an empty string
    cleaned_text = unicode_pattern.sub('', text)
    
    return cleaned_text

def remove_question_mark(s):
    """
    Removes all '?' characters from the input string.

    Parameters:
    s (str): The input string from which '?' will be removed.

    Returns:
    str: The modified string with '?' characters removed.
    """
    return s.replace('?', '')

def remove_braces(text):
    cleaned_text = text.replace("(", "").replace(")", "")
    return cleaned_text


def preprocess(path):
    data_load = []
    with open(path, 'r') as file:
        count = 0
        s = []
        s_new = []
        for line in tqdm(file):
           
            data_new = {}
            data = json.loads(line)
            s.append(data)
            data_new['columns_original'] = s[count]['table']['cols']
            s_new.append(data)
            # data_new['columns_original'] = s_new[count]['table']['cols']
            label_col = s[count]['label_col'][0]
            label_col = remove_braces(label_col)
            label_col = unidecode(label_col)
            label_col = remove_unicode(label_col)
            label_col = remove_special_characters(label_col)
            # label_col = 'sos '+ label_col+' eos'
            label_col = label_col.lower()
            label_col = label_col.split()

            row_label_i = s[count]['label_row']
            # question = 'sos '+remove_question_mark(s[count]['question'])+' eos'
            question = remove_question_mark(s[count]['question'])
            question = remove_special_characters(remove_unicode(unidecode(remove_braces(question))))
            question_tokenized = question.lower().split()

            filtered_tokens = [word for word in question_tokenized if word.lower() not in stop_words]
            for i in range(len(s[count]['table']['cols'])):
                s[count]['table']['cols'][i] = remove_braces(s[count]['table']['cols'][i])
                s[count]['table']['cols'][i] =  unidecode(s[count]['table']['cols'][i])
                s[count]['table']['cols'][i] = remove_unicode(s[count]['table']['cols'][i])
                s[count]['table']['cols'][i] = remove_special_characters(s[count]['table']['cols'][i])
                # s[count]['table']['cols'][i] = 'sos '+ s[count]['table']['cols'][i] + ' eos'
                s[count]['table']['cols'][i] = s[count]['table']['cols'][i] 
                s[count]['table']['cols'][i] = s[count]['table']['cols'][i].lower().split()

            
            for i in range(len(s[count]['table']['rows'])):
                # s[count]['table']['rows'][i] = ['sos']+s[count]['table']['rows'][i]+['eos']
                for j in range(len(s[count]['table']['rows'][i])):
                    s[count]['table']['rows'][i][j] = remove_braces(s[count]['table']['rows'][i][j])
                    s[count]['table']['rows'][i][j] = unidecode(s[count]['table']['rows'][i][j])
                    s[count]['table']['rows'][i][j] = remove_unicode(s[count]['table']['rows'][i][j])
                    s[count]['table']['rows'][i][j] = remove_special_characters(s[count]['table']['rows'][i][j])
                    s[count]['table']['rows'][i][j] = s[count]['table']['rows'][i][j].lower()
                    # row_tokens.append(s[count]['table']['rows'][i][j].lower().split())
                # s[count]['table']['rows'][i] = [word for item in s[count]['table']['rows'][i] for word in item.split()]
               


            

            data_new['question'] = filtered_tokens
            data_new['columns'] = s[count]['table']['cols']
            data_new["rows"] = s[count]['table']['rows']
            # data_new['rows'] = row_tokens
            data_new['col_label'] = s[count]['table']['cols'].index(label_col)
            data_new['row_labels'] = row_label_i
            data_new["qid"] = s[count]["qid"]
           
            data_load.append(data_new)
            count+=1
            # print(f'data {count} is being processed!')
            # if count ==1:
            #     break

    print('data processed!')   
    

    return data_load

def column_data(data):
    data_column = []

    for i in range(len(data)):
        for j in range(len(data[i]['columns'])):
            data_all_new = {}
            data_all_new['question'] = data[i]['question']
            data_all_new['column'] = data[i]['columns'][j]
            if j == data[i]['col_label']:
                data_all_new['col_label'] = 1
            else:
                data_all_new['col_label'] = 0
                
            data_column.append(data_all_new)

    print('column data generated!')

    return data_column

def row_data(data):
    data_row = []
    for i in tqdm(range(len(data))):
        data_all_new = {}
        rows_new = data[i]['rows']
        row_label = data[i]['row_labels']
        row_label.sort(reverse=True)

        for j in range(len(row_label)):
            data_all_new['question'] = data[i]['question']
            data_all_new['row'] = data[i]['rows'][j]
            data_all_new['row_label'] = 1
            # for k in range(len(data_all_new['row'])):
            #     if i == data[i]
            data_row.append(data_all_new)
        
        for label in row_label:
            del rows_new[label]

        if len(rows_new)>2*len(row_label):
            rows_new =  random.sample(rows_new,2*len(row_label))
            for j in range(len(rows_new)):
                data_all_new = {}
                data_all_new['question'] = data[i]['question']
                data_all_new['row'] = rows_new[j]
                data_all_new['row_label'] = -1
                data_row.append(data_all_new)
        
        elif len(rows_new)==0:
            continue
        elif len(rows_new)<len(row_label):
            rows_new =  random.sample(rows_new,(len(rows_new)))
            for j in range(len(rows_new)):
                data_all_new = {}
                data_all_new['question'] = data[i]['question']
                data_all_new['row'] = rows_new[j]
                data_all_new['row_label'] = -1
                data_row.append(data_all_new)


        else:
            rows_new =  random.sample(rows_new,(len(row_label)))
            for j in range(len(rows_new)):
                data_all_new = {}
                data_all_new['question'] = data[i]['question']
                data_all_new['row'] = rows_new[j]
                data_all_new['row_label'] = -1
                data_row.append(data_all_new)


    
    print('row data generated!')

    
    return data_row


