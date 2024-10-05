import numpy as np
import torch 
from preprocesstext import *
from tqdm import tqdm

columns = []
with open("predicted_val.txt", "r") as file:
    # Iterate over each line in the file
    for line in file:
        # Split the line into individual integers
        integers = list(map(int, line.strip().split()))
        # Process the integers as needed
        for integer in integers:
            columns.append(integer)

def list_interect(list1, list2):
    intersection_percentage = len(set(list1) & set(list2)) / max(len(list1), len(list2)) * 100
    return intersection_percentage

data = preprocess('A2_val.jsonl')

row_predicted = []
rows_gold = []

def find_elements(lst, index):
    if index == 0:
        left = []
        right = lst[index + 1:index + 5]
    else:
        left = lst[index - 2:index]
        right = lst[index + 1:index + 3]
    if index == len(lst)-1:
        right = lst[index-4:index]
        left = []
    

    return left+right

def tokenize_row(list):
    return  [word for string in list for word in re.findall(r'\b\w+\b', string)]


    



