import math
from operator import truediv
import typing

import numpy as np
import sys
import data
from data.split import split

class DecisionTreeClassifierNode:
    def __init__(self, dataset: np.ndarray, depth: int):
        self.depth = depth
        [label, label_count] = np.unique(dataset[:,-1], return_counts= True)
        if (len(label_count) == 0):
            print(f"depth = {self.depth}, dataset = {dataset}")
            print('error: node with empty dataset')
            return
        elif(len(label_count) == 1):
            self.leaf = True
            self.room = label
            self.elements_under_leaf = dataset.shape[0]
            return 
        else:
            self.leaf = False
            [self.split_value, self.column] = self.find_split(dataset)
            lower_dataset = dataset[dataset[:, self.column] < self.split_value]
            upper_dataset = dataset[dataset[:, self.column] > self.split_value]
            self.lowerBranch = DecisionTreeClassifierNode(lower_dataset, depth + 1)
            self.upperBranch = DecisionTreeClassifierNode(upper_dataset, depth + 1)
            return 

    # evaluate the whole tree that the current node is a part of
    def evaluate(self, test_db):
        return np.mean(np.equal(self.predict(test_db[:,:-1]),test_db[:, -1]))
        
    def prune(self, validation, root):
        if self.leaf:
            return
        else:
            self.lowerBranch.prune(validation, root)
            self.upperBranch.prune(validation, root)
        
        if self.lowerBranch.leaf and self.upperBranch.leaf:
            validationAccuracyBefore = root.evaluate(validation)
            self.room = self.lowerBranch.room if self.lowerBranch.elements_under_leaf < self.upperBranch.elements_under_leaf else self.upperBranch.room
            self.leaf = True
            validationAccuracyAfter = root.evaluate(validation)
            if validationAccuracyAfter < validationAccuracyBefore:
                self.leaf = False
                self.room = None
            else:
                self.elements_under_leaf = self.lowerBranch.elements_under_leaf + self.upperBranch.elements_under_leaf
                self.lowerBranch = None
                self.upperBranch = None

    
    
    def predict_row(self, x):
        if self.leaf:
            return self.room
        elif x[self.column] < self.split_value:
            return self.lowerBranch.predict_row(x)
        return self.upperBranch.predict_row(x)

    def predict(self, x):
        results = list()
        for r in x:
            results.append(self.predict_row(r))
        return np.squeeze(np.array(results))

    def compute_entropy(self, dataset):
        [_, y] = np.unique(dataset[:,-1], return_counts=True)

        running_sum = 0
        for r in y:
            running_sum -= (float(r) / float(np.sum(y))) * math.log2(float(r) / float(np.sum(y)))
        return running_sum


    def find_split(self, dataset):
        best_total_entropy = sys.float_info.max
        best_column = 0
        best_split_value = 0
        for c in range(0,dataset.shape[1] - 1):
            sorted_dataset  = dataset[np.argsort(dataset[:,c])]
            for i in enumerate(sorted_dataset[:-1]):
                if sorted_dataset[i[0], c] ==  sorted_dataset[i[0]+1, c]:
                    continue
                lower = sorted_dataset[:i[0]]
                upper = sorted_dataset[i[0]:]
                lower_entropy = self.compute_entropy(lower)
                upper_entropy = self.compute_entropy(upper)
                total_entropy = (i[0] / dataset.shape[0]) * lower_entropy + ((dataset.shape[0] - i[0]) / dataset.shape[0]) * upper_entropy
                if best_total_entropy > total_entropy:
                    best_total_entropy = total_entropy
                    best_column = c
                    best_split_value = (sorted_dataset[i[0], c] +  sorted_dataset[i[0]+1, c]) / 2.0
        return (best_split_value, best_column)

    def generate_confusion_matrix(self, test_set):
        confusion_matrix = np.zeros((4,4))
        predictions = self.predict(test_set[:, :-1])
        actual = test_set[:, -1]
        assert len(actual) == len(predictions)
        for row, col in zip(actual,predictions):
            confusion_matrix[int(row)-1,int(col)-1] += 1
        return confusion_matrix
        
    
   

      
