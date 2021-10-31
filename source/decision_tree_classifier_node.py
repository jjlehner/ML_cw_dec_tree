import math
import typing

import numpy as np
import sys

import matplotlib
from matplotlib import pyplot

import draw

class DecisionTreeClassifierNode:

    def __init__(self, dataset: np.ndarray, depth: int):

        # Initialize branches, label
        self.lower_branch = None
        self.upper_branch = None

        # Initialize meta-information
        self.depth: int = depth
        self.label = ''
        self.leaf = False
        self.split_value = 0
        self.column = 0
        self.width = 0
        self.elements_under_leaf = 0

        # Evaluate label and unique values in data set
        label, unique_counts = np.unique(dataset[:, -1], return_counts=True)
        label_count = len(unique_counts)

        # If only one sample present, flag node as leaf
        if label_count == 1:
            self.leaf = True
            self.width = 10
            self.label = int(label[0])
            self.elements_under_leaf = dataset.shape[0]

        # Otherwise, split the data set and create two sub-branches
        else:
            self.split_value, self.column = self.find_split(dataset)
            self.label = f'x{self.column} : < {self.split_value} ≤'

            lower_dataset = dataset[dataset[:, self.column] < self.split_value]
            upper_dataset = dataset[dataset[:, self.column] > self.split_value]
            self.lower_branch = DecisionTreeClassifierNode(lower_dataset, depth + 1)
            self.upper_branch = DecisionTreeClassifierNode(upper_dataset, depth + 1)

            self.width += self.lower_branch.width
            self.width += self.upper_branch.width
    
    def evaluate(self, test_db) -> float:
        """ Evaluates a tree

        Arguments
        ---------
        test_db: np.ndarray
            the dataset with which to evaluate the tree gainst

        Returns
        -------
        accuracy: float
            the accuracy of the tree with the test_db testing set
        """
        
        return np.mean(np.equal(self.predict(test_db[:,:-1]),test_db[:, -1]))
        
    def prune(self, validation: np.ndarray, root: 'DecisionTreeClassifierNode'):
        """ Reduce overfitting across the tree to increase generalyse to unknown data. 
                Change the state of the tree by removing/pruning nodes that decrease the accuracy of the Decision Tree

        Arguments
        ---------
        validation: np.ndarray
            The dataset used to evaluate the change in accuracy after pruning a node

        root: DecisionTreeClassifierNode
            The root node of the tree on which to apply the pruning
        """
        
        if self.leaf:
            return
        else:
            self.lower_branch.prune(validation, root)
            self.upper_branch.prune(validation, root)
        
        if self.lower_branch.leaf and self.upper_branch.leaf:
            validation_accuracy_before = root.evaluate(validation)
            self.label = self.lower_branch.label if self.lower_branch.elements_under_leaf > self.upper_branch.elements_under_leaf else self.upper_branch.label
            self.leaf = True
            validation_accuracy_after = root.evaluate(validation)
            if validation_accuracy_after < validation_accuracy_before:
                self.leaf = False
                self.label = f'x{self.column} : < {self.split_value} ≤'
            else:
                self.elements_under_leaf = self.lower_branch.elements_under_leaf + self.upper_branch.elements_under_leaf
                self.lower_branch = None
                self.upper_branch = None
                self.split_value = None


    def predict_row(self, dataset: np.ndarray) -> np.ndarray:
        """ Predict the value of the first row in a dataset

        Arguments
        ---------
        dataset: np.ndarray


        Returns
        -------
        row_prediction: np.ndarray
            the predicted row value
        """

        if self.leaf:
            return self.label

        align_lower = dataset[self.column] < self.split_value
        branch = self.lower_branch if align_lower else self.upper_branch

        return branch.predict_row(dataset)

    def predict(self, dataset: np.ndarray) -> np.ndarray:
        """ Predict the result for each row in a dataset

        Arguments
        ---------
        dataset: np.ndarray
            dataset, the rows of which to predict

        Returns
        -------
        predictions: np.ndarray
            list of predictions for each row of the associated test dataset
        """
        return np.squeeze(np.array([self.predict_row(row_index) for row_index in dataset]))

    def compute_entropy(self, dataset) -> float:
        """ Evaluate the entropy of a dataset

        Arguments
        ---------
        dataset: np.ndarray
            dataset, the entropy of which to calculate

        Returns
        -------
        entropy: float
            entropy of the dataset
        """

        # Extract a subarray of unique elements in the dataset
        _, unique_elements = np.unique(dataset[:, -1], return_counts=True)

        # Perform
        running_sum = 0
        for element in unique_elements:
            unique_value = float(np.sum(unique_elements))
            element_value = float(element)

            fraction = element_value / unique_value
            running_sum -= fraction * math.log2(fraction)

        return running_sum

    def find_split(self, dataset: np.ndarray) -> typing.Tuple[float, int]:
        """ Finds the ideal split value in a dataset

        Arguments
        ---------
        dataset: np.ndarray
            dataset of which to calculate the best split

        Returns
        -------
        best_split, best_column: typing.Tuple[float, int]
            the best split, and best column values
        """

        # Initialize loop variants
        best_entropy = sys.float_info.max
        best_column = 0
        best_split = 0

        # Iterate over each column in the dataset (ignoring the label column)
        column_count = dataset.shape[1]
        for column_index in range(0, column_count - 1):

            # Sort the data set, and extract the column
            sorted_dataset = dataset[np.argsort(dataset[:, column_index])]
            column = sorted_dataset[:, column_index]

            # Iterate over each row in the column
            for row_index in enumerate(sorted_dataset[:-1]):

                # Ignore consecutive identical rows
                if column[row_index[0]] == column[row_index[0] + 1]:
                    continue

                # Create a lower and upper dataset, and evaluate their
                # respective entropies
                lower = sorted_dataset[:row_index[0]]
                upper = sorted_dataset[row_index[0]:]
                lower_entropy = self.compute_entropy(lower)
                upper_entropy = self.compute_entropy(upper)

                # Evaluate the number of elements in each dataset
                row_count = dataset.shape[0]
                lower_average = row_index[0] / row_count
                upper_average = (row_count - row_index[0]) / row_count

                # Calculate the total entropy
                lower_sum = lower_average * lower_entropy
                upper_sum = upper_average * upper_entropy
                total_entropy = lower_sum + upper_sum

                # Update loop invariants, if appropriate
                if best_entropy > total_entropy:
                    best_entropy = total_entropy
                    best_column = column_index

                    # Calculate the new best split
                    lower_split = column[row_index[0]]
                    upper_split = column[row_index[0] + 1]
                    best_split = (lower_split + upper_split) / 2.0

        # Return the best split, column values
        return (best_split, best_column)

    def generate_confusion_matrix(self, test_set) -> np.ndarray:
        """ Generates a confusion matrix

        Arguments
        ---------
        test_set: np.ndarray
            the test set used to generate the confusion matrix

        Returns
        -------
        confusion_matrix: np.array
            numpy array of size (test_set.shape[0],test_set.shape[0])
        """
        confusion_matrix = np.zeros((4,4))
        predictions = self.predict(test_set[:, :-1])
        actual = test_set[:, -1]
        assert len(actual) == len(predictions)
        for row, col in zip(actual,predictions):
            confusion_matrix[int(row)-1,int(col)-1] += 1
        return confusion_matrix
        
    def draw_segments(self,
            axes: matplotlib.axes,
            origin: typing.List) -> typing.List:

        """ Evaluate segments of a node and its children

        Arguments
        ---------
        origin: typing.List
            the origin of the node

        Returns
        -------
        segments: List
            the line segments comprising the node and its children
        """

        if not self.leaf:
            lower_width = self.lower_branch.width
            upper_width = self.upper_branch.width

            lower_origin = origin.copy()
            lower_origin[0] += (lower_width - self.width) / 2
            lower_origin[1] -= 10

            upper_origin = origin.copy()
            upper_origin[0] += (self.width - upper_width) / 2
            upper_origin[1] -= 10

            lower_root = np.add(lower_origin, [0, 10])
            upper_root = np.add(upper_origin, [0, 10])

            draw.line(axes, lower_root, upper_root)
            draw.line(axes, lower_origin, lower_root)
            draw.line(axes, upper_origin, upper_root)

            self.lower_branch.draw_segments(axes, lower_origin)
            self.upper_branch.draw_segments(axes, upper_origin)

        label_size = draw.label(axes, origin, f'{self.label}')
        draw.box(axes, origin, np.add(label_size, [5, 5]))

    def draw(self):
        """ Plot a node and its children
        """

        figure, axes = pyplot.subplots()
        self.draw_segments(axes, [0, 0])

        axes.autoscale_view(True, True, True)
        pyplot.axis('equal')
        pyplot.axis('off')
        pyplot.show()

    def max_depth(self):
        if self.leaf:
            return self.depth;
        return max(self.lower_branch.max_depth(), self.upper_branch.max_depth())