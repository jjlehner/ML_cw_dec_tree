import math
import typing

import numpy
import sys


class ClassifierNode:

    def __init__(self, dataset: numpy.ndarray, depth: int):

        # Initialize branches, label
        self.lower_branch = None
        self.upper_branch = None
        self.label = None

        self.depth: int = depth
        self.leaf = False
        self.split_value = 0
        self.column = 0

        # Evaluate label and unique values in data set
        label, unique_counts = numpy.unique(dataset[:, -1], return_counts=True)
        label_count = len(unique_counts)

        # If only one sample present, flag node as leaf
        if label_count == 1:
            self.leaf = True
            self.label = label

        # Otherwise, split the data set and create two sub-branches
        else:
            self.split_value, self.column = self.find_split(dataset)

            lower_dataset = dataset[dataset[:, self.column] < self.split_value]
            upper_dataset = dataset[dataset[:, self.column] > self.split_value]

            self.lower_branch = ClassifierNode(lower_dataset, depth + 1)
            self.upper_branch = ClassifierNode(upper_dataset, depth + 1)

    def predict_row(self, dataset: numpy.ndarray) -> numpy.ndarray:
        """ Predict the value of the first row in a dataset

        Arguments
        ---------
        dataset: numpy.ndarray


        Returns
        -------
        row_prediction: numpy.ndarray
            the predicted row value
        """

        if self.leaf:
            return self.label

        align_lower = dataset[self.column] < self.split_value
        branch = self.lower_branch if align_lower else self.upper_branch

        return branch.predict_row(dataset)

    def predict(self, dataset: numpy.ndarray) -> numpy.ndarray:
        """ Predict the result for each row in a dataset

        Arguments
        ---------
        dataset: numpy.ndarray
            dataset, the rows of which to predict

        Returns
        -------
        predictions: numpy.ndarray
        """

        return [self.predict_row(row_index) for row_index in dataset]

    def compute_entropy(self, dataset) -> float:
        """ Evaluate the entropy of a dataset

        Arguments
        ---------
        dataset: numpy.ndarray
            dataset, the entropy of which to calculate

        Returns
        -------
        entropy: float
            entropy of the dataset
        """

        # Extract a subarray of unique elements in the dataset
        _, unique_elements = numpy.unique(dataset[:, -1], return_counts=True)

        # Perform
        running_sum = 0
        for element in unique_elements:
            unique_value = float(numpy.sum(unique_elements))
            element_value = float(element)

            fraction = element_value / unique_value
            running_sum -= fraction * math.log2(fraction)

        return running_sum

    def find_split(self, dataset: numpy.ndarray) -> typing.Tuple[float, int]:
        """ Finds the ideal split value in a dataset

        Arguments
        ---------
        dataset: numpy.ndarray
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
            sorted_dataset = dataset[numpy.argsort(dataset[:, column_index])]
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
