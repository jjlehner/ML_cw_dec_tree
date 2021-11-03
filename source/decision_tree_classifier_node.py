import math
import typing

import numpy
import sys

import matplotlib
from matplotlib import pyplot

import draw

class Graphics:

    def __init__(self):
        self.label = ''

        self.x = 0
        self.y = 0

        self.width = 0
        self.height = 0

        self.modifier = 0

class DecisionTreeClassifierNode:

    def __init__(self, dataset: numpy.ndarray, depth: int):

        # Initialize branches, label
        self.lower_branch = None
        self.upper_branch = None
        self.parent = None

        # Initialize meta-information
        self.depth: int = depth
        self.leaf = False
        self.split_value = 0
        self.column = 0
        self.width = 0
        self.elements_under_leaf = 0

        self.graphics = Graphics()

        # Evaluate label and unique values in data set
        label, unique_counts = numpy.unique(dataset[:, -1], return_counts=True)
        label_count = len(unique_counts)

        # If only one sample present, flag node as leaf
        if label_count == 1:
            self.leaf = True
            self.width = 10
            self.label = int(label[0])
            self.graphics.label = f'{self.label}'
            self.elements_under_leaf = dataset.shape[0]

        # Otherwise, split the data set and create two sub-branches
        else:
            self.split_value, self.column = self.find_split(dataset)
            self.graphics.label = f'< {int(self.split_value)} ≤'

            lower_dataset = dataset[dataset[:, self.column] < self.split_value]
            upper_dataset = dataset[dataset[:, self.column] > self.split_value]

            self.lower_branch = DecisionTreeClassifierNode(lower_dataset, depth + 1)
            self.upper_branch = DecisionTreeClassifierNode(upper_dataset, depth + 1)

            self.lower_branch.parent = self
            self.upper_branch.parent = self

            self.width += self.lower_branch.width
            self.width += self.upper_branch.width

    def evaluate(self, test_db) -> float:
        """ Evaluates a tree

        Arguments
        ---------
        test_db: numpy.ndarray
            the dataset with which to evaluate the tree gainst

        Returns
        -------
        accuracy: float
            the accuracy of the tree with the test_db testing set
        """

        return numpy.mean(numpy.equal(self.predict(test_db[:,:-1]),test_db[:, -1]))

    def prune(self, validation: numpy.ndarray, root: 'DecisionTreeClassifierNode'):
        """ Reduce overfitting across the tree to increase generalyse to unknown data.
                Change the state of the tree by removing/pruning nodes that decrease the accuracy of the Decision Tree

        Arguments
        ---------
        validation: numpy.ndarray
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
            list of predictions for each row of the associated test dataset
        """
        return numpy.squeeze(numpy.array([self.predict_row(row_index) for row_index in dataset]))

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

    def generate_confusion_matrix(self, test_set) -> numpy.ndarray:
        """ Generates a confusion matrix

        Arguments
        ---------
        test_set: numpy.ndarray
            the test set used to generate the confusion matrix

        Returns
        -------
        confusion_matrix: numpy.array
            numpy array of size (test_set.shape[0],test_set.shape[0])
        """
        confusion_matrix = numpy.zeros((4,4))
        predictions = self.predict(test_set[:, :-1])
        actual = test_set[:, -1]
        assert len(actual) == len(predictions)
        for row, col in zip(actual,predictions):
            confusion_matrix[int(row)-1,int(col)-1] += 1
        return confusion_matrix

    def calculate_positions(self, y: int = 0):

        # Depth-first calculate childrens' positions
        if not self.leaf:
            self.lower_branch.calculate_positions(y - 1)
            self.upper_branch.calculate_positions(y - 1)

        # Update {x, y} positions
        self.graphics.y = y
        if self.parent and self != self.parent.lower_branch:
            self.graphics.x = self.parent.lower_branch.graphics.x + 1
        else:
            self.graphics.x = 0

        # Preclude overlaps
        if not self.leaf:
            left_contour = {}
            right_contour = {}

            self.upper_branch.get_contour('left', left_contour)
            self.lower_branch.get_contour('right', right_contour)

            largest_offset = 0
            for depth in left_contour:
                if depth not in right_contour:
                    continue

                left_index = left_contour[depth]
                right_index = right_contour[depth]

                if right_index >= left_index:
                    offset = right_index - left_index
                    largest_offset = max(largest_offset, offset + 1)
            self.upper_branch.graphics.x += largest_offset
            self.upper_branch.graphics.modifier += largest_offset

        # Centre over children
        if not self.leaf:

            lower_x = self.lower_branch.graphics.x
            upper_x = self.upper_branch.graphics.x
            average = (lower_x + upper_x) / 2

            if self.parent and self != self.parent.lower_branch:
                self.graphics.modifier = self.graphics.x - average
            else:
                self.graphics.x = average

    def get_contour(self, sign: str, contour: typing.Dict, modifier_sum: int = 0):
        """
        """

        # Calculate modifier-offset (non-local) position
        offset = self.graphics.x + modifier_sum

        # Update contour map, either for the left or right, depending on sign
        if self.graphics.y not in contour:
            contour[self.graphics.y] = offset
        else:
            present_contour = contour[self.graphics.y]
            if sign == 'left':
                contour[self.graphics.y] = min(present_contour, offset)
            elif sign == 'right':
                contour[self.graphics.y] = max(present_contour, offset)

        # Add children to contour map
        modifier_sum += self.graphics.modifier
        if not self.leaf:
            self.lower_branch.get_contour(sign, contour, modifier_sum)
            self.upper_branch.get_contour(sign, contour, modifier_sum)

    def draw_segments(self, axes, modifier_sum: int = 0):
        self.graphics.x += modifier_sum
        modifier_sum += self.graphics.modifier

        self.graphics.x *= 5
        self.graphics.y *= 7.5

        origin = [self.graphics.x, self.graphics.y]
        if not self.leaf:
            self.lower_branch.draw_segments(axes, modifier_sum)
            self.upper_branch.draw_segments(axes, modifier_sum)

            draw.line(axes, origin, [self.lower_branch.graphics.x, self.lower_branch.graphics.y])
            draw.line(axes, origin, [self.upper_branch.graphics.x, self.upper_branch.graphics.y])

        label_size = draw.label(axes, origin, self.graphics.label)
        draw.box(axes, origin, numpy.add(label_size, [1.25, 1.25]))


    def draw(self):
        """ Plot a node and its children
        """

        figure, axes = pyplot.subplots()

        self.calculate_positions()
        self.draw_segments(axes)

        axes.autoscale_view(True, True, True)
        pyplot.axis('equal')
        pyplot.axis('off')
        pyplot.show()

    def max_depth(self):
        if self.leaf:
            return self.depth
        return max(self.lower_branch.max_depth(), self.upper_branch.max_depth())
