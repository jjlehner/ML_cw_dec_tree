import typing

import numpy

import data
import decision_tree


def slice_dataset(dataset) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
    """ Slice a dataset into a testing and training portion

    Arguments
    ---------
    dataset: numpy.ndarray
        the dataset to slice

    Returns
    -------
    training, test: typing.Tuple[numpy.ndarray, numpy.ndarray]
        the training and testing portions
    """

    element_count = len(dataset)

    training_set = dataset[:int(element_count * 0.8)]
    test_set = dataset[int(element_count * 0.8):]

    return (training_set, test_set)


if __name__ == '__main__':

    # Load clean and noisy data sets
    clean_data = data.load_clean()
    noisy_data = data.load_noisy()

    # Create and seed a random number generator, and use it to shuffle the data
    # sets
    random_generator = numpy.random.default_rng()
    random_generator.shuffle(clean_data)
    random_generator.shuffle(noisy_data)

    # Slice data sets into training and test portions
    noisy_train_data, noisy_test_data = slice_dataset(noisy_data)
    clean_train_data, clean_test_data = slice_dataset(clean_data)

    # Create a decision tree from the training data sets
    clean_tree = decision_tree.ClassifierNode(clean_train_data[:-1], 0)
    noisy_tree = decision_tree.ClassifierNode(noisy_train_data[:-1], 0)

    # Make predictions on the testing data sets
    clean_prediction = clean_tree.predict(clean_test_data[:, :-1])
    noisy_prediction = noisy_tree.predict(noisy_test_data[:, :-1])

    # Report accuracy over the testing data
    clean_accuracy = numpy.mean(clean_prediction[0] != clean_test_data[:, -1])
    noisy_accuracy = numpy.mean(noisy_prediction[0] != noisy_test_data[:, -1])
    print(clean_accuracy)
    print(noisy_accuracy)

    clean_tree.draw()
