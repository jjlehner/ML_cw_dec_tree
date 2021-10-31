import numpy as np

import data
from decision_tree import DecisionTreeClassifierNode as DTCN

def generate_folds(dataset: np.ndarray, k: int=10, validation: bool=False) -> np.ndarray:
    """ Generates all possibl folds given a dataset

    Arguments
    ---------
    dataset: np.ndarray
        The dataset to generate folds from

    k: int
        The number of folds to produce

    validation: bool
        Whether to produce a validation sets as well as test and train sets
    
    Returns
    -------
    folds_test:
        k (or k*(k-1) if a validation set is requested) folds of testing data samples
    folds_validation:
        If specified, k*(k-1) folds of validation data samples
    folds_training:
        k (or k*(k-1) if a validation set is requested) folds of training data samples
    """
    # suffle the dataset randomly
    rng = np.random.default_rng()
    rng.shuffle(dataset)

    # initialize the lists of folds
    folds_test = list()
    folds_validation = list()
    folds_train = list()

    # compute the size of each fold
    fold_size = int(dataset.shape[0]/k)

    # generate the folds
    if validation:
        for j in range(0, k):
            sub_dataset = dataset[fold_size:-1]
            for i in range(0, k-1):
                # reserve 1 fold for the test dataset (single roll)
                folds_test.append(dataset[0:fold_size])
                # reserve 1 fold for the validation dataset (double roll)
                folds_validation.append(sub_dataset[0:fold_size])
                # add the remaining 8 folds to the training dataset
                folds_train.append(sub_dataset[fold_size:-1])
                # roll the folds for next iteration
                sub_dataset = np.roll(sub_dataset, shift=fold_size, axis=0)
            # roll the folds for next iteration
            dataset = np.roll(dataset, shift=fold_size, axis=0)
        return (np.squeeze(np.asarray(folds_test)), np.squeeze(np.asarray(folds_validation)), np.squeeze(np.asarray(folds_train)))
    else:
        for i in range(0, k):
            # reserve 1 fold for the test dataset
            folds_test.append(dataset[0:fold_size])
            # add the remaining 9 folds to the training dataset
            folds_train.append(dataset[fold_size:-1])
            # roll the folds for next iteration
            dataset = np.roll(dataset, shift=fold_size, axis=0)
        return (np.squeeze(np.asarray(folds_test)), np.squeeze(np.asarray(folds_train)))


def evaluate(test_db: np.ndarray, trained_tree: DTCN) -> float:
    """ Evaluates the tree by returning its accuracy

    Arguments
    ---------
    test_db: np.ndarray
        The dataset used to evaluate the tree against

    trained_tree: DecisionTreeClassifierNode
        The root node of the tree which is being evaluated
    
    Returns
    -------
    output: float
        The tree's accuracy
    """
    return trained_tree.evaluate(test_db)

def decision_tree_learning(dataset: np.ndarray, pruning: bool = False, verbose: bool = False) -> DTCN:
    """ Generates decision tree

    Arguments
    ---------
    dataset: np.ndarray
        The dataset used to train, test and validate the tree

    pruning: bool
        Decides whether to prune the tree using a validation flold in order to potentially incrase 
        accuracy
    
    verbose: bool
        Boolean to print more details on the steps done

    Returns
    -------
    best_tree: DecisionTreeClassifierNode
        A trained tree with the highest level of accuracy on every possible fold
    """
        
    # Number of folds for the cross-validation
    k = 10
    
    # generate said folds
    folds_test = None
    folds_train = None
    folds_validation = None
    if not pruning:
        (folds_test, folds_train) = generate_folds(dataset, k=k)
    else:
        (folds_test, folds_validation, folds_train) = generate_folds(dataset, k=k, validation=True)

    # initialize global metrics
    global_confusion_matrix = np.zeros((4,4))
    global_accuracy = 0
    
    nb_of_folds = folds_train.shape[0]
    if verbose: print(f'Number of cross-validation folds = {nb_of_folds}')
    
    best_tree = None
    best_accuracy = 0.0
    average_depth = 0

    # train and test each set of folds on the dataset
    for i in range(0, nb_of_folds):
        # Training: train the tree on the training dataset
        tree = DTCN(folds_train[i][:-1],0)
        if pruning: tree.prune(folds_validation[i], tree)
        # Testing: generate the confusion matrix over the test dataset
        global_confusion_matrix += tree.generate_confusion_matrix(folds_test[i]) / nb_of_folds
        # evaluate the accuracy of the current fold
        accuracy = evaluate(folds_test[i], tree)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_tree = tree
        if verbose: print(f' - Accuracy of fold {i} = {accuracy}')
        global_accuracy += accuracy / nb_of_folds
        average_depth += tree.max_depth() / nb_of_folds

    # compute the global metrics
    confusion_average = np.trace(global_confusion_matrix) / np.sum(global_confusion_matrix)
    if verbose: print(f'Global Accuracy: {global_accuracy}')
    if verbose: print('\nThe confusion matrix is:')
    if verbose: print(global_confusion_matrix)
    if verbose: print(f'Confusion matrix average accuracy: {confusion_average}') 
    
    # compute the precision, recall and F1 for each class (= each Room)
    precision = list()
    recall = list()
    f1 = list()
    for i in range(0, 4):
        # precision = element at [i,i] over the sum of the ith row
        precision.append(global_confusion_matrix[i,i] / np.sum(global_confusion_matrix[:, i]))
        # recall = element at [i,i] over the sum of the ith column
        recall.append(global_confusion_matrix[i,i] / np.sum(global_confusion_matrix[i]))
        # F1 = 2 x precision x recall / (precision + recall)
        f1.append((2 * precision[i] * recall[i]) / (precision[i] + recall[i]))
        if verbose: print(f" -> Room {i+1}: Precision = {precision[i]}, Recall = {recall[i]}, F1 = {f1[i]}")
    if verbose: print(f"Average Depth of tree: {average_depth}")
    return best_tree

if __name__ == '__main__':
    # load both the clean and noisy datasets
    clean_data = data.load_clean()
    noisy_data = data.load_noisy()
    
    #train and test the tree over both datasets
    print("\n\nStep 3 - Results with clean data:\n")
    decision_tree_learning(clean_data, verbose=True)
    print("\n\nStep 3 - Results with noisy data:\n")
    tree = decision_tree_learning(noisy_data, verbose=True)
    tree.draw()
    print(tree.evaluate(clean_data))

    print("\nStep 4 - Pruning results with clean data:")
    decision_tree_learning(clean_data, pruning=True, verbose=True)
    print("\n\nStep 4 - Pruning results with noisy data:")
    pruned_tree = decision_tree_learning(noisy_data, pruning=True, verbose=True)
    pruned_tree.draw()
    print(pruned_tree.evaluate(clean_data))
    
    



