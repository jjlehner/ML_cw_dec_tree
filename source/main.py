import data
from decision_tree import DecisionTreeClassifierNode as DTCN
import numpy as np


def generate_folds(dataset: np.ndarray, k: int=10, validation: bool=False):
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
            # reserve 1 fold for the test dataset
            #folds_test.append(dataset[0:fold_size])
            sub_dataset = dataset[fold_size:-1]
            for i in range(0, k-1):
                folds_test.append(dataset[0:fold_size])
                # reserve 1 fold for the validation dataset
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


def evaluate(test_db, trained_tree):
    return trained_tree.evaluate(test_db)


def train_and_test(dataset: np.ndarray, pruning: bool = False):

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
        print(folds_validation.shape)

    # initialize global metrics
    globalConfusionMatrix = np.zeros((4,4))
    globalAccuracy = 0
    fold_index = 0              # iteration counter for debugging
    
    nb_of_folds = folds_train.shape[0]
    print(f'nb_of_folds = {nb_of_folds}')

    # train and test each set of folds on the dataset
    for i in range(0, nb_of_folds):
        # Training: train the tree on the training dataset
        treeClean = DTCN(folds_train[i][:-1],0)
        if pruning: treeClean.prune(folds_validation[i], treeClean)
        # Testing: generate the confusion matrix over the test dataset
        globalConfusionMatrix += treeClean.generate_confusion_matrix(folds_test[i]) / nb_of_folds
        # evaluate the accuracy of the current fold
        accuracy = evaluate(folds_test[i], treeClean)
        print(f' - Accuracy of fold {i} = {accuracy}')
        globalAccuracy += accuracy / nb_of_folds

    # compute the global metrics
    confusionAverage = np.trace(globalConfusionMatrix) / np.sum(globalConfusionMatrix)
    print(f"Confusion average accuracy {confusionAverage}, Global Accuracy {globalAccuracy}")
    print(globalConfusionMatrix)
    
    # compute the precision, recall and F1 for each class (= each Room)
    precision = list()
    recall = list()
    f1 = list()
    for i in range(0, 4):
        # precision = element at [i,i] over the sum of the ith row
        precision.append(globalConfusionMatrix[i,i] / np.sum(globalConfusionMatrix[:, i]))
        # recall = element at [i,i] over the sum of the ith column
        recall.append(globalConfusionMatrix[i,i] / np.sum(globalConfusionMatrix[i]))
        # F1 = 2 x precision x recall / (precision + recall)
        f1.append((2 * precision[i] * recall[i]) / (precision[i] + recall[i]))
        print(f" -> Room {i+1}: Precision = {precision[i]}, Recall = {recall[i]}, F1 = {f1[i]}")
    

if __name__ == '__main__':
    # load both the clean and noisy datasets
    clean_data = data.load_clean()
    noisy_data = data.load_noisy()

    #train and test the tree over both datasets
    print("\nStep 3 Results with clean data:")
    train_and_test(clean_data)
    print("\nStep 3 Results with noisy data:")
    train_and_test(noisy_data)

    print("\nStep 4 Pruning results with clean data:")
    train_and_test(clean_data, pruning=True)
    print("\nStep 4 Pruning results with noisy data:")
    train_and_test(noisy_data, pruning=True)

    
    