from decision_tree import *

if __name__ == '__main__':
    # load both the clean and noisy datasets
    clean_data = data.load_clean()
    noisy_data = data.load_noisy()
    
    #train and test the tree over both datasets
    print("\n\nStep 3 - Results with clean data:\n")
    decision_tree_learning(clean_data, verbose=True)
    print("\n\nStep 3 - Results with noisy data:\n")
    decision_tree_learning(noisy_data, verbose=True)

    print("\nStep 4 - Pruning results with clean data:")
    decision_tree_learning(clean_data, pruning=True, verbose=True)
    print("\n\nStep 4 - Pruning results with noisy data:")
    decision_tree_learning(noisy_data, pruning=True, verbose=True)
    
    



