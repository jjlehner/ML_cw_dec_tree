from decision_tree import *

if __name__ == '__main__':
    # Load both the clean and noisy datasets
    clean_data = data.load_clean()
    noisy_data = data.load_noisy()
    
    # Train and evaluate the tree over both datasets
    print("\n\nStep 3 - Results with clean data:\n")
    tree_clean = decision_tree_learning(clean_data, verbose=True)
    print("\n\nStep 3 - Results with noisy data:\n")
    tree_noisy = decision_tree_learning(noisy_data, verbose=True)

    # Train, prune, and evaluate the tree over both datasets 
    print("\nStep 4 - Pruning results with clean data:")
    pruned_tree_clean = decision_tree_learning(clean_data, pruning=True, verbose=True)
    print("\n\nStep 4 - Pruning results with noisy data:")
    pruned_tree_noisy = decision_tree_learning(noisy_data, pruning=True, verbose=True)
