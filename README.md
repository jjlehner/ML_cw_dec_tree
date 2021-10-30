# Machine Learning Coursework: Decision Trees

<br />

## Installation guide

### Compatibility

This code was developped and tested on Ubuntu 20.04 using Python 3.9.1.

### Installation process

#### Getting the codebase

In order to use the codebase, it first needs to be cloned to your local machine using `$ git clone https://github.com/jjlehner/ML_cw_dec_tree.git` in whatever parent folder you want the codebase to be in.

You can then enter the project folder using `$ cd ML_cw_dec_tree`.

#### Accessing the main script

The main function of the script is in [main.py](/source/main.py). Open this file to write your own script using the existing codebase.

<br />

## Using the program

### Loading training and testing databases

Any dataset can be loaded into the program as an `numpy.ndarray` using 

```python
    my_database = data.load_dataset('path_to_database_file')
```

### Learning process

To generate a decision tree based on a training dataset, use the `decision_tree_learning` function that has the following signature:

```python
    decision_tree_learning(dataset: np.ndarray, pruning: bool = False, verbose: bool = False) -> DTCN
```
Where `DTCN` is the return type of the function, and is a `DecisionTreeClassfierNode`, the root node of the tree. Two optional named parameters are also exposed:

- `pruning`: a boolean that indicates if the function should prune the tree after it is generated.
- `verbose`: a boolean that indicates if the function should print the evaluation metrics of the tree tested with cross-validation on a subset of the training dataset.

For example, the function can therefore be called as follows on our custom dataset:

```python
    my_tree = decision_tree_learning(my_dataset, pruning=True)
```

#### Bonus: Drawing the learned tree

The resulting tree of the learning process can be printed out using the method `draw()` on the `DTCN` (`DecisionTreeClassifierNode`) object as follows:

```python
    my_tree.draw()
```

And this will output a similar image as the one below:

<p align="center">
    <img src="documentation/images/clean-tree.png" alt="Hot plate">
</p>

### Testing the tree

The tree generated from the `decision_tree_learning` function can be evaluated using the `evaluate` function. This function has the following signature:

```python
    evaluate(test_db: np.ndarray, trained_tree: DTCN) -> float:
```

Where `test_db` is an numpy array containing the input test samples, and `trained_tree` takes a `DTCN` object: the root of the decision tree.

This function returns the **accuracy** of the decision tree evaluated on the test dataset.

It can for example be called as follows:

```python
    evaluate(my_test_database, my_tree)
```

### Running the program

The `main.py` script now being completed, the program can be ran by typing `$ python main.py` or `$ python3 main.py` in the terminal, depending on your Python installation.