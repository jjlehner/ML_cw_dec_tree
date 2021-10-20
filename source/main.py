import data
from decision_tree import DecisionTreeClassifierNode as DTCN
import numpy as np

if __name__ == '__main__':
    clean_data = data.load_clean()
    noisy_data = data.load_noisy()

    rng = np.random.default_rng()
    rng.shuffle(clean_data)
    rng.shuffle(noisy_data)

    train_clean_data = clean_data[:int(len(clean_data)*0.8)]
    test_clean_data = clean_data[int(len(clean_data)*0.8):]

    train_noisy_data = noisy_data[:int(len(noisy_data)*0.8)]
    test_noisy_data = noisy_data[int(len(noisy_data)*0.8):]

    treeClean = DTCN(train_clean_data[:-1],0)
    treeNoisy = DTCN(train_noisy_data[:-1],0)

    print(np.mean(treeClean.predict(test_clean_data[:,:-1])[0]!=test_clean_data[:, -1]))
    print(np.mean(treeNoisy.predict(test_noisy_data[:,:-1])[0]!=test_noisy_data[:, -1]))

    