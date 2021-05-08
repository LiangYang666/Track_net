import json
import os
import random

data_dir = "../../data/mydata"
ratio = 0.7


if __name__ == "__main__":
    with open(os.path.join(data_dir, 'all.json'), 'r') as f:
        all_labels = json.load(f)
    total = len(all_labels)
    all_labels = list(all_labels.items())
    random.shuffle(all_labels)
    n = int(total*ratio)
    train_labels = dict(all_labels[:n])
    test_labels = dict(all_labels[n:])
    with open(os.path.join(data_dir, 'train.json'), 'w') as f:
        json.dump(train_labels, f, indent=2)
    with open(os.path.join(data_dir, 'test.json'), 'w') as f:
        json.dump(test_labels, f, indent=2)


