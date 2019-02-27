import os

DATASET_DIR_TRAIN = "prepared_arc_dataset/train/"
DATASET_DIR_DEV = "prepared_arc_dataset/dev/"
DATASET_DIR_TEST = "prepared_arc_dataset/test/"

choices = [('train', DATASET_DIR_TRAIN), ('dev', DATASET_DIR_DEV), ('test', DATASET_DIR_TEST)]

label_counts_per_set = {}
total_examples_per_set = {}
for (tag, dir) in choices:
    total_examples = 0
    label_counts = {}
    for filename in os.listdir(dir):
        if filename.endswith(".jpg"):
            label = filename.split("-")[0]
            total_examples += 1
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
    label_counts_per_set[tag] = dict(label_counts)
    total_examples_per_set[tag] = total_examples

labels = sorted(list(label_counts_per_set['train'].keys()))
# print("%s set statistics across %d labels." % (tag, len(labels)))
print("Statistics (train, dev, test")
for label in labels:
    print("\t%s, count: (%d, %d, %d), percentage: (%.2f%%, %.2f%%, %.2f%%)" %
            (label,
             label_counts_per_set['train'][label],
             label_counts_per_set['dev'][label],
             label_counts_per_set['test'][label],
             label_counts_per_set['train'][label] * 100.0 / total_examples_per_set['train'],
             label_counts_per_set['dev'][label] * 100.0 / total_examples_per_set['dev'],
             label_counts_per_set['test'][label] * 100.0 / total_examples_per_set['test'])
    )
