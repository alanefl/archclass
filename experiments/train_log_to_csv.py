"""
Converts .log files from training into CSVs
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file', required=True,
                    help="Path to log file")
parser.add_argument('--model_name', required=True,
                    help="The name of your model.")


args = parser.parse_args()

curr_epoch_no = 0

train_accuracy = None
eval_accuracy = None
train_loss = None
eval_loss = None

print(
    "epoch,train_acc_%s,train_loss_%s,eval_acc_%s,eval_loss_%s" % (
        args.model_name, args.model_name, args.model_name, args.model_name
    )
)

with open(args.file, 'r') as f:
    for line in f:
        line = line.strip()[29:].strip()
        if line.startswith("Epoch"):
            if train_accuracy:
                print(",".join([str(curr_epoch_no), train_accuracy, train_loss, eval_accuracy, eval_loss]))
            curr_epoch_no = int(line.split(" ")[1].split("/")[0])
        else:
            if "Train" in line:
                train_accuracy, train_loss = [line[2:].split(":")[2].split(";")[0].strip(), line[2:].split(":")[-1].strip()]
            elif "Eval" in line:
                eval_accuracy, eval_loss = [line[2:].split(":")[2].split(";")[0].strip(), line[2:].split(":")[-1].strip()]
