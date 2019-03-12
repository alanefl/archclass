# Deep Architectural Style Classification

This is a project by Paavani Dua, Alan Flores-Lopez, and Alex Wade for Stanford's CS230 (Deep Learning) class.

Make sure to import the tf_metrics library:
```bash
pip install git+https://github.com/guillaumegenthial/tf_metrics.git
```

## Fetching the Dataset

A slightly prepared dataset is stored on Google Drive.  To download the dataset and install
it (unzip it, rename the files, and predictably split them into train/dev/test), run
the following commands:

```bash
$ cd data
$ python build_dataset.py
```

It will take a little while to download and unzip the files.  Keep in mind that the dataset
today is about 1GB in size.  Once the script finishes running, you should see the
`data/prepared_arc_dataset` directory ready with `dev`, `test` and `train` subdirectories,
each containing many images.  As written, the split is 70% train, 20% dev, and 10% test.
Since we don't have too much training data at this point, we want a good
representative amount of the dataset in our dev set.

Dataset credit: https://sites.google.com/site/zhexuutssjtu/projects/arch.

## Prototyping a New Model

The following describes the steps you need to take to create and train a
new model.

### Creating a Model

The way the code is setup, different models are easily pluggable.  All you need to do is
define a function (or functions) in a new python module that take some inputs and
return _logits_.

```
(inputs, params, reuse, is_training) -> [Your New Model] -> (logits)
```

Let's say you want to add a new model called `my_awesome_model`.

1. First add your model's name to the MODELS dict in `constants.py`

    ```python
    MODELS = [
        "multinomial-logistic-regression",
        "cnn-baseline",
        "my_awesome_model"
    ]
    ```
2. Then, register your model in the main model-making routine in `model/model_fn.py`

    In the `model_fn` function, you'll see this chunk of code:

    ```python
    if model == "multinomial-logistic-regression":
        logits = build_multinomial_logistic_regression_model(
            inputs, params, reuse=reuse, is_training=is_training
        )
    elif model == "cnn-baseline":
        logits = build_basic_cnn_model(
            inputs, params, reuse=reuse, is_training=is_training
        )
    else:
        raise ValueError("Unsupported model name: %s" % model)
    ```

    Add your new model to this if-else statement:

     ```python
    if model == "multinomial-logistic-regression":
        logits = build_multinomial_logistic_regression_model(
            inputs, params, reuse=reuse, is_training=is_training
        )
    elif model == "cnn-baseline":
        logits = build_basic_cnn_model(
            inputs, params, reuse=reuse, is_training=is_training
        )
    elif model == "my-awesome-model":  # <----
       logits = build_my_awesome_model(
            inputs, params, reuse=reuse, is_training=is_training
        )
    else:
        raise ValueError("Unsupported model name: %s" % model)
    ```

    In the next step, you will add a new submodule containing your function
    `build_awesome_model`, but don't forget to just go ahead and import it now.
    At the top of this file:

    ```python
    from model.my_awesome_model.model import build_my_awesome_model
    ```

3. Now you need to actually write the model that computes the TensorFlow graph
for your model idea.

    Make a new directory under `model/` called `my_awesome_model` and add to it
    empty files `__init__.py` and `model.py`.  The init file will stay empty,
    and in `model.py` you should define the function `build_my_awesome_model`
    that takes in `inputs, params, reuse=reuse, is_training=is_training` and
    returns logits over the 25 possible architectural styles. This is where
    the bulk of your code will be, and where you should define everything
    that's needed for your model.

    In the end, the new files will be like this:

    - `model/my_awesome_model/model.py`
    - `model/my_awesome_model/__init__.py`

4.  The next and final step is to create an experiment directory for your model.

    Make a new directory: `experiments/my_awesome_model` and create a file called
    `params.json` inside of it.  The `params.json` file contains a dictionary of
    hyperparameters for your model.  To get you started on the parameters you need
    to use, you can check out the format of `experiments/test_model/params.json`
    or `experiments/multinomial_logistic_regression/params.json`. You can also
    extend this JSON file as you like, if your model function requires additional
    hyperparameters.


### Training a Model

Assuming that you've followed the steps in the previous section, you can train
your awesome model with your awesome model parameters with this command:

```bash
python train.py --model_dir experiments/my_awesome_model --model my-awesome-model
```

You'll see training statistics (including performance on train/eval) sets printed
to the terminal. You'll also see that the `experiments/my_awesome_model` directory
has been populated with a bunch of goodies (by default, these are on the `.gitignore`)
and won't be registered with or pushed to GitHub.

## Training On GPU

Coming soon.  We still need to get our AWS credits for this part.

## Hyperparameter Search

To search over hyperparameters, use the following script. Note that the job name should be
unique (i.e. not a name that's been used before). This script might take a while as it is
basically running `train.py` once for each hyperparameter value you want to search for.

```bash
python search_hyperparams.py --job_name learning_test --parent_dir experiments/my_awesome_model
--model my-awesome-model
```

After you have run `search_hyperparams.py`, you can view the results of the testing as follows:

```bash
python synthesize_results.py --parent_dir experiments/my_awesome_model
```

## Test Set Evaluation

Finally, to evaluate your model on the test data, run the following.

```bash
python evaluate.py --model_dir experiments/my_awesome_model --model my-awesome-model
```

This simply reports loss, average accuracy, average precision, recall, and F1 scores (using weighted averages). However, you can display per-class metrics by running the following:

```bash
python evaluate.py --model_dir experiments/my_awesome_model --model my-awesome-model --mode per-class
```

To display a confusion matrix, which is saved as figure.png,
as well as other statistics such as class distribution, run this.

```bash
python evaluate.py --model_dir experiments/my_awesome_model --model my-awesome-model --mode confusion
```

Finally, to get examples of images which the network misclassified, run this. Note: you will need to clear the test_summaries directory between runs.

```bash
python evaluate.py --model_dir experiments/my_awesome_model --model my-awesome-model --mode bad-images
```

Then run this command to display a TensorBoard. You will need to navigate to localhost:6006 to display the page. The images will be under the Images tab.

```bash
python -m tensorboard.main --logdir=experiments/my_awesome_model/test_summaries
```
