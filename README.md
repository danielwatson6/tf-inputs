# tf-inputs

This package provides easy-to-write input pipelines for TensorFlow that automatically
integrate with the `tf.data` API.

## Overview

A quick, full example of a training script with an optimized input pipeline:

```python
import tensorflow as tf
import tf_inputs as tfi

# Recursively find all files inside directory and parse them with `parse_fn`.
inputs = tfi.Input.from_directory(
    "/path/to/data_dir", parse_fn=tf.image.decode_png, batch_size=16,
    num_parallel_calls=4
)

# Supposing `my_model_fn` builds the computational graph of some image model.
# Built Keras style-- calling the instance returns the iterator input tensor,
# and until this is done, no ops are added to the computational graph.
train_op, outputs = my_model_fn(inputs())

# Training loop.
with tf.Session().as_default():
    inputs.initialize()  # or `sess.run(inputs.initializer)` is fine too
    while True:
        try:
            inputs.run(train_op)  # replace `sess.run` with `inputs.run`
        except tf.errors.OutOfRangeError:
            break
```

You may still use `sess.run` if you prefer, though we override it to automatically
handle `feed_dict` passing for TensorFlow's feedable iterators and placeholders. If you
need to pass an explicit session you may also use `inputs.run(ops, session=sess)`.

## Installation

`tf-inputs` supports TensorFlow 1.13 and python 3.7. We use no other 3rd party python
modules. Make sure to have your favorite TensorFlow binary installed (i.e., `tensorflow`,
`tensorflow-gpu`, or your own custom wheel built from source) prior to installing
`tf-inputs`.

```
pip install tf-inputs
```

## Switch between training and validation datasets

This can get quite messy with the `tf.data` API. See the
[documentation](https://www.tensorflow.org/guide/datasets#creating_an_iterator)
yourself. `tf-inputs` handles it this way:

```python
train_inputs = tfi.Input.from_directory("/data/training", **options)
valid_inputs = tfi.Input.from_directory("/data/validation", **options)
inputs = tfi.TrainValidInput(train_inputs, valid_inputs)

...

with tf.Session().as_default():
    inputs.initialize()
    inputs.run([train_op, output_op])  # receives a training batch
    inputs.run(output_op, valid=True)  # receives a validation batch
```

If you do not have separate datasets for training and validation, you may use:

```python
inputs = tfi.TrainValidSplit(inputs, num_valid_examples)
```

## Methods to read data

`tf-inputs` supports a variety of ways to read data besides `Input.from_directory`:

```python
# Provide the file paths yourself:
inputs = tfi.Input.from_file_paths(["data/file1.txt", "data/file2.txt"], **options)
```

```python
# Provide the `tf.data.Dataset` instance yourself (yielding single input elements):
inputs = tfi.Input.from_dataset(dataset, **blah)
```

```python
# Same as above, but preventing any graph building a priori:
inputs = tfi.Input.from_dataset_fn(get_dataset, **blah)
```

```python
# Lowest level: subclass `tfi.Input` and override `read_data` to return the dataset:
class MyInput(tfi.Input):
    def __init__(self, my_arg, my_kwarg="foo", **kwargs):
        super().__init__(**kwargs)
        self.my_arg = myarg
        ...
    
    def read_data(self):
        return tf.data.Dataset.from_tensor_slices(list(range(self.my_arg)))
```

Usually there is no need to use the lower level methods. One common example is when the
user wishes to yield `(input, label)` pairs and they live in different files. You may
use `tfi.Zip` for this, as long as the number of elements match:

```python
# Multi task learning training input pipeline.
sentences_en = tfi.Input.from_directory("data/training/english")
sentences_fr = tfi.Input.from_directory("data/training/french")
sentiment_labels = tfi.Input.from_directory("data/training/labels")

inputs = tfi.Zip(images, sentences_fr, sentiment_labels)

def my_model(inputs, training=True):
    if training:
        x, y1, y2 = inputs
    ...
```

## Training over multiple epochs

Just catch the `tf.errors.OutOfRangeError` and restart the iterator:

```python
# Inside a `tf.Session`:
inputs.initialize()
while epoch < max_epochs:
    try:
        inputs.run(train_op)
    except tf.errors.OutOfRangeError:
        inputs.initialize()
        epochs += 1
```

## Multiple elements per file

Just set `flatten=True` flag with `Input.from_directory` or `Input.from_file_paths`:

```python
# Inputs split by an arbitrary delimiter in a text file:
inputs = tfi.Input.from_directory(
    'path/to/text/files', batch_size=32, flatten=True,
    parse_fn=lambda x: tf.string_split(x, delimiter='\n\n'),
)
```
