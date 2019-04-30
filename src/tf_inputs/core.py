import copy
import json
import os

import tensorflow as tf


class Input:
    """Input pipeline."""

    def __init__(
        self,
        batch_size=16,
        num_parallel_calls=1,
        prefetch_size=1,
        preprocess_fn=None,
        preprocess_batch_fn=None,
        shuffle_size=10000,
        name="input_pipeline",
    ):
        """Constructor.

        Keyword args
            num_parallel_calls (int): number of CPU cores to work in parallel when
                applying `preprocess_fn` to the inputs. TensorFlow recommends setting
                this to the number of available CPU cores.
            prefetch_size (int): number of batches to prefetch in memory. Set to 0 for
                no prefetch-- though this defeats the point of having a multithreaded
                input pipeline.
            preprocess_fn (function): function to preprocess individual inputs prior to
                batching. This should manipulate Tensors. If native python is required,
                wrap the logic with `tf.py_func`.
            preprocess_batch_fn (function): function to preprocess input batches. This
                should manipulate Tensors. If native python is required, wrap the logic
                with `tf.py_func`.
            shuffle_size (int): number of examples per shuffling group. Set to 0 for no
                shuffling.
            name (str): name scope for the input pipeline.

        """
        self._batch_size = batch_size
        self._num_parallel_calls = num_parallel_calls
        self._prefetch_size = prefetch_size
        self._preprocess_fn = preprocess_fn
        self._preprocess_batch_fn = preprocess_batch_fn
        self._shuffle_size = shuffle_size
        self._name = name

        self._dataset = None
        self._iterator = None
        self._input_tensor = None

    # Alternative ways of instantiating an input pipeline without subclassing.

    @classmethod
    def from_directory(cls, dir_path, **kwargs):
        """Create an input pipeline whose elements are the contents of all files found.

        This will recursively read directories to find all files and will open them.
        All keyword args are passed to `Input.from_file_paths`.

        Args
            dir_path (str): path to the directory to recursively search.

        Returns
            Input: an input pipeline.

        """

        file_paths = []
        for root, _, filenames in os.walk(dir_path):
            for filename in filenames:
                file_paths.append(os.path.join(root, filename))

        return cls.from_file_paths(file_paths, **kwargs)

    @classmethod
    def from_file_paths(cls, file_paths, parse_fn=None, flatten=False, **kwargs):
        """Create an input pipeline whose elements are the contents of the given files.

        If `parse_fn` is provided, it will map the file contents. All other keyword
        args are passed to the constructor.

        Args
            dir_path (Iterable): an iterable of strings containing all the file paths.

        Keyword args
            parse_fn (function): function to parse individual file contents. This
                should manipulate Tensors. If native python is required, wrap the
                logic with `tf.py_func`. Note this shouldn't be mixed with or replaced
                by `preprocess_fn` since the latter is also used when the model is used
                in production, i.e., inputs are fed directly rather than from files.
            flatten (bool): if set to True, each file will yield its outermost
                dimension number of elements, rather than a single element.

        Returns
            Input: an input pipeline.

        """

        def map_fn(filename_tensor):
            file_contents = tf.io.read_file(filename_tensor)
            if parse_fn is not None:
                file_contents = parse_fn(file_contents)
            return file_contents

        def dataset_fn():
            dataset = tf.data.Dataset.from_tensor_slices(file_paths)
            if not "num_parallel_calls" in kwargs:
                kwargs["num_parallel_calls"] = 1
            if flatten:
                # Interleave with cycle_length=1 is exactly a flat map, but allows to
                # specify the nummber of parallel calls.
                return dataset.interleave(
                    lambda f: tf.data.Dataset.from_tensor_slices(map_fn(f)),
                    1,
                    num_parallel_calls=kwargs["num_parallel_calls"],
                )

            return dataset.map(map_fn, num_parallel_calls=kwargs["num_parallel_calls"])

        return cls.from_dataset_fn(dataset_fn, **kwargs)

    @classmethod
    def from_dataset(cls, dataset, **kwargs):
        """Create an input pipeline from a `tf.data.Dataset` instance."""
        dataset_fn = lambda: dataset
        return cls.from_dataset_fn(dataset_fn, **kwargs)

    @classmethod
    def from_dataset_fn(cls, dataset_fn, **kwargs):
        self = cls(**kwargs)
        self.read_data = dataset_fn
        return self

    @property
    def name(self):
        return self._name

    @property
    def dataset(self):
        return self._dataset

    @property
    def iterator(self):
        return self._iterator

    @property
    def initializer(self):
        return self.iterator.initializer

    # Avoid building in the graph unless it is accessed.
    @property
    def input_tensor(self):
        if self._input_tensor is None:
            try:
                self._input_tensor = self.iterator.get_next(name="input_tensor")
            except AttributeError:
                raise AttributeError(
                    "Attempting to get the `input_tensor` of an input pipeline that "
                    "hasn't been built."
                )
        return self._input_tensor

    def read_data(self):
        """Get a `tf.data.Dataset` instance from the raw data.

        Note that TFBuild handles shuffling, batching, preprocessing, and prefetching,
        so the returned dataset instance from this method only needs to read the data.

        Returns
            tf.data.Dataset: the dataset instance whose elements represent
                individual dataset elements (i.e., not batched/preprocessed/etc).

        """
        raise NotImplementedError

    def _build_dataset(self, dataset):
        """Finish building the `tf.data.Dataset` instance.

        This method adds shuffling, batching, and preprocessing to the dataset.

        Args
            dataset (tf.data.Dataset): a `tf.data.Dataset` whose elements represent
                individual dataset elements (i.e., not batched/preprocessed/etc).

        """
        if self._shuffle_size:
            self._dataset = dataset.shuffle(self._shuffle_size)

        if self._preprocess_fn is None:
            self._dataset = self._dataset.batch(self._batch_size)
        else:
            self._dataset = self._dataset.apply(
                tf.data.experimental.map_and_batch(
                    self._preprocess_fn,
                    self._batch_size,
                    num_parallel_calls=self._num_parallel_calls,
                )
            )
        if self._preprocess_batch_fn is not None:
            self._dataset = self._dataset.map(
                self._preprocess_batch_fn, num_parallel_calls=self._num_parallel_calls
            )

    def __call__(self):
        """Build the iterator and add prefetching ops to the dataset.

        Keyword args
            name (str): the name scope for the input pipeline ops.

        """
        with tf.name_scope(self.name):
            self._build_dataset(self.read_data())
            self._dataset = self._dataset.prefetch(self._prefetch_size)
            self._iterator = self.dataset.make_initializable_iterator()
            return self.input_tensor

    def initialize(self, session=None):
        """Initialize or reinitialize the input pipeline.

        Keyword args
            session (tf.Session): a TensorFlow session.

        """
        if session is None:
            session = tf.get_default_session()
        session.run(self.initializer)

    def run(self, ops, session=None, **kwargs):
        """Run the specified graph operations.

        Args
            ops: a TensorFlow op, or a list of TensorFlow ops.

        Keyword args
            session (tf.Session): a TensorFlow session.

        Returns
            The returned values after running each op.

        """
        if session is None:
            session = tf.get_default_session()
        return session.run(ops, **kwargs)


class TrainValidInput(Input):
    """Top-level input pipeline supporting training and validation switches."""

    def __init__(self, train_input, valid_input, **kwargs):
        """Constructor.

        Args
            train_input (Input): training input pipeline.
            valid_input (Input): validation input pipeline.

        """
        super().__init__(**kwargs)
        self._train_input = train_input
        self._valid_input = valid_input
        self._handle = None
        self._train_handle = None
        self._valid_handle = None

    @property
    def dataset(self):
        raise AttributeError(
            "`TrainValidInput` instances do not have a `dataset` attribute."
        )

    @property
    def initializer(self):
        return [self.train_input.initializer, self.valid_input.initializer]

    @property
    def train_input(self):
        return self._train_input

    @property
    def valid_input(self):
        return self._valid_input

    @property
    def handle(self):
        return self._handle

    # Make sure to only set the handles when inside a session, and only once.

    def get_train_handle(self, session=None):
        if self._train_handle is None:
            if session is None:
                session = tf.get_default_session()
            self._train_handle = session.run(self.train_input.iterator.string_handle())
        return self._train_handle

    def get_valid_handle(self, session=None):
        if self._valid_handle is None:
            if session is None:
                session = tf.get_default_session()
            self._valid_handle = session.run(self.valid_input.iterator.string_handle())
        return self._valid_handle

    # The overridables should NOT be called; the datasets fed should read the data and
    # override these on their own.

    def read_data(self):
        raise AttributeError("`TrainValidInput` instances have no `read_data` method.")

    def _build_dataset(self, dataset):
        raise AttributeError(
            "`TrainValidInput` instances have no `_build_dataset` method."
        )

    def __call__(self):
        """Build the iterator and add prefetching to the dataset."""
        with tf.name_scope(self.name):
            # Build the train and validation inputs.
            self.train_input()
            self.valid_input()
            self._handle = tf.placeholder(tf.string, shape=[], name="string_handle")
            self._iterator = tf.data.Iterator.from_string_handle(
                self.handle,
                self.train_input.dataset.output_types,
                self.train_input.dataset.output_shapes,
            )
            return self.input_tensor

    def run(self, ops, valid=False, **kwargs):
        """Run the specified graph operations for the desired input pipeline.

        Args
            ops: a TensorFlow op, or a list of TensorFlow ops.

        Keyword args
            valid (bool): set to True to use the validation input pipeline.

        Returns
            The returned values after running each op.

        """
        _run = lambda h: super(TrainValidInput, self).run(
            ops, feed_dict={self.handle: h}, **kwargs
        )
        if valid:
            # If the validation iterator is exhausted, reset it and run normally.
            try:
                return _run(self.get_valid_handle())
            except tf.errors.OutOfRangeError:
                super().run(self.valid_input.initializer, **kwargs)
                return _run(self.get_valid_handle())

        return _run(self.get_train_handle())


class TrainValidSplit(TrainValidInput):
    """Create a `TrainValidInput` instance by splitting a single input."""

    def __init__(self, input_pipeline, num_valid, **kwargs):
        """Constructor.

        Args
            input_pipeline (Input): input pipeline to split.

        """
        train_input = input_pipeline
        valid_input = copy.deepcopy(input_pipeline)

        old_train_build_dataset = train_input._build_dataset
        old_valid_build_dataset = valid_input._build_dataset

        def train_build_dataset(this, dataset):
            """Build the full dataset."""
            dataset = dataset.skip(num_valid)
            old_train_build_dataset(dataset)

        def valid_build_dataset(this, dataset):
            """Build the full dataset."""
            dataset = dataset.take(num_valid)
            old_valid_build_dataset(dataset)

        train_input._build_dataset = train_build_dataset.__get__(train_input)
        valid_input._build_dataset = valid_build_dataset.__get__(valid_input)
        super().__init__(train_input, valid_input, **kwargs)


class FeedableInput(Input):
    """Input pipeline based on TensorFlow's `feed_dict` mechanism.

    Note this is a very suboptimal pipeline for iterating over a dataset. The
    intended use case is to have a quick solution to run the model over provided inputs.

    """

    def __init__(self, placeholders_fn, **kwargs):
        """Constructor.

        Args
            placeholders_fn (function): a method taking no arguments that will
                return a tuple of `tf.placeholder` instances or a single placeholder.

        """
        super().__init__(**kwargs)
        self._placeholders_fn = placeholders_fn

    @property
    def dataset(self):
        raise AttributeError("`FeedableInput` instances have no `dataset` attribute.")

    @property
    def iterator(self):
        raise AttributeError("`FeedableInput` instances have no `iterator` attribute.")

    # Overriden to avoid the `tf.data` API usage.
    @property
    def initializer(self):
        return tf.no_op

    @property
    def input_tensor(self):
        if self._input_tensor is None:
            try:
                self._input_tensor = self._placeholders
                if self._preprocess_fn is not None:
                    self._input_tensor = self._preprocess_fn(self._input_tensor)
            except AttributeError:
                raise AttributeError(
                    "Attempting to get the `input_tensor` of an input pipeline that "
                    "hasn't been built."
                )
        return self._input_tensor

    def read_data(self):
        raise AttributeError("`FeedableInput` instances have no `read_data` method.")

    def _build_dataset(self, dataset):
        raise AttributeError(
            "`FeedableInput` instances have no `_build_dataset` method."
        )

    def __call__(self):
        with tf.name_scope(self.name):
            placeholders = self._placeholders_fn()
            if not isinstance(placeholders, tuple):
                placeholders = (placeholders,)
            self._placeholders = placeholders
            return self.input_tensor

    # Overriden to avoid the `tf.data` API usage.
    def run(self, ops, inputs, **kwargs):
        feed_dict = {p: v for p, v in zip(self._placeholders, inputs)}
        return super().run(ops, feed_dict=feed_dict, **kwargs)


class Zip(Input):
    """Input pipeline that yields examples of multiple other pipelines.

    IMPORTANT: the zipped input pipelines must yield the same total number of elements.

    IMPORTANT: the zipped input pipelines' batch sizes will be overriden to the batch
    size of the `Zip` instance.

    """

    def __init__(self, inputs, **kwargs):
        """Constructor.

        Args
            inputs (tuple): a list of `Input` instances, all yielding the same number
                of elements.

        """
        super().__init__(**kwargs)
        self._inputs = inputs

    def read_data(self):
        datasets = []
        for _input in self._inputs:
            _input._batch_size = self._batch_size
            # Ordering matters when zipping, so shuffling can be set in the instance.
            _input._shuffle_size = 0
            _input._build_dataset(_input.read_data())
            datasets.append(_input.dataset)
        return tf.data.Dataset.zip(tuple(datasets))
