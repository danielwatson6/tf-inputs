from collections import Counter
import os

import tensorflow as tf

from tf_inputs.core import Input


def string_split(strings, sep=" ", pad_value="<PAD>"):
    """Get a padded string tensor by splitting the given string tensor.

    Args
        strings (Tensor): a 1D Tensor of dtype string.

    Keyword args
        sep (str): split delimiter. If set to '', will split by characters.
        pad_value (str): string to use for padding.

    Returns
        Tensor: a 2D Tensor of dtype string.

    """
    original_shape = tf.shape(strings)
    if len(strings.shape) == 0:
        strings = tf.expand_dims(strings, 0)
    sp = tf.strings.split(strings, sep=sep)  # a SparseTensor.
    return tf.sparse.to_dense(sp, default_value=pad_value)


class Mapping:
    """Type <-> id mapping ops builder."""

    def __init__(self, save_path, data_paths, vocab_size=None, sep=" "):
        self._save_path = save_path
        self._vocab_size = vocab_size
        self._sep = sep

        self._type_to_id = None
        self._id_to_type = None

        if not os.path.isfile(save_path):
            type_counts = Counter()

            for file_path in data_paths:
                with open(file_path) as f:
                    for line in f:
                        if sep == "":
                            line_iter = line
                        else:
                            line_iter = line.split(sep)
                        for token in line_iter:
                            type_counts.update((token,))

            with open(save_path, "w") as f:
                f.write("<PAD>\n<UNK>\n<SOS>\n<EOS>\n")
                if self.vocab_size is None:
                    for type_, _ in type_counts.most_common(self.vocab_size - 4):
                        f.write(type_ + "\n")
                else:
                    for type_, _ in type_counts.most_common():
                        f.write(type_ + "\n")

    @property
    def save_path(self):
        return self._save_path

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def sep(self):
        return self._sep

    @property
    def name(self):
        return os.path.basename(self._save_path)

    def TypeToId(self):
        if self._type_to_id is None:
            # TODO: contrib will not exist in TensorFlow 2.0. Find an equivalent way of
            # doing this.
            # TensorFlow bug: using `reuse=tf.AUTO_REUSE` in `tf.variable_scope` won't
            # reuse the tables, so we have to manually check for their existence to
            # avoid duplicating the mapping in RAM.
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                self._type_to_id = tf.contrib.lookup.index_table_from_file(
                    self.save_path,
                    vocab_size=self.vocab_size,
                    default_value=1,
                    name="type_to_id",
                )

        return lambda x: self._type_to_id.lookup(string_split(x, sep=self.sep))

    def IdToType(self):
        if self._id_to_type is None:
            # TODO: contrib will not exist in TensorFlow 2.0. Find an equivalent way of
            # doing this.
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                self._id_to_type = tf.contrib.lookup.index_to_string_table_from_file(
                    self.save_path,
                    vocab_size=self.vocab_size,
                    default_value="<UNK>",
                    name="id_to_type",
                )

        return self._id_to_type.lookup


# UNDER CONSTRUCTION.
class CharMapping(Mapping):
    """Char <-> id mapping ops builder."""
