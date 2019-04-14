from collections import Counter
import os

import tensorflow as tf

from tf_inputs.core import Input


def string_split(strings, sep=" ", pad_value="<PAD>"):
    """Get a padded string tensor by splitting the given string tensor.

    Args
        strings (Tensor): a rank n Tensor of dtype string.

    Keyword args
        sep (str): split delimiter. If set to '', will split by characters.
        pad_value (str): string to use for padding.

    Returns
        Tensor: a rank n+1 Tensor of dtype string.

    """
    if not isinstance(strings, tf.Tensor):
        strings = tf.convert_to_tensor(strings)

    original_shape = strings.shape

    dense_string_split = lambda x: tf.sparse.to_dense(
        tf.strings.split(x, sep=sep), default_value=pad_value
    )

    if len(original_shape) == 0:
        strings = tf.expand_dims(strings, 0)
    if len(original_shape) > 1:
        strings = tf.reshape(strings, [-1])

    result = dense_string_split(strings)

    if len(original_shape) == 0:
        return result[0]
    if len(original_shape) > 1:
        return tf.reshape(result, list(original_shape) + [result.shape[-1]])

    return result


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
                            type_counts.update((token.strip(),))

            with open(save_path, "w") as f:
                f.write("<PAD>\n<UNK>\n<SOS>\n<EOS>\n")
                if self.vocab_size is not None:
                    for type_, count in type_counts.most_common(self.vocab_size - 4):
                        if type_ not in ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]:
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
