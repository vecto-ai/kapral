import logging
import os
from collections import namedtuple

import numpy as np
from kapral.utils.data import get_uncompressed_size
from vecto.utils.metadata import WithMetaData

from .iterators import (EOD, CharIterator, DirIterator, FileIterator,
                        FileLineIterator, LoopedLineIterator, SentenceIterator,
                        SequenceIterator, SlidingWindowIterator, TokenIterator,
                        TokenizedSequenceIterator, ViewLineIterator)
from .tokenization import (DEFAULT_JAP_TOKENIZER, DEFAULT_SENT_TOKENIZER,
                           DEFAULT_TOKENIZER)

logger = logging.getLogger(__name__)


TreeElement = namedtuple('TreeElement', ["filename", "bytes"])


class BaseCorpus:
    """Cepresents a body of text in single or multiple files"""

    def get_sliding_window_iterator(self,
                                    left_ctx_size=2,
                                    right_ctx_size=2,
                                    tokenizer=None,
                                    verbose=0):
        if tokenizer is None:
            if self.language == 'jap':
                tokenizer = DEFAULT_JAP_TOKENIZER
            else:
                tokenizer = DEFAULT_TOKENIZER
        return SlidingWindowIterator(
            self.get_tokenized_line_iterator(tokenizer=tokenizer),
            left_ctx_size=left_ctx_size,
            right_ctx_size=right_ctx_size)

    def get_token_iterator(self, tokenizer=None, verbose=False):
        if tokenizer is None:
            if self.language == 'jap':
                tokenizer = DEFAULT_JAP_TOKENIZER
            else:
                tokenizer = DEFAULT_TOKENIZER
        return TokenIterator(self.get_tokenized_line_iterator(tokenizer, verbose))

    def get_character_iterator(self, verbose=False, yield_eod=False):
        return CharIterator(self.get_line_iterator(verbose=verbose, yield_eod=yield_eod))

    def get_tokenized_line_iterator(self, tokenizer=None, verbose=False):
        if tokenizer is None:
            if self.language == 'jap':
                tokenizer = DEFAULT_JAP_TOKENIZER
            else:
                tokenizer = DEFAULT_SENT_TOKENIZER
        return TokenizedSequenceIterator(self.get_line_iterator(verbose=verbose),
                                         tokenizer=tokenizer,
                                         verbose=verbose)

    def get_sentence_iterator(self, tokenizer=None, verbose=False):
        return SentenceIterator(self.get_character_iterator())

    def get_sequence_iterator(self, sequence_length, tokenizer):
        return SequenceIterator(self.get_line_iterator(),
                                sequence_length=sequence_length,
                                tokenizer=tokenizer)

    def get_looped_sequence_iterator(self, sequence_length, tokenizer, rank, size, min_length=0, reset_on_new_line=False):
        return SequenceIterator(self.get_looped_line_iterator(rank, size),
                                sequence_length=sequence_length,
                                tokenizer=tokenizer,
                                minimal_length=min_length,
                                reset_on_new_line=reset_on_new_line)


class Corpus(BaseCorpus, WithMetaData):

    def __init__(self, path, language='eng'):
        super().__init__(path)
        self.path = path
        self.language = language

    def load_dir_strucute(self):
        self.tree = []
        accumulated_size = 0
        for file in DirIterator(self.path):
            accumulated_size += get_uncompressed_size(file)
            self.tree.append(TreeElement(file, accumulated_size))
        self.metadata["total_bytes"] = self.total_bytes
        # print(self.tree)
        # TODO: use named tuples here
        # self.tree = [TreeElement("file1", 10), TreeElement("file2", 15)]

    @property
    def total_bytes(self):
        return self.tree[-1].bytes

    def get_file_and_offset(self, global_position, start_of_range=True, epsilon=0):
        assert global_position <= self.total_bytes
        lo = 0
        hi = len(self.tree)
        while (True):
            current = (lo + hi) // 2
            # print(f"lo {lo}, hi {hi}, pos {pos}")
            if lo >= hi:
                if current > 0:
                    offset = max(global_position - self.tree[current - 1].bytes, 0)
                else:
                    offset = global_position
                if start_of_range:
                    if self.tree[current].bytes - global_position < epsilon:
                        if current < len(self.tree) - 1:
                            current += 1
                            offset = 0
                else:
                    if current > 0:
                        if offset < epsilon:
                            offset = self.tree[current - 1].bytes - (self.tree[current - 2].bytes if current > 1 else 0)
                            current -= 1
                return current, offset

            if self.tree[current].bytes >= global_position:
                hi = current
            if self.tree[current].bytes < global_position:
                lo = current + 1

    def get_line_iterator(self, verbose=False, yield_eod=False):
        dir_iter = DirIterator(self.path, verbose=verbose)
        return FileLineIterator(dir_iter, yield_eod=yield_eod)

    def get_looped_line_iterator(self, rank=0, size=1):
        assert rank < size
        byte_start = self.total_bytes * rank // size
        node_start = self.get_file_and_offset(byte_start, start_of_range=True, epsilon=0)
        iterator = LoopedLineIterator(self.tree, node_start)
        return iterator

    def get_document_iterator(self):
        dir_iter = DirIterator(self.path, verbose=False)
        line_iter = FileLineIterator(dir_iter, yield_eod=True)
        last_doc = None
        # TODO: this yelds one empty doc at the end of the corpus
        while True:
            if last_doc != None:
                if last_doc.reached_eoc:
                    break
                    # raise StopIteration
                assert last_doc.reached_eod, "Can't yield new document until the previous one exhausts its line iterator"
            last_doc = Document(line_iter)
            yield last_doc


class CorpusView(BaseCorpus):
    def __init__(self, file_corpus, rank, size):
        assert rank < size
        self.corpus = file_corpus
        self.rank = rank
        self.size = size

    def get_line_iterator(self, verbose=False):
        byte_start, byte_end = self.rank_and_size_to_pos(self.rank, self.size)
        # TODO: read epsilon from config ^_^
        node_start = self.corpus.get_file_and_offset(byte_start, start_of_range=True, epsilon=0)
        node_end = self.corpus.get_file_and_offset(byte_end, start_of_range=False, epsilon=0)
        # CREATE ITERATOR HERE
        # iterate over precomputed tree of files and sizes
        # iterated this file/this offset to last-file last offset
        iterator = ViewLineIterator(self.corpus.tree, verbose=False, start=node_start, end=node_end)
        return iterator

    def rank_and_size_to_pos(self, rank, size):
        assert rank < size
        start = self.corpus.total_bytes * rank // size
        end = self.corpus.total_bytes * (rank + 1) // size
        return start, end


# TODO: make this deprecated and use Corpus instead
# class FileCorpus(BaseCorpus):
#     """Cepresents a body of text in a single file"""

#     def get_line_iterator(self, verbose=False):
#         return FileLineIterator(FileIterator(self.path, verbose=verbose))


# class DirCorpus(BaseCorpus):
#     """Cepresents a body of text in a directory"""

#     def get_line_iterator(self, verbose=False):
#         return FileLineIterator(DirIterator(self.path, verbose=verbose))


# old code below ----------------------------------


# def FileSlidingWindowCorpus(path, left_ctx_size=2, right_ctx_size=2, tokenizer=DEFAULT_TOKENIZER, verbose=0):
#    """
#    Reads text from `path` line-by-line, splits each line into tokens and/or sentences (depending on tokenizer),
#    and yields training samples for prediction-based distributional semantic models (like Word2Vec etc).
#    Example of one yielded value: {'current': 'long', 'context': ['family', 'dashwood', 'settled', 'sussex']}
#    :param path: text file to read (can be archived)
#    :param tokenizer: tokenizer to use to split into sentences and tokens
#    :param verbose: whether to enable progressbar or not
#    :return:
#    """
#    return SlidingWindowIterator(
#        TokenizedSequenceIterator(
#            FileLineIterator(
#                FileIterator(path, verbose=verbose)),
#            tokenizer=tokenizer),
#        left_ctx_size=left_ctx_size,
#        right_ctx_size=right_ctx_size)


def DirSlidingWindowCorpus(path, left_ctx_size=2, right_ctx_size=2, tokenizer=DEFAULT_TOKENIZER, verbose=0):
    """
    Reads text from all files from all subfolders of `path` line-by-line,
    splits each line into tokens and/or sentences (depending on `tokenizer`),
    and yields training samples for prediction-based distributional semantic models (like Word2Vec etc).
    Example of one yielded value: {'current': 'long', 'context': ['family', 'dashwood', 'settled', 'sussex']}
    :param path: text file to read (can be archived)
    :param tokenizer: tokenizer to use to split into sentences and tokens
    :param verbose: whether to enable progressbar or not
    :return:
    """
    return SlidingWindowIterator(
        TokenizedSequenceIterator(
            FileLineIterator(
                DirIterator(path, verbose=verbose)),
            tokenizer=tokenizer),
        left_ctx_size=left_ctx_size,
        right_ctx_size=right_ctx_size)


def corpus_chain(*corpuses):
    """
    Join all copuses into a big single one. Like `itertools.chain`, but with proper metadata handling.
    :param corpuses: other corpuses or iterators
    :return:
    """
    return IteratorChain(corpuses)


# TODO: make it a part of Corpus class
def load_path_as_ids(path, vocabulary, tokenizer=DEFAULT_TOKENIZER):
    # use proper tokenizer from cooc
    # options to ignore sentence bounbdaries
    # specify what to do with missing words
    # replace numbers with special tokens
    result = []
    if os.path.isfile(path):
        # TODO: why file corpus does not need language? 
        ti = Corpus(path).get_token_iterator(tokenizer=tokenizer)
    else:
        if os.path.isdir(path):
            ti = DirCorpus(path).get_token_iterator(tokenizer)
        else:
            raise RuntimeError("source file does not exist")
    for token in ti:
        w = token  # specify what to do with missing words
        result.append(vocabulary.get_id(w))
    return np.array(result, dtype=np.int32)


class Document(BaseCorpus):
    def __init__(self, line_iter):
        self.line_iter = line_iter
        self.reached_eod = False
        self.reached_eoc = False

    def line_iter_wrapper(self, **kwargs):
        for line in self.line_iter:
            # while True:
            # line = next(self.line_iter)
            if line is EOD:
                self.reached_eod = True
                break
            yield line
        else:
            self.reached_eoc = True

    def get_line_iterator(self, **kwargs):
        return self.line_iter_wrapper(**kwargs)
