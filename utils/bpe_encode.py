import torch

from utils.remi import REMI
from utils.vocab import RemiVocab
from utils.utils import chr_except_space
from utils.utils import ord_except_space

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace


class MusicTokenizer:
    """
    Convert MIDI to REMI to BPE tokens
    Set bpe_path as the path of pre-trained BPE vocabulary
    """

    def __init__(self, bpe_path):
        self.remi_vocab = RemiVocab()

        bpe_vocab = Tokenizer(BPE())
        bpe_vocab.pre_tokenizer = Whitespace()
        self.bpe_vocab = bpe_vocab.from_file(bpe_path)

        a = ord(" ")
        b = len(self.remi_vocab)

        self.f_to_byte = lambda x: chr_except_space(x, a, b)
        self.f_to_int = lambda x: ord_except_space(x, chr(b), chr(a))

    def midi2remi(self, file_path):
        # MIDI to REMI
        remi = REMI(file_path)
        events, meta_info = remi.get_remi_events()

        return events, meta_info

    def encode(self, events):
        # encode to BPE tokens
        events = self.remi_vocab.encode(events)
        events = list(map(lambda x: self.f_to_byte(x), events))
        events = self.bpe_vocab.encode("".join(events)).ids

        return events

    def decode(self, events):
        if torch.is_tensor(events):
            events = events.tolist()

        # decode to REMI token
        events = self.bpe_vocab.decode(events, skip_special_tokens=False)
        events = events.replace(" ", "")

        events = list(map(self.f_to_int, events))
        events = self.remi_vocab.decode(events)

        return events

    def add_tokens(self, tokens):
        self.bpe_vocab.add_tokens(tokens)

    def encode_meta(self, events):
        if isinstance(events, str):
            return self.bpe_vocab.encode(events).ids
        elif isinstance(events, list):
            return [self.bpe_vocab.encode(event).ids[0] for event in events]
        else:
            raise Exception("inappropriate data type!")

    def decode_meta(self, events):
        return [self.bpe_vocab.decode([event]) for event in events]
