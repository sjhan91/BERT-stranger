import json
import torch

from time import time
from BERT import BERT_Lightning

from torch import nn
from torch.nn.utils.rnn import pad_sequence

from utils.vocab import *
from utils.constants import *
from utils.bpe_encode import MusicTokenizer


class BERTStranger:
    def __init__(self, device):
        # load BERT-stranger model
        with open("./config/config.json", "r") as f:
            config = json.load(f)

        bpe_path = "./tokenizer/tokenizer.json"
        bpe_meta_path = "./tokenizer/tokenizer_meta.json"

        # define model
        model = BERT_Lightning(
            dim=config["dim"],
            depth=config["depth"],
            heads=config["heads"],
            dim_head=int(config["dim"] / config["heads"]),
            mlp_dim=int(4 * config["dim"]),
            max_len=config["max_len"],
            rate=config["rate"],
            bpe_path=bpe_path,
        )

        model = model.load_from_checkpoint(
            "./model/BERT-stranger.ckpt",
            dim=config["dim"],
            depth=config["depth"],
            heads=config["heads"],
            dim_head=int(config["dim"] / config["heads"]),
            mlp_dim=int(4 * config["dim"]),
            max_len=config["max_len"],
            rate=config["rate"],
            bpe_path=bpe_path,
        ).to(device)

        self.model = model
        self.device = device
        self.tokenizer = MusicTokenizer(bpe_meta_path)

    def get_bars(self, events):
        return [i for i, event in enumerate(events) if f"{BAR_KEY}_" in event]

    def get_ar(self, P, t, L, j):
        count = 0
        for l in range(L - j - 1):
            count += P[t][l] * P[t][l + j]

        return count

    def get_mean_token(self, tokens, KEY):
        value = [int(token.split("_")[-1]) for token in tokens if KEY in token]
        value = str(int(np.around(np.mean(value))))

        return [KEY + "_" + value]

    def extract_loop(self, file_path):
        file_name = file_path.split("/")[-1].split(".")[0]

        # MIDI to REMI
        events, meta_info = self.tokenizer.midi2remi(file_path)

        # get bars
        bars = self.get_bars(events)

        contexts = list(zip(bars[:-1], bars[1:])) + [(bars[-1], len(events))]
        contexts = [
            (start, end)
            if (end - start) <= (MAX_TOKEN_LEN - 1)
            else (start, start + (MAX_TOKEN_LEN - 1))
            for (start, end) in contexts
        ]

        music = []
        for j, (start, end) in enumerate(contexts):
            bar = events[start:end] + [EOB_TOKEN]

            # REMI to BPE tokens
            bar = self.tokenizer.encode(bar)
            bar = torch.tensor(bar, dtype=torch.long).to(self.device)
            music.append(bar)

        pad_idx = RemiVocab().to_i(PAD_TOKEN)
        music = pad_sequence(music, batch_first=True, padding_value=pad_idx)

        # inference
        self.model.eval()
        with torch.no_grad():
            _, h = self.model(music)

        # normalize in music space
        h = (h - h.mean(0)) / h.std(0)

        # get cosine similarity
        SD = torch.corrcoef(h)

        # get time-lag matrix
        L = torch.zeros(SD.shape)

        for i in range(L.shape[0]):
            L[i:, i] = torch.diagonal(SD, -i)

        # get full probaboloty-lag matrix
        P = torch.zeros(L.shape)

        for t in range(L.shape[0]):
            denom = (t + 1) - torch.arange(L.shape[0])

            P[t, :] = L[: t + 1, :].sum(0) / denom
            P[t, t:] = 0

        # get autocorrelation function
        loop_idx = []
        for t in range(P.shape[0]):
            temp = []
            # set maximum lag as 16
            for j in range(16 + 1):
                value = self.get_ar(P, t, P.shape[1], j)

                if torch.is_tensor(value):
                    value = value.item()

                temp.append(value)

            lag_peak = np.argsort(temp)[::-1][1]

            if lag_peak % 4 == 0 and t >= lag_peak:
                loop_idx.append([t - lag_peak, t])

        # empty loop check
        if len(loop_idx) == 0:
            raise ValueError("No extracted loop")

        results = []
        # extract condition and loop
        for j, (start, end) in enumerate(loop_idx):
            bar_length = end - start

            start_idx = contexts[start:end][0][0]
            end_idx = contexts[start:end][-1][-1]

            # 0. loop
            loop = events[start_idx:end_idx] + [EOB_TOKEN]

            # 1. inst
            inst = list(set([token for token in loop if INSTRUMENT_KEY in token]))

            meta_results = []
            # mean_pitch, mean_tempo, mean_velocity, mean_duration
            for key in [PITCH_KEY, TEMPO_KEY, VELOCITY_KEY, DURATION_KEY]:
                value = self.get_mean_token(loop, key)
                meta_results.append(value)

            # 6. density
            density = meta_info["groove_pattern"][start:end]
            density = np.mean(list(map(lambda x: len(x), density)))
            density = ["Density_" + str(int(np.around(density)))]

            # 7. chord
            chord = meta_info["chord"][start:end]
            chord = [item[0] for item in chord if len(item) > 0]

            # 8. bar_length
            bar_length = ["Length_" + str(bar_length)]

            results.append(
                {
                    "loop": loop,
                    "inst": inst,
                    "mean_pitch": meta_results[0],
                    "mean_tempo": meta_results[1],
                    "mean_velocity": meta_results[2],
                    "mean_duration": meta_results[3],
                    "density": density,
                    "chord": chord,
                    "bar_length": bar_length,
                    "file_name": file_name,
                }
            )

        return results
