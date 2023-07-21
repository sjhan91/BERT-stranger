# BERT-stranger Offical

This repository is the implementation of "BERT-stranger".


## Getting Started

### Environments
* Python 3.8.8
* Ubuntu 20.04.2 LTS
* Read [requirements.txt](/requirements.txt) for other Python libraries

### Results
You can extract multiple loops from MIDI, which are represented as REMI+ (1 \<bar\>, 32 \<tempo\>, 129 \<instrument>, 128 \<pitch\>, 128 \<pitch drum\>, 48 \<position\>, 58 \<duration\>, and 32 \<velocity\>).

### Examples
```python
import torch

from time import time
from loop import BERTStranger

# initialize model with GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# define model
model = BERTStranger(device)

file_path = "./05459b96a64358886fe91ce47a9ac91d.mid"

start_time = time()
results = model.extract_loop(file_path)
print(f"Loop extraction for {time() - start_time:.2f} sec")
```

### Download Pre-trained Model
TBD. You should put the downloaded model on "./model/".

## References
Sangjun Han, Hyeongrae Ihm, Woohyung Lim (LG AI Research)