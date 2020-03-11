# -*- coding: utf-8 -*-

import numpy as np

from typing import List, Tuple


def make_batches(size: int, batch_size: int) -> List[Tuple[int, int]]:
    nb_batch = int(np.ceil(size / float(batch_size)))
    res = [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, nb_batch)]
    return res
