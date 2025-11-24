import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
import optax
from functools import partial
from typing import Tuple, Dict, NamedTuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
import numpy as np
import math