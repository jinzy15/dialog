from __future__ import unicode_literals, print_function, division
from Config import *
import transformer.Modules as Modules
from dataset.torchset import VectorchSet
from transformer.Models.EDModels import EDModels
from torch.utils.data import DataLoader
from utils.Normalizer import ch_normalizeAString

from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
import numpy as np
from gensim.models import KeyedVectors