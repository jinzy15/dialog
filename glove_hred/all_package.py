import torch
import random
import transformer.Modules as Modules
from torch import optim
import utils.Lang as Lang
from dataset.torchset import *
import torch.nn as nn
from Config import *
import transformer.Modules as Modules
from transformer.Models.gloveHRED import gloveHRED
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from gensim.models import KeyedVectors

