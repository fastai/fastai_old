import csv, gc, gzip, os, pickle, shutil, sys, warnings, string
import abc, collections, hashlib, itertools, operator
import math, matplotlib.pyplot as plt, numpy as np, pandas as pd, random
import scipy.stats, scipy.special
import PIL
import abc, collections, hashlib, itertools, operator
import mimetypes, inspect, typing
import html, re, spacy

from abc import abstractmethod, abstractproperty
from collections import abc,  Counter, defaultdict, Iterable, namedtuple, OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import copy, deepcopy
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from fast_progress import master_bar, progress_bar
from functools import partial, reduce
from IPython.core.debugger import set_trace
from matplotlib import patches, patheffects
from numpy import array, cos, exp, log, sin, tan, tanh
from operator import attrgetter, itemgetter
from pathlib import Path
from spacy.symbols import ORTH

#for type annotations
from fast_progress.fast_progress import MasterBar, ProgressBar
from matplotlib.patches import Patch
from numbers import Number
from pandas import Series, DataFrame
from typing import Any, AnyStr, Callable, Collection, Dict, Hashable, Iterator, List, Mapping, NewType, Optional 
from typing import Sequence, Tuple, TypeVar, Union
