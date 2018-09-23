import re

def strip_fastai(s):  return re.sub(r'^fastai\.', '', s)

