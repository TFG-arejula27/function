
# pyMemo: python3 image memoization with approximate computing

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Release](https://img.shields.io/badge/stable-3.0-green.svg)](https://github.com/acaldero/pymemo/)


## Table of contents

- [Get started](#get-started)
- [bibliography](#bibliography)


## Get started

1) First you need to install the python3 packages:
```bash
  : Installing packages (basic + opencv + libraries) with sudo... 
     pip3 install numpy yaml hashlib imutils datetime time
     pip3 install opencv
     pip3 install wheel setuptools twine
```

2) Then, clone the proyect:
```bash
  git clone https://github.com/acaldero/pymemo
```

3) Finally, run the example(s):
```bash
  ./do_run.sh
```


## Bibliography

### 1) main_memoimage.py based initially based on:
* Example 'code_122' from https://github.com/JimmyHHua/opencv_tutorials
* Example from https://github.com/spandanpal22/Motion-Detection-Using-OpenCV
* Example from https://realpython.com/primer-on-python-decorators/


### 2) Python3 decorator for memoization:

In https://gist.github.com/Susensio/61f4fee01150caaac1e10fc5f005eb75 is said:

"...Simply using functools.lru_cache won't work because numpy.array is mutable and not hashable..."

Alternative:
```python
from numpy_lru_cache_decorator import np_cache

@np_cache()
def function(array):
    ...
```


### 3) OpenCV 4.0 and DNN:

* Tutorial of OpenCV by examples at https://github.com/JimmyHHua/opencv_tutorials
* Examples of OCV+DNN at https://github.com/opencv/opencv/tree/master/samples/dnn


