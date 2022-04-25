#
#  Copyright 2020-2021 Gabriel Gonzalez Castane, Rafael Tolosana Calasanz, Alejandro Calderon Mateos, Iñigo Arejula Aisa
#
#  This file is part of memo_image.
#
#  memo_image is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  memo_image is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with memo_image.  If not, see <http://www.gnu.org/licenses/>.
#


__version__ = '1.5'
__author__  = 'Gabriel Gonzalez Castane, Rafael Tolosana Calasanz, Alejandro Calderon Mateos, Iñigo Arejula Aisa'
__email__   = "gabriel.castane@insight-centre.org, rafaelt.unizar@gmail.com, acaldero@inf.uc3m.es"


import functools, atexit, hashlib
import datetime, time
import numpy as np
import cv2, yaml, imutils

cls_model = "./model/google/classification_classes_ILSVRC2012.txt"

'''
MEMO: reset memoization
'''
def memo_clear():
    memo_info = {}
    return memo_info


'''
MEMO: load from file
'''
def memo_reload(file_name=r'memo.yaml'):
    memo_info = {}

    try:
        with open(file_name, 'r') as file:
            memo_info = yaml.load(file, Loader=yaml.Loader)
    except OSError as e:
        print('lib_image: There is not a "' + file_name + '" file... skipping...')

    return memo_info


'''
MEMO: save to file
'''
def memo_update(memo_info, file_name=r'memo.yaml'):
    with open(file_name, 'w') as file:
        yaml.dump(memo_info, file)


'''
MEMO: memoization decorator
'''
def memo_exact(func):
    @functools.wraps(func)

    def wrapper_memo(*args, **kwargs):
        e1 = cv2.getTickCount()
        image_raw = imutils.resize(args[0], width=1024)
        image_md5 = hashlib.md5(image_raw).hexdigest()
        if image_md5 in wrapper_memo.cache:
           val = wrapper_memo.cache[image_md5]
           e2 = cv2.getTickCount()
           t  = 1000.0 * (e2 - e1) / cv2.getTickFrequency()
           wrapper_memo.time['time'] = t
           wrapper_memo.time['hit']  = 1
           return val

        e1 = cv2.getTickCount()
        wrapper_memo.cache[image_md5] = func(*args, **kwargs)
        e2 = cv2.getTickCount()
        t  = 1000.0 * (e2 - e1) / cv2.getTickFrequency()
        wrapper_memo.time['time'] = t
        wrapper_memo.time['hit']  = 0
        return wrapper_memo.cache[image_md5]

    wrapper_memo.cache = memo_reload('memo.yaml')
    wrapper_memo.time  = {}
    atexit.register(memo_update, wrapper_memo.cache, 'memo.yaml')

    return wrapper_memo



'''
MEMO: memoization decorator
'''
def memo_approx(func):
    @functools.wraps(func)

    def wrapper_memo(*args, **kwargs):
        classes = None
        with open(cls_model, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')
        e1 = cv2.getTickCount()
        image_raw = imutils.resize(args[0], width=1024)

        # get working images
        md5_canonical   = wrapper_memo.time['md5_canonical']
        image_canonical = wrapper_memo.time['image_canonical']
        image_before    = wrapper_memo.time['image_before']
        if image_before is None:
            image_before    = image_raw
            image_canonical = image_raw
            md5_canonical   = hashlib.md5(image_canonical).hexdigest()

       # get image_delta
        image_delta    = cv2.absdiff(image_before, image_raw)

        _, image_delta = cv2.threshold(image_delta, 255-int(args[1]), 255, cv2.THRESH_TOZERO)

        # check if canonical needs to be updated
        wrapper_memo.time['image_before'] = image_raw
        if np.all((image_delta == 0)):
            wrapper_memo.time['image_canonical'] = image_raw
            wrapper_memo.time['md5_canonical']   = hashlib.md5(image_raw).hexdigest()
            image_canonical = image_raw
            md5_canonical   = wrapper_memo.time['md5_canonical']

        #print(image_delta)
        #cv2.imshow('delta', image_delta)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        if md5_canonical in wrapper_memo.cache:
            val                                         = wrapper_memo.cache[md5_canonical]
            e2                                          = cv2.getTickCount()
            t                                           = 1000.0 * (e2 - e1) / cv2.getTickFrequency()
            wrapper_memo.time['time']                   = t
            wrapper_memo.time['hit']                    = 1
            #realclassification = func(*args, **kwargs)
            #wrapper_memo.time['real_label_id']          = realclassification['classId']
            #wrapper_memo.time['real_label']             = classes[realclassification['classId']]
            #wrapper_memo.time['real_label_confidence']  = realclassification['confidence']
            #if (val['classId'] != realclassification['classId']):
            #    wrapper_memo.time['error'] = True
            #else:
            #    wrapper_memo.time['error'] = False
            return val
        #else:
            #wrapper_memo.time['error'] = None
            #wrapper_memo.time['real_label_id']          = 0
            #wrapper_memo.time['real_label']             = None
            #wrapper_memo.time['real_label_confidence']  = None

        #e1 = cv2.getTickCount()
        wrapper_memo.cache[md5_canonical] = func(*args, **kwargs)
        e2 = cv2.getTickCount()
        t  = 1000.0 * (e2 - e1) / cv2.getTickFrequency()
        wrapper_memo.time['time'] = t
        wrapper_memo.time['hit']  = 0
        return wrapper_memo.cache[md5_canonical]

    wrapper_memo.cache = memo_reload('memo.yaml')
    wrapper_memo.time  = {}
    wrapper_memo.time['cannonical_frame'] = 0
    wrapper_memo.time['time'] = 0
    wrapper_memo.time['hit']  = 0
    wrapper_memo.time['image_before']    = None
    wrapper_memo.time['image_canonical'] = None
    wrapper_memo.time['md5_canonical']   = None
    #wrapper_memo.time['error']           = None
    #wrapper_memo.time['real_label']      = None
    #wrapper_memo.time['real_label_id']   = 0
    #wrapper_memo.time['real_label_confidence']   = 0.0

    atexit.register(memo_update, wrapper_memo.cache, 'memo.yaml')

    return wrapper_memo

