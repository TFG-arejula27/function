#
#  Copyright 2020-2021 Gabriel Gonzalez Castane, Rafael Tolosana Calasanz, Alejandro Calderon Mateos, Iñigo Arejula Aisa
#
#  This file is part of memo_time.
#
#  memo_time is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  memo_time is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with memo_time.  If not, see <http://www.gnu.org/licenses/>.
#


__version__ = '1.2'
__author__  = 'Gabriel Gonzalez Castane, Rafael Tolosana Calasanz, Alejandro Calderon Mateos, Iñigo Arejula Aisa'
__email__   = "gabriel.castane@insight-centre.org, rafaelt.unizar@gmail.com, acaldero@inf.uc3m.es"


import functools
import datetime, time
import cv2


'''
TIME: reset measurements
'''
def time_clear(time_info):
    time_info = {}
    return time_info


'''
TIME: time measurement
'''
def time_chronometer(func):
    @functools.wraps(func)

    def memo_wrapper(*args, **kwargs):
        e1 = cv2.getTickCount()
        value = func(*args, **kwargs)
        e2 = cv2.getTickCount()
        t  = 1000.0 * (e2 - e1) / cv2.getTickFrequency()
        memo_wrapper.time['time'] = t
        return value

    memo_wrapper.time  = {}
    return memo_wrapper

