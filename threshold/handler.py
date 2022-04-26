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


#
# Imports
#

import cv2
import numpy as np
from imutils.video import VideoStream
import imutils
import sys, getopt, argparse
from   function.lib_memo import *
from   function.lib_time import *


#
# General parameters
#
workdir="/home/app/function/"

bin_model = workdir+"model/google/bvlc_googlenet.caffemodel"
protxt    = workdir+"model/google/bvlc_googlenet.prototxt"
cls_model = workdir+"model/google/classification_classes_ILSVRC2012.txt"



#
# Functions for frame processing
#

# DNN
@memo_approx
def main_dnn_classify(image,threshold):
  # Load names of classes
  classes = None
  with open(cls_model, 'rt') as f:
     classes = f.read().rstrip('\n').split('\n')

  # load CNN model
  net = cv2.dnn.readNetFromCaffe(protxt, bin_model)

  # Run a model
  blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224), (104, 117,123), False, crop=False)
  net.setInput(blob)
  out = net.forward()

  # Get a class with a highest score.
  out        = out.flatten()
  classId    = np.argmax(out)
  confidence = out[classId]

  # Put efficiency information.
  t, _     = net.getPerfProfile()
  tcompute = (t * 1000.0) / cv2.getTickFrequency()

  info = {
           'classId':    classId,
           'confidence': confidence,
           'tcompute':   tcompute,
           'label2':     '%s: %.4f' % (classes[classId] if classes else 'Class #%d' % classId, confidence)
         }
  return info

# MD
@time_chronometer
def main_md_detect(ycfg, frame):
  frame_2 = imutils.resize(frame, width=500)
  frame_3 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)
  gray    = cv2.GaussianBlur(frame_3, (21, 21), 0)

  frameBase = ycfg['frame_b_gray']
  ycfg['frame_b_gray'] = gray
  ycfg['frame_b_orig'] = frame

  if ycfg['frame_0_gray'] is None:
     ycfg['frame_0_gray'] = gray
     ycfg['frame_0_orig'] = frame
     return None

  frameDelta = cv2.absdiff(frameBase, gray)
  thresh     = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
  thresh     = cv2.dilate(thresh, None, iterations=2)
  cnts       = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts       = imutils.grab_contours(cnts)
  for c in cnts:
      if cv2.contourArea(c) < ycfg['min_area']:
         continue
      info = {
               'classId':    '',
               'confidence': '',
               'tcompute':   '',
               'label2':     ''
             }
      return info

  return None


#
# Debug
#

def frameshow(ycfg, frame):
    ## transformations current frame...
    frame_1 = imutils.resize(frame, width=1024)
   #frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)

    #cv2.imshow('blur', frame_1);
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    ## transformations base frame...
    if (ycfg['frame_b_orig'] is None):
        return frame_1

    frame_f = imutils.resize(ycfg['frame_b_orig'], width=1024)
   #frame_f = cv2.cvtColor(frame_f, cv2.COLOR_BGR2GRAY)

    ## diff with base frame
    frameDelta = cv2.absdiff(frame_f, frame_1)
    frameDelta = cv2.fastNlMeansDenoising(frameDelta, h=10) ;
    thresh     = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    cv2.imshow('absdiff', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #print('f1')
    #print(frame_1)
    #print('ff')
    #print(frame_f)

    return frame_1


#
# Main
#

def usage(exit_val):
  print('')
  print('usage: ./memo.py -i <input_file_name>')
  print('')
  sys.exit(exit_val)


def handle(argv):
  threshold = 127
  # PARAMETERS
  try:
    opts, args = getopt.getopt(argv, "hi:t:", ["ifile=","threshold="])
  except getopt.GetoptError:
    usage(2)

  input_file_name= workdir+'assets/people-walking.mp4'
  for opt, arg in opts:
     if opt == '-h':
        usage(1)
     elif opt in ("-i", "--ifile"):
        input_file_name = arg
     elif opt in ("-t", "--threshold"):
        threshold = arg
  print("source: " + input_file_name)
  print("threshold: " + str(threshold))

  # LOOP for each frame of video...
  ycfg  = {}
  ystat = {}
  try:
    ## ycfg['cap'] = cv2.VideoCapture(0)
    ycfg['min_area']     = 500
    ycfg['cap']          = cv2.VideoCapture(input_file_name)
    ycfg['fgbg']         = cv2.createBackgroundSubtractorMOG2()
    ycfg['frame_0_gray'] = None
    ycfg['frame_0_orig'] = None
    ycfg['frame_b_gray'] = None
    ycfg['frame_b_orig'] = None

    ystat['frame_id']   = 1
    ystat['total_hits'] = 0
    ystat['total_time'] = 0
    while(1):
        ## Get frame
        ret, frame = ycfg['cap'].read()
        if frame is None:
           break

        ## debugging...
        #frame_p = frameshow(ycfg, frame)

        ## Processing frame...
        # DNN_CLASSIFY (compute similar)
        info2 = main_dnn_classify(frame, threshold)
        # MD_DETECT (motion detection)
        info3 = main_md_detect(ycfg, frame)

        ## Show frame stats...
        print(' ### frame: ' + str(ystat['frame_id']) + ' ########################## ')
        print(' * ' + 'dnn_classify.time'.ljust(25)     + ': ' + str(main_dnn_classify.time['time']) + ' (ms)')
        print(' * ' + 'dnn_classify.hit'.ljust(25)      + ': ' + str(main_dnn_classify.time['hit']) )
        print(' * ' + 'dnn_classify.class_id'.ljust(25) + ': ' + info2['label2'])
        print(' * ' + 'md_detect.time'.ljust(25)        + ': ' + str(main_md_detect.time['time'])    + ' (ms)')

        ## Update global stats...
        ystat['frame_id']   = ystat['frame_id'] + 1
        ystat['total_hits'] = ystat['total_hits'] + main_dnn_classify.time['hit']
        ystat['total_time'] = ystat['total_time'] + main_dnn_classify.time['time']
  except KeyboardInterrupt:
    print("Ctrl-C")

  # Show global stats
  print(' ### Global #################################### ')
  print(' * ' + 'Number of frames'.ljust(25) + ': ' + str(ystat['frame_id']) )
  print(' * ' + 'Number of hits'.ljust(25)   + ': ' + str(ystat['total_hits']) )
  print(' * ' + 'Average time'.ljust(25)     + ': ' + str(ystat['total_time'] / ystat['frame_id']) + ' (ms)' )

  # Free resources
  ycfg['cap'].release()
  cv2.destroyAllWindows()


