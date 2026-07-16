#!/bin/bash

# Path to www folder (update if needed)
WWW_PATH=~/mjpg-streamer/mjpg-streamer-experimental/www

# Stream from first USB2.0 PC CAM (probably /dev/video5)
mjpg_streamer -i "input_uvc.so -d /dev/video5 -r 640x480 -f 25" \
              -o "output_http.so -p 8080 -w $WWW_PATH" &

# Stream from second USB2.0 PC CAM (probably /dev/video1)
mjpg_streamer -i "input_uvc.so -d /dev/video1 -r 640x480 -f 25" \
              -o "output_http.so -p 8081 -w $WWW_PATH" &

wait