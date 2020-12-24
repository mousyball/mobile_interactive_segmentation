#!/bin/bash

download_from_gdrive() {
    file_id=$1
    file_name=$2

    # create temp directory for cookies and html
    tmp_dir=~/tmp/gdown
    if [ ! -d $tmp_dir ]; then
      mkdir -p $tmp_dir
    fi

    # first stage to get the warning html
    curl -c ~/tmp/gdown/cookies \
    "https://drive.google.com/uc?export=download&id=$file_id" > \
    ~/tmp/gdown/intermezzo.html

    # second stage to extract the download link from html above
    download_link=$(cat ~/tmp/gdown/intermezzo.html | \
    grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | \
    sed 's/\&amp;/\&/g')
    curl -L -b ~/tmp/gdown/cookies \
    "https://drive.google.com$download_link" > $file_name

    # remove temp files and directory
    rm -r $tmp_dir
}

model_link_id='1RE-uEw0njTvxzfG4JvHAy7_wTwiG2Rv9'
model_name='hrnet32_ocr128_lvis.pth'

download_from_gdrive $model_link_id $model_name