#!/bin/bash

data_root="../data"

for fld_name in google flickr; do
    for js_fn in `ls ${data_root}/${fld_name}`; do
        json_file="${data_root}/${fld_name}/${js_fn}"
        echo $json_file
        # delete line which has backslash
        sed -i '/\\/d' $json_file
    done
done
