#!/bin/bash

# Remove unwanted files, including swap file, 
# python compile files [pyc, pyd] and etcs. 

root_path=".."

for fld_name in `ls ${root_path}`; do 
    cd ${root_path}/${fld_name}
    rm *.pyc
    rm *.*~
    rm */*.pyc
    rm */*.*~
done




