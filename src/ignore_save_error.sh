#!/bin/bash
n=1 #Image Set number to start processing batch
final_image=826 #Last Image Set number in your batch
cellprofiler -p ML/Batch_data.h5 -c -r -f $n -l $final_image |& tee log.txt #Here the output from terminal is saved to log.txt.

while [ $? -ne 1 ]; do
    #Parse log.txt file, extracting the image set # at which the error occurs
    no_save_error_image=$(sed -n '/^.*ValueError:/{n;p}' log.txt | sed -n -e 's/^.*Image # //p' | sed -n -e 's/, module SaveImages.*$//p')
    n_=$(($no_save_error_image + 1))
    #Restart CellProfiler at the next image set
    echo 'Restarting at Image set #' $n_ 
    cellprofiler -p ML/Batch_data.h5 -c -r -f $n_ -l $final_image |& tee log.txt
    #Once the final image set is reached stop the script
    if ((n_==$final_image)); then
       echo 'The last image set #:' $n_', has been processed. All images have been processed. The script will now exit.' 
       break
    fi
done 
