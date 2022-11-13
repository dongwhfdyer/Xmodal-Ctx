# if the folder has no ".pth" file, then delete it
runsFolder="runs/train"
for folder in `ls $runsFolder`
do
    echo $folder
    if [ ! -d $runsFolder/$folder/weights ]
    then
        echo "delete $runsFolder/$folder"
        rm -rf $runsFolder/$folder
    else
        if [ ! -f $runsFolder/$folder/weights/model_cur.pth ]
        then
            echo "delete $runsFolder/$folder"
            rm -rf $runsFolder/$folder
        fi
    fi


done