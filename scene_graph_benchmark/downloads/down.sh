linkList=(
"https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/vinvl_vg_x152c4.pth"
)

for link in ${linkList[@]}; do
    flag=0
    while true; do
      if [ $flag -eq 0 ]; then
        echo "Downloading $link"
        wget -c $link
        if [ $? -eq 0 ]; then
          flag=1
        fi
      else
        break
      fi
      sleep 2
    done
done
