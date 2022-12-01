#!/bin/bash


cd Inputs/genos_g
declare -ga FILELIST=(*.csv)
cd ..
cd ..

chmod +x ann_final_2.py

for i in "${FILELIST[@]}"
do
	python ann_final_2.py -x Inputs/genos_g/"$i" -y Inputs/phenos_f/"$i"
done

echo "all done"

cd Outputs
awk 'FNR==1 && NR!=1{next;}{print}' *.txt > ann_accuracy.txt
