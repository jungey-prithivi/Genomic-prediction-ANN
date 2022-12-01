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

#cd Outputs
#cat *_temp.csv > normal_gwas_gblup_0_b.csv
#sed -i 1i"phenotype,heritability,gblup_accuracy,gwas_gblup_accuracy" normal_gwas_gblup_0_b.csv
#rm *_temp.csv

#cd ..
#awk 'FNR==1 && NR!=1{next;}{print}' *.txt > gwas_ann_accuracy.txt