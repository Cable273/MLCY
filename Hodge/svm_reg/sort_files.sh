#!/usr/bin/env sh

cd ./acc_data/
for split in $(seq 0.05 0.05 0.95)
do
	split_last_digit=$(echo $split | awk -F '.' '{print $2}' | awk -F '' '{print $2}')
	if [[ $split_last_digit == 0 ]]; then
		split=$(echo $split | sed s'/.$//')
	fi
		
	filename=$(ls  | grep "hodge_acc_split,$split,")

	mkdir -p "split_$split"

	for file in $filename
	do
		mv $file "./split_$split/$file"
	done
done
