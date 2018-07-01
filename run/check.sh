echo "check given row/column ......"

for i in {0..256..16}
do
	echo "${i}.txt"
	file="${i}.txt"
	awk '$2==1 {print}' ${file}
done
