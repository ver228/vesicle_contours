#!/bin/bash

#$ -P rittscher.prjc -q short.qc
#$ -t 1-821 -pe shmem 2

module use -a /mgmt/modules/eb/modules/all
module load Anaconda3/5.1.0
source activate tierpsy

#FILESSOURCE="$HOME/worm-ts-classification/collect_data/aggregation/files2process.txt"
FILES_SOURCE="$HOME/workspace/files2process.txt"
SRC_SCRIPT="$HOME/GitLab/vesicle_contours/scripts/contours/process_file.py"

FILE_SRC=$(awk "NR==$SGE_TASK_ID" $FILES_SOURCE)

echo "Username: " `whoami`
echo $FILE_SRC
python $SRC_SCRIPT $FILE_SRC

exit 0