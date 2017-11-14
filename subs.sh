#!/bin/bash
#
# creates a script that greps selected words.
#

FILE_NAME=$1
OLD_WORD=$2
NEW_WORD=$3

for file in *"${FILE_NAME}"*; do
   echo "$0: Analising ${file}"
   grep "${OLD_WORD}" "${file}"
   sed s/"${OLD_WORD}"/"${NEW_WORD}"/ < "${file}"
   sed s/"${OLD_WORD}"/"${NEW_WORD}"/ < "${file}" > ".${file}"
   mv ".${file}" "${file}"
done
