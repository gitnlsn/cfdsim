#!/bin/sh
THIS_NAME="PythonContainer v0.03"
PROG_NAME=$1
DIRR_NAME=$2
WORD=$3
WORDGREP="wordGreper.sh"

echo "${THIS_NAME}: begin."

if [ -e "${PROG_NAME}" ]; then      # A) program found
   cp "${PROG_NAME}" ~/Desktop
   cp "${WORDGREP}" ~/Desktop
   cd ~/Desktop
   if [ ! -d "${DIRR_NAME}" ]; then
      echo "${THIS_NAME}: Creating new directory."
      mkdir "${DIRR_NAME}"
   fi # created new directory if there is none
   mv "${PROG_NAME}" "${DIRR_NAME}/"   # copy program to directory
   mv "${WORDGREP}" "${DIRR_NAME}/"    # takes wordGreper to directory
   cd "${DIRR_NAME}"                   # moves into directory
   ./wordGreper.sh Objective
   echo "${THIS_NAME}: running program."
   python "${PROG_NAME}" > "output.txt"   # executes python program
else                                # B) program not found
   echo "${THIS_NAME}: program not found. Abort."
fi

echo "${THIS_NAME}: end."
