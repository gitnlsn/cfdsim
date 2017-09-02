#!/bin/sh
THIS_NAME="Update Git v0.01"
COMMIT_DESCRIPTION=$*

echo "${THIS_NAME}: begin."

git add .
git commit -m \'"${COMMIT_DESCRIPTION}"\'
git push origin master
#echo \'"${COMMIT_DESCRIPTION}"\'

echo "${THIS_NAME}: end."
