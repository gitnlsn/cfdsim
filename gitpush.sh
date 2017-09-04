#!/bin/sh
THIS_NAME="Push Git v0.01"
COMMIT_DESCRIPTION=$*

echo "${THIS_NAME}: begin."

git add .
git commit -m \'"${COMMIT_DESCRIPTION}"\'
git push origin master

echo "${THIS_NAME}: end."
