#!/bin/bash
#
# creates a script that greps selected words.
#

if [ "$#" -ge 1 ];
then
  echo "#!/bin/bash"                             >  check.sh
  for word in $*
  do
    echo "cat output.txt | grep ${word} > ${word}.txt" >> check.sh
    echo "cat ${word}.txt"                             >> check.sh
  done
  chmod +x check.sh
fi
