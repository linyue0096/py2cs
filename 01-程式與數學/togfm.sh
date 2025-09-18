#cp -r ./$1 ./$1/_doc
# set -x
mkdir -p ./$1/_doc/
cp ./$1/*.md ./$1/_doc/

# perl -i.bak.markdown  -p \
#perl -0777 -i -p \
#  -e 's/\\\[\s*(.*?)\s*\\\]/\n```math\n\1\n```\n/gs;' \
#  -e 's/\\\(\s*(.*?)\s*\\\)/ \$`\1`\$ /g;' \
#  ./$1/_doc/*.md

perl -0777 -i -p \
  -e 's/(\S)\*\*\s*(.*?)\s*\*\*(\S)/$1 \*\*$2\*\* $3/gs;' \
  ./$1/_doc/*.md
