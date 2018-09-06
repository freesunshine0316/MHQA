#!/bin/bash
if [ $# -lt 4 ]; then
	echo "Usage: $0 inpath outpath lib_path numThreads"
	exit 1
fi

libpath=$3
numThreads=$4

#=====java configuration=======
cur_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
QA_HOME_PATH="$(dirname "$cur_dir")"

javaPackage="$cur_dir/lib/qa.jar:${libpath}/commons-lang3-3.4.jar:${libpath}/gson-2.3.1.jar:${libpath}/stanford-corenlp-3.8.0.jar:${libpath}/stanford-english-corenlp-2016-10-31-models.jar:${libpath}/json-simple-1.1.1.jar"
javaClass="zgwang.watson.wea.factoid.process.ProcessMultihopQA"

java -Xmx30000m -cp ${javaPackage} ${javaClass} -in $1 -out $2 -numThreads ${numThreads}
