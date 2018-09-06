#!/bin/bash
if [ $# -lt 4 ]; then
	echo "Usage: $0 inpath outpath wordvecPath numThreads"
	exit 1
fi

cur_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

inPath=$1
outPath=$2
wordvecPath=$3
numThreads=$4

# configuration 
javaPackage="$cur_dir/lib/lucene-analyzers-common-7.2.0.jar:$cur_dir/lib/lucene-core-7.2.0.jar:$cur_dir/lib/lucene-queryparser-7.2.0.jar:$cur_dir/lib/qa.jar"

# collect word embeddings
indexPath=${wordvecPath}/luceneIndex/
wordvecPath=${wordvecPath}/wordvec.bin.gz
echo "Collecting word embeddings ... "
javaClass="zgwang.watson.wea.rc.SQuAD.CollectWordVecWithLuceneSearch"
java -Xmx9000m -cp ${javaPackage} ${javaClass} -mode collect_wordvec -model $wordvecPath -wordvecType word2vec -index $indexPath -inPath ${inPath} -outPath ${outPath} -numThreads $numThreads
echo "DONE!"

