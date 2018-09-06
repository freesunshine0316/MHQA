# dataset
/dccstor/linfengsong/exp.multihop_qa

# Collect word vectors
Usage: 
> ${base_dir}/scripts/collect_word_vec.sh vocab_path outpath wordvecPath numThreads

Example:
> /u/zhigwang/zhigwang1/pycham_workspace/MHQA/scripts/collect_word_vec.sh /u/zhigwang/zhigwang1/tmp/vocab.txt /u/zhigwang/zhigwang1/tmp/wordvec.txt /dccstor/amit_squad/models/zhigwang/encode_models_small/glove.840B.300d 1



# preprocess script
Usage: 
> ${base_dir}/scripts/process_MHQA.sh inpath outpath lib_path numThreads

Example:
> /u/zhigwang/zhigwang1/pycham_workspace/MHQA/scripts/process_MHQA.sh /dccstor/linfengsong/exp.multihop_qa/data.wikihop/dev.compQA.json /u/zhigwang/zhigwang1/tmp/dev.tok /dccstor/amit_squad/models/zhigwang/encode_models_small/lib 1

In the output file, each line corresponds to a json object for one question. 

# Pre-compuate ELMo vectors
${base_dir}/src/ELMo_for_SQuAD.py --in_path inpath --out_prefix outpath --elmo_path /dccstor/amit_squad/models/zhigwang/encode_models_small/elmo
