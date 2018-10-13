
#jbsub -interactive -q x86_24h -cores 5+1 -mem 80G -name gen_elmo_$1 \
    python ./src/ELMo_for_MHQA.py --in_path ./data/$1\_v2.json --out_prefix ./data/elmo_$1 --elmo_path /dccstor/amit_squad/models/zhigwang/encode_models_small/elmo

