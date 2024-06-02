echo '开始训练'

#单模MoE
python run_moe.py\
    --checkpoint_path ../model_storage/pretrain_910000_moe_seed2023_adda_addmerge_all \
    --seed 2023


#num
python train_num.py\
    --checkpoint_path ../model_storage/num_seed2023_adda_all \
    --seed 2023


#text
python train_text.py\
    --checkpoint_path ../model_storage/text_seed2023_adda_addmergeA_all \
    --seed 2023


python train_text.py\
    --checkpoint_path ../model_storage/text_seed42_adda_addmergeA_all \
    --seed 42


python train_text.py\
    --checkpoint_path ../model_storage/text_seed3407_adda_addmergeA_all \
    --seed 3407