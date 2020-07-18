CUDA_VISIBLE_DEVICES=6 python3 main.py --status train

#CUDA_VISIBLE_DEVICES=5 python3 main.py --status test \
#					--savemodel "MLL_saved_model.0.model" \
#					--savedset "MLL_saved_modeld.set" \
#python3 main.py --status test \
#		--train ./data/msra_train.txt \
#		--dev ./data/msra_valid.txt \
#		--test ./data/msra_test.txt \
#		--savemodel ./data/saved_model \

# python main.py --status decode \
# 		--raw ../data/onto4ner.cn/test.char.bmes \
# 		--savedset ../data/onto4ner.cn/saved_model \
# 		--loadmodel ../data/onto4ner.cn/saved_model.13.model \
# 		--output ../data/onto4ner.cn/raw.out \
