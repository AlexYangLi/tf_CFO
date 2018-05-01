python inference.py \
--entdect_type "bilstm_crf" \
--subnet_type "transe" \
--entdect "../entity_detection/save_model/bilstm_crf-5" \
--relnet "../relation_network/relation_rank/save_model/RelRank_restrict-18" \
--subnet "../subject_network/sub_transe/save_model/SubTransE-8"
