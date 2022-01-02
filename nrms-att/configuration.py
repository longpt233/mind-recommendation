import os

from reco_utils.recommender.deeprec.deeprec_utils import download_deeprec_resources
from reco_utils.recommender.newsrec.io.mind_iterator import MINDIterator
from reco_utils.recommender.newsrec.newsrec_utils import prepare_hparams
from reco_utils.recommender.newsrec.newsrec_utils import get_mind_data_set


# ghi đè lạ config 
epochs = 4
seed = 42
MIND_type = 'small'
data_root_path = '/content/drive/MyDrive/20211/rec-sys/mind-recomendation/data'
data_path = f'{data_root_path}/{MIND_type}'
show_step = 10000

# set up the path of dataset
train_news_file = os.path.join(data_path, 'train', r'news.tsv')
train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')

# này nếu chạy thật = large 
# test_news_file = os.path.join(data_path, 'test', r'news.tsv')
# test_behaviors_file = os.path.join(data_path, 'test', r'behaviors.tsv')

# TODO: modify the embedding file
wordEmb_file = os.path.join(data_root_path, "utils", "embedding.npy")
userDict_file = os.path.join(data_root_path, "utils", "uid2index.pkl")
wordDict_file = os.path.join(data_root_path, "utils", "word_dict.pkl")
yaml_file = os.path.join(data_root_path, "utils", r'nrms.yaml')  # lấy 1 cái này ra để check xem có thư mục utils chưa 
# cái file này là mặc định, file nrms-att với nrms-entity khác nhé 


# tiến hành tải nếu chưa có 
mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(MIND_type)
if not os.path.exists(train_news_file):
    print("not have train file, dowloadding .....")
    download_deeprec_resources(mind_url, os.path.join(data_path, 'train'), mind_train_dataset)
if not os.path.exists(valid_news_file):
    print("not have valid file, dowloadding .....")
    download_deeprec_resources(mind_url, os.path.join(data_path, 'valid'), mind_dev_dataset)
if not os.path.exists(yaml_file):
    print("not have utils file, dowloadding .....")
    utils_url = r'https://recodatasets.blob.core.windows.net/newsrec/'
    download_deeprec_resources(utils_url, os.path.join(data_root_path, 'utils'), mind_utils)


# phải trueyefn vào 1 trong 2 nrms-att.yaml or nrms-entity.yaml
def load_trainer(yaml_name=None, log_file=None): 
    yaml_path = os.path.join(data_root_path, "utils", yaml_name) 
    log_path = os.path.join(data_path, "log")
    os.makedirs(log_path, exist_ok=True)

    hparams = prepare_hparams(yaml_path, wordEmb_file=wordEmb_file, wordDict_file=wordDict_file,
                              epochs=epochs, show_step=show_step, userDict_file=userDict_file)
    # set up log file
    log_file = log_file if log_file else hparams.log_file
    log_file = os.path.join(log_path, log_file)
    hparams.log_file = log_file
    print(hparams.to_string())
    if hparams.model_type == "nrms_entity":
        print("load nrms-entity trainer")
        from reco_utils.recommender.newsrec.trainers.entity_trainer import EntityTrainer
        trainer = EntityTrainer(hparams, MINDIterator, seed)
    else:
        print("load nrms-att trainer")
        from reco_utils.recommender.newsrec.trainers.base_trainer import BaseTrainer
        # set trainer
        trainer = BaseTrainer(hparams, MINDIterator, seed)
    return trainer
