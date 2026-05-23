from random import shuffle

from src.loading.data_loading import load_dataset_with_stats, save_dataset_with_stats
from src.patches import TEST_DATASET_PATCH, TRAIN_DATASET_PATCH,LICHESS_DATASET_PATH

games_dataset = load_dataset_with_stats(LICHESS_DATASET_PATH)

TEST_SIZE = 0.1

games_index = int((1 - TEST_SIZE) * len(games_dataset))

shuffle(games_dataset)
train_dataset = games_dataset[:games_index]
test_dataset = games_dataset[games_index:]
print(len(train_dataset), len(test_dataset))


save_dataset_with_stats(train_dataset, TRAIN_DATASET_PATCH)
save_dataset_with_stats(test_dataset, TEST_DATASET_PATCH)
