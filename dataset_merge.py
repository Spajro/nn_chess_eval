from random import shuffle

from src.loading.data_loading import load_data_from_file
from src.loading.data_loading import save_dataset_to_csv
from src.patches import GAMES_DATASET_PATCH, PUZZLE_DATASET_PATCH, TEST_DATASET_PATCH, TRAIN_DATASET_PATCH

games_dataset = load_data_from_file(GAMES_DATASET_PATCH)
puzzle_dataset = load_data_from_file(PUZZLE_DATASET_PATCH)
len(games_dataset), len(puzzle_dataset)

shuffle(games_dataset)
shuffle(puzzle_dataset)

TEST_SIZE = 0.1

games_index = int((1 - TEST_SIZE) * len(games_dataset))
puzzle_index = int((1 - TEST_SIZE) * len(puzzle_dataset))

train_dataset = games_dataset[:games_index] + puzzle_dataset[:puzzle_index]
test_dataset = games_dataset[games_index:] + puzzle_dataset[puzzle_index:]
len(train_dataset), len(test_dataset)

save_dataset_to_csv(train_dataset, TRAIN_DATASET_PATCH)
save_dataset_to_csv(test_dataset, TEST_DATASET_PATCH)
