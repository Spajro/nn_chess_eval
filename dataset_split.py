from random import shuffle

from src.data_loading import load_data_from_file
from src.data_loading import save_dataset_to_csv
from src.patches import GAMES_DATASET_PATCH, TEST_DATASET_PATCH, TRAIN_DATASET_PATCH

games_dataset = load_data_from_file(GAMES_DATASET_PATCH)
even = []
odd = []
for fen, value in games_dataset:
    if abs(value) <= 100:
        even.append((fen, value))
    else:
        odd.append((fen, value))

print("Even:", len(even), " Odd: ", len(odd))
if len(even) < len(odd):
    odd = odd[:len(even)]
elif len(even) > len(odd):
    even = even[:len(odd)]

TEST_SIZE = 0.1

even_index = int((1 - TEST_SIZE) * len(even))
odd_index = int((1 - TEST_SIZE) * len(odd))
print("Even train:", len(even[:even_index]), " Odd train:", len(odd[:odd_index]))
print("Even test:", len(even[even_index:]), " Test:", len(odd[odd_index:]))
train_dataset = even[:even_index] + odd[:odd_index]
test_dataset = even[even_index:] + odd[odd_index:]

shuffle(train_dataset)
shuffle(test_dataset)

save_dataset_to_csv(train_dataset, TRAIN_DATASET_PATCH)
save_dataset_to_csv(test_dataset, TEST_DATASET_PATCH)
