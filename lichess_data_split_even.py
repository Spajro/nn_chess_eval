from random import shuffle

from src.loading.data_loading import load_dataset_with_stats, save_dataset_with_stats
from src.patches import TEST_DATASET_PATCH, TRAIN_DATASET_PATCH, LICHESS_DATASET_PATH

TEST_SIZE = 0.1

games_dataset = load_dataset_with_stats(LICHESS_DATASET_PATH)
even = []
odd = []
for fen, stats, value in games_dataset:
    if value[0] == 'M':
        continue
    if abs(int(value)) <= 100:
        even.append((fen, stats, value))
    else:
        odd.append((fen, stats, value))

print("Even:", len(even), " Odd: ", len(odd))
if len(even) < len(odd):
    odd = odd[:len(even)]
elif len(even) > len(odd):
    even = even[:len(odd)]

shuffle(even)
shuffle(odd)

even_index = int((1 - TEST_SIZE) * len(even))
odd_index = int((1 - TEST_SIZE) * len(odd))
print("Even train:", len(even[:even_index]), " Odd train:", len(odd[:odd_index]))
print("Even test:", len(even[even_index:]), "Odd test:", len(odd[odd_index:]))
train_dataset = even[:even_index] + odd[:odd_index]
test_dataset = even[even_index:] + odd[odd_index:]
print("train:", len(train_dataset), " test:", len(test_dataset))

save_dataset_with_stats(train_dataset, TRAIN_DATASET_PATCH)
save_dataset_with_stats(test_dataset, TEST_DATASET_PATCH)
