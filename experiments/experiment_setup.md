# Setup of experiments

## sizes of Tupi data

Guajajara: 3571 tokens, 810 different, 497 sentences
Tupinamba: 2546 tokens, 1224 different, 348 sentences
Karo: 2318 tokens, 779 different, 647 sentences

Kaapor: 366 tokens, 222 different, 83 sentences
Makurap: 146 tokens, 80 different, 31 sentences
Munduruku: 828 tokens, 333 different, 124 sentences
Akuntsu: 408 tokens, 223 different; 101 sentences

## Experiments Tupi

| Embedding | fine-tune train   | fine-tune target  | Model         |
| ---       | ---               | ---               | ---           |
| Multi     | -                 | no                | flair multi   |
| Multi     | -                 | yes               | flair multi   |
| Multi     | yes               | yes               | all 3         |
| own       | -                 | yes               | all 3         |
| own       | -                 | no                | all 3         |
| own       | -                 | yes               | all 7         |
| Multi     | yes               | yes               | all 7         |

## Experiments SK

| Embedding | Model train   |
| ---       | ---           |
| Multi     | Kakataibo     |
| Multi     | flair multi   |
| Multi     | SK            |
| Multi     | SK - Kakataibo|
| own       | Kakataibo     |
| Own       | SK            |
| own       | SK - Kakataibo|
