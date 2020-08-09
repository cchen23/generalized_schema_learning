import json
import os

from directories import base_dir


with open(os.path.join(base_dir, 'experiment_parameters.json'), 'r') as f:
    experiment_parameters = json.load(f)

embedding_size = 50

# Fr experiment creation.
fixed_train_instances = {'Person': ['Mariko', 'Pradeep', 'Sarah', 'Julian', 'Jane', 'John'],
        'Drink': ['latte', 'water', 'juice', 'milk', 'espresso', 'chocolate'],
        'Dessert': ['mousse', 'cookie', 'candy', 'cupcake', 'cheesecake', 'pastry']
        }
fixed_test_instances = {'Person': ['Olivia', 'Will', 'Anna', 'Bill'],
        'Drink': ['coffee', 'tea'],
        'Dessert': ['cake', 'sorbet']
        }
