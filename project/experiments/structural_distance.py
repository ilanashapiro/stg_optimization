import os, sys

composer_graphs = {}
DIRECTORY = '/home/ilshapiro/project/datasets'

with open('dataset_composers_in_phylogeny.txt', 'r') as file:
  for line in file:
    composer_graphs[line.strip()] = []

def build_composers_dict(base_dir=DIRECTORY, kunstderfuge_only=False):
  for root, _, files in os.walk(base_dir):
    relative_path = os.path.relpath(root, base_dir)
    path_parts = relative_path.split(os.sep)

    if len(path_parts) >= 3:
      source_dir = path_parts[-2]
      composer_dir = path_parts[-3] 
      
      for file in files:
        if file.endswith("_augmented_graph_flat.pickle"):
          full_path = os.path.join(root, file)
          if composer_dir in composer_graphs and (not kunstderfuge_only or source_dir == 'kunstderfuge'):
            composer_graphs[composer_dir].append(full_path)

build_composers_dict(kunstderfuge_only=False)
composer_graphs = {composer: graphs for composer, graphs in composer_graphs.items() if len(graphs) >= 5}

for composer, graphs in composer_graphs.items():
  print(composer, len(graphs))