## "Deriving Representative Structure From Music Corpora"
This is the repository for our paper.

## Abstract:
Western music is an innately hierarchical system of interacting levels of structure, from fine-grained melody to high-level form. In order to analyze music compositions holistically and at multiple granularities, we propose a unified, hierarchical meta-representation of musical structure called the *structural temporal graph* (STG). For a single piece, the STG is a data structure that defines a hierarchy of progressively finer structural musical features and the temporal relationships between them. We use the STG to enable a novel approach for deriving a representative structural summary of a music corpus, which we formalize as a dually NP-hard combinatorial optimization problem extending the Generalized Median Graph problem. Our approach first applies simulated annealing to develop a measure of *structural distance* between two music pieces rooted in graph isomorphism. Our approach then combines the formal guarantees of SMT solvers with nested simulated annealing over structural distances to produce a structurally sound, representative *centroid* STG for an entire corpus of STGs from individual pieces. To evaluate our approach, we conduct experiments verifying that structural distance accurately differentiates between music pieces, and that derived centroids accurately structurally characterize their corpora.

## Summary of Repository
All pre-processing of data (code converting MIDI-CSV and MIDI-MP3, and code for generating the single-level analyses that constitute the STG from these files) is found in the folder `project/analyses`.

The file `build_graph.py` in the top level of the project folder contains the runner code from Section 3 of the paper: creating the STGs from music pieces and adding to our dataset, and augmenting, compressing, and visualizing STGs. `parse_analyses.py` contains helper functions for `build_graph.py` that build the STGs according to the analyses in the beginning of Section 6 experiments.

The folder `project/centroid` contains the code from Sections 4 and 5 in the paper. `simanneal_centroid.py` contains the implementation of both annealers (single-level graph alignment annealing and bi-level centroid annealing) from Sections 4.1 and 5.1. Approximate centroids are generated using this file. `z3_matrix_projection_incremental.py` contains the Z3 solver code for repairing the approximate centroids from Section 5.2: the final, repaired, structurally sound centroids are generated using this file.

The folders `project/experiments/structural_distance` and `project/experiments/centroid` respectively contain the code for the Section 6.1 (structural distance) and Section 6.2 (centroid) experiments (both mathematical and musical evaluations).

Individual files also contain docstrings relating the code to the relevant parts of the paper.

## Information about the experiments
In the Structural Distance experiment from Section 6.1, we do the mathematical evaluation in `project/experiments/structural_distance/structural_distance_synthetic` (run the file structural_distance_synthetic.py in this folder), and the musical evaluation in `project/experiments/structural_distance/structural_distance_music` (run the file `analyze_results.py` in this folder). Details about the 32 pieces used for the Structural Distance Music Evaluation Experiment (Section 6.1) can be found in the file `project/experiments/structural_distance/structural_distance_music/input_pieces.txt`. The 210 set combinations of these pieces used in the experiment can be found in `project/experiments/structural_distance/structural_distance_music/set_combinations.txt`.

In the Centroids Experiment from Section 6.2, we do the mathematical evaluation in project/experiments/centroid/synthetic_centroid_experiment (run the file `synthetic_centroid_experiment.py` in this folder), and the musical evaluation in `project/experiments/centroid/substructure_frequency_experiment` (run the file `substructure_frequency_experiment.py` in this folder). Details about the pieces used for the Centroid Music Evaluation Experiment (Section 6.2) can be found in the file `project/experiments/centroid/corpora/info.txt`.

All the relevant material, including the generated/synthetic synthetic STGs used to get the results in the paper, are in these folders.

## Information about the dataset
Our dataset, which is built on MIDI files from the Kunstderfuge (https://www.kunstderfuge.com/) and Classical Piano MIDI Database (http://www.piano-midi.de/midi_files.htm) datasets, is found in project/datasets. It has the structure:
- datasets
  - composer
    - kunstderfuge
    - classical_piano_midi_db
      - piece_name
        - MIDI file
        - paired MP3 file
        - paired CSV file
        - STG pickle file (created with build_graph.py)
        - output files of analyses (segments, motives, etc)
We generated the paired MP3 and CSV files, as well as the STGs.
The full dataset was too large to include, so we just include one composer folder (Beethoven) along with 4 sample pieces, two of which are referenced in the paper. The full dataset would be made publicly available following publication.

**NOTE:** We had to omit files from this repo due to size limitations. These do not include code files that we modified, but do include most of our dataset and several output files (only the relevant output files in the experiments are kept. The Harmony Transformer dataset and pretrained model checkpoints are not included in the analyses folder).
