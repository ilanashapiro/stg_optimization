import os, centroid_quantitative

composer = "haydn"
centroid_quantitative.delete_initial_alignments_dirs([composer])
centroid_quantitative.generate_centroid(os.path.join(centroid_quantitative.DIRECTORY, composer), composer)