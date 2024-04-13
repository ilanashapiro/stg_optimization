import os, shutil, sys, re, glob
import networkx as nx 
import multiprocessing 

# dir_prefix = "/home/ilshapiro/project/"
# db_dir_prefix = "/home/ilshapiro/project/classical_piano_midi_db/"
dir_prefix = "/home/jonsuss/Ilana_Shapiro/constraints/"
# dir_prefix = "/Users/ilanashapiro/Documents/constraints_project/project/"

sys.path.append(dir_prefix)
sys.path.append(f"{dir_prefix}/centroid")

import build_graph
import simanneal_centroid_run, simanneal_centroid_helpers

DIRECTORY = dir_prefix + "classical_piano_midi_db/"

centroid_composer_dirs = [
	DIRECTORY + "albeniz",
	DIRECTORY + "bach",
	DIRECTORY + "clementi",
	DIRECTORY + "haydn",
	DIRECTORY + "schumann",
	]

centroid_list = []
for dir in centroid_composer_dirs:
	for file in os.listdir(dir):
		# Construct full file path
		full_path = os.path.join(dir, file)

		# Check for "centroid_.*_final.txt"
		if re.match(r"centroid_(?!.*node_mapping).*_final\.txt", file):
			centroid_list.append(nx.read_edgelist(full_path))

test_composer_dir = DIRECTORY + "mozart"
STG_augmented_list = []
info_list = []
for piece_dir in os.listdir(test_composer_dir):
	full_dir_path = os.path.join(test_composer_dir, piece_dir)
	segment_filepaths = glob.glob(os.path.join(full_dir_path, "*_sf_fmc2d_segments.txt"))
	motive_filepaths = glob.glob(os.path.join(full_dir_path, "*_motives4.txt"))
	if len(segment_filepaths) > 0 and len(motive_filepaths) > 0:
		segment_filepath = segment_filepaths[0]
		motive_filepath = motive_filepaths[0]
		G, _ = build_graph.generate_graph(segment_filepath, motive_filepath)
		build_graph.augment_graph(G)
		STG_augmented_list.append(G)
		info_list.append(full_dir_path)

STG_augmented_list = STG_augmented_list[:5]
padded_matrices, idx_node_mapping = simanneal_centroid_helpers.pad_adj_matrices(centroid_list + STG_augmented_list)
padded_A_g_list = padded_matrices[:len(centroid_list)]
padded_STG_list = padded_matrices[len(centroid_list):]

albeniz_A_g = padded_A_g_list[0]
bach_A_g = padded_A_g_list[1]
clementi_A_g = padded_A_g_list[2]
haydn_A_g = padded_A_g_list[3]
schumann_A_g = padded_A_g_list[4]

# for i, A_G in enumerate(padded_STG_list):
# 	_, albeniz_cost = simanneal_centroid_run.align_graph_pair(A_G, albeniz_A_g, idx_node_mapping)
# 	_, bach_cost = simanneal_centroid_run.align_graph_pair(A_G, bach_A_g, idx_node_mapping)
# 	_, clementi_cost = simanneal_centroid_run.align_graph_pair(A_G, clementi_A_g, idx_node_mapping)
# 	_, haydn_cost = simanneal_centroid_run.align_graph_pair(A_G, haydn_A_g, idx_node_mapping)
# 	_, schumann_cost = simanneal_centroid_run.align_graph_pair(A_G, schumann_A_g, idx_node_mapping)
# 	print(info_list[0])
# 	print("ALBENIZ:", albeniz_cost, "BACH", bach_cost, "CLEMENTI", clementi_cost, "HAYDN", haydn_cost, "SCHUMANN", schumann_cost)
	
def align_and_compute_costs(A_G):
    # The function receives one graph and computes the costs for all alignments
    _, albeniz_cost = simanneal_centroid_run.align_graph_pair(A_G, albeniz_A_g, idx_node_mapping)
    _, bach_cost = simanneal_centroid_run.align_graph_pair(A_G, bach_A_g, idx_node_mapping)
    _, clementi_cost = simanneal_centroid_run.align_graph_pair(A_G, clementi_A_g, idx_node_mapping)
    _, haydn_cost = simanneal_centroid_run.align_graph_pair(A_G, haydn_A_g, idx_node_mapping)
    _, schumann_cost = simanneal_centroid_run.align_graph_pair(A_G, schumann_A_g, idx_node_mapping)
    
    # Return the results in a structured form
    return (albeniz_cost, bach_cost, clementi_cost, haydn_cost, schumann_cost)

def main():
	with multiprocessing.Pool() as pool:
		results = pool.map(align_and_compute_costs, padded_STG_list)
		for result, info in zip(results, info_list):
			print(info)
			albeniz_cost, bach_cost, clementi_cost, haydn_cost, schumann_cost = result
			print("ALBENIZ:", albeniz_cost, "BACH", bach_cost, "CLEMENTI", clementi_cost, "HAYDN", haydn_cost, "SCHUMANN", schumann_cost)

if __name__ == "__main__":
    main()

# 200 iterations SA align

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/mozart/mz_332_2 ---- YES
# ALBENIZ: 13.341664064126334 BACH 15.811388300841896 CLEMENTI 12.884098726725126 HAYDN 12.0 SCHUMANN 12.165525060596439


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/mozart/mz_311_2
# ALBENIZ: 16.217274740226856 BACH 18.411952639521967 CLEMENTI 15.968719422671311 HAYDN 15.132745950421556 SCHUMANN 15.132745950421556


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/mozart/mz_311_1
# ALBENIZ: 32.17141588429082 BACH 33.27160951922825 CLEMENTI 32.01562118716424 HAYDN 31.54362059117501 SCHUMANN 31.73326330524486


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/mozart/mz_545_3
# ALBENIZ: 9.591663046625438 BACH 13.341664064126334 CLEMENTI 9.273618495495704 HAYDN 8.246211251235321 SCHUMANN 7.211102550927978


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/mozart/mz_333_2 -- YES
# ALBENIZ: 15.7797338380595 BACH 18.303005217723125 CLEMENTI 15.588457268119896 HAYDN 15.0 SCHUMANN 15.066519173319364



# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/chopin/chpn_op10_e12 YES
# ALBENIZ: 8.306623862918075 BACH 12.36931687685298 CLEMENTI 8.06225774829855 HAYDN 6.708203932499369 SCHUMANN 6.244997998398398


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/chopin/chpn-p11
# ALBENIZ: 7.874007874011811 BACH 12.083045973594572 CLEMENTI 7.483314773547883 HAYDN 6.0 SCHUMANN 6.0


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/chopin/chpn_op35_4
# ALBENIZ: 20.784609690826528 BACH 22.538855339169288 CLEMENTI 20.493901531919196 HAYDN 19.949937343260004 SCHUMANN 20.09975124224178


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/chopin/chpn-p5
# ALBENIZ: 7.810249675906654 BACH 12.206555615733702 CLEMENTI 7.681145747868608 HAYDN 6.082762530298219 SCHUMANN 6.244997998398398


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/chopin/chpn-p2 MAYBE
# ALBENIZ: 18.027756377319946 BACH 20.074859899884732 CLEMENTI 17.804493814764857 HAYDN 17.233687939614086 SCHUMANN 17.11724276862369



# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/grieg/grieg_elfentanz
# ALBENIZ: 9.16515138991168 BACH 12.96148139681572 CLEMENTI 8.246211251235321 HAYDN 7.483314773547883 SCHUMANN 7.745966692414834


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/grieg/grieg_march
# ALBENIZ: 13.820274961085254 BACH 16.09347693943108 CLEMENTI 13.45362404707371 HAYDN 12.529964086141668 SCHUMANN 12.609520212918492


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/grieg/grieg_once_upon_a_time MAYBE
# ALBENIZ: 9.055385138137417 BACH 13.114877048604 CLEMENTI 8.602325267042627 HAYDN 8.0 SCHUMANN 7.874007874011811


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/grieg/grieg_walzer YES
# ALBENIZ: 8.54400374531753 BACH 12.36931687685298 CLEMENTI 7.937253933193772 HAYDN 6.557438524302 SCHUMANN 5.744562646538029


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/grieg/grieg_waechter
# ALBENIZ: 8.717797887081348 BACH 12.489995996796797 CLEMENTI 8.48528137423857 HAYDN 6.928203230275509 SCHUMANN 6.324555320336759


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/grieg/grieg_halling
# ALBENIZ: 7.937253933193772 BACH 11.958260743101398 CLEMENTI 7.280109889280518 HAYDN 5.744562646538029 SCHUMANN 5.916079783099616


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/grieg/grieg_voeglein
# ALBENIZ: 8.602325267042627 BACH 12.083045973594572 CLEMENTI 7.483314773547883 HAYDN 6.324555320336759 SCHUMANN 6.782329983125268


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/grieg/grieg_zwerge
# ALBENIZ: 42.22558466143482 BACH 43.116122274620196 CLEMENTI 42.035699113967404 HAYDN 41.86884283091664 SCHUMANN 41.86884283091664


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/grieg/grieg_brooklet
# ALBENIZ: 21.330729007701542 BACH 23.08679276123039 CLEMENTI 21.047565179849187 HAYDN 20.615528128088304 SCHUMANN 20.663978319771825


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/grieg/grieg_album
# ALBENIZ: 21.095023109728988 BACH 23.0 CLEMENTI 20.952326839756964 HAYDN 20.615528128088304 SCHUMANN 20.518284528683193


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/grieg/grieg_berceuse
# ALBENIZ: 11.269427669584644 BACH 14.106735979665885 CLEMENTI 10.908712114635714 HAYDN 9.539392014169456 SCHUMANN 9.746794344808963


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/grieg/grieg_butterfly YES
# ALBENIZ: 9.16515138991168 BACH 13.114877048604 CLEMENTI 8.602325267042627 HAYDN 8.0 SCHUMANN 7.211102550927978


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/grieg/grieg_wanderer
# ALBENIZ: 8.18535277187245 BACH 12.12435565298214 CLEMENTI 7.54983443527075 HAYDN 6.082762530298219 SCHUMANN 6.244997998398398


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/grieg/grieg_spring
# ALBENIZ: 18.920887928424502 BACH 20.92844953645635 CLEMENTI 18.76166303929372 HAYDN 18.16590212458495 SCHUMANN 17.944358444926362


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/grieg/grieg_kobold
# ALBENIZ: 26.419689627245813 BACH 27.784887978899608 CLEMENTI 26.19160170741759 HAYDN 25.768197453450252 SCHUMANN 25.88435821108957



# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/ravel/rav_gib
# ALBENIZ: 12.767145334803704 BACH 15.524174696260024 CLEMENTI 12.529964086141668 HAYDN 11.532562594670797 SCHUMANN 11.269427669584644


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/ravel/ravel_miroirs_1
# ALBENIZ: 38.704004960727254 BACH 39.6232255123179 CLEMENTI 38.62641583165593 HAYDN 38.31448812133603 SCHUMANN 38.31448812133603






# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/clementi/clementi_opus36_4_2 MAYBE
# ALBENIZ: 8.06225774829855 BACH 12.12435565298214 CLEMENTI 7.54983443527075 HAYDN 6.082762530298219 SCHUMANN 6.244997998398398


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/clementi/clementi_opus36_1_1 YES
# ALBENIZ: 10.04987562112089 BACH 13.152946437965905 CLEMENTI 9.433981132056603 HAYDN 8.06225774829855 SCHUMANN 8.426149773176359


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/clementi/clementi_opus36_5_2
# ALBENIZ: 14.696938456699069 BACH 17.029386365926403 CLEMENTI 14.422205101855956 HAYDN 13.564659966250536 SCHUMANN 13.038404810405298


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/clementi/clementi_opus36_3_2
# ALBENIZ: 8.54400374531753 BACH 12.36931687685298 CLEMENTI 8.06225774829855 HAYDN 6.708203932499369 SCHUMANN 6.4031242374328485


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/clementi/clementi_opus36_2_2
# ALBENIZ: 7.937253933193772 BACH 11.958260743101398 CLEMENTI 7.280109889280518 HAYDN 5.744562646538029 SCHUMANN 5.916079783099616


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/clementi/clementi_opus36_3_1
# ALBENIZ: 17.46424919657298 BACH 19.72308292331602 CLEMENTI 17.291616465790582 HAYDN 16.822603841260722 SCHUMANN 16.583123951777


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/clementi/clementi_opus36_1_2
# ALBENIZ: 8.94427190999916 BACH 12.884098726725126 CLEMENTI 7.211102550927978 HAYDN 7.615773105863909 SCHUMANN 7.745966692414834


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/clementi/clementi_opus36_4_1
# ALBENIZ: 16.46207763315433 BACH 19.0 CLEMENTI 16.217274740226856 HAYDN 15.968719422671311 SCHUMANN 15.652475842498529


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/clementi/clementi_opus36_6_2
# ALBENIZ: 21.047565179849187 BACH 22.737634001804146 CLEMENTI 20.85665361461421 HAYDN 20.223748416156685 SCHUMANN 20.37154878746336


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/clementi/clementi_opus36_5_3 YES
# ALBENIZ: 9.38083151964686 BACH 12.806248474865697 CLEMENTI 8.94427190999916 HAYDN 7.211102550927978 SCHUMANN 7.615773105863909


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/clementi/clementi_opus36_1_3
# ALBENIZ: 9.219544457292887 BACH 13.228756555322953 CLEMENTI 9.219544457292887 HAYDN 8.306623862918075 SCHUMANN 8.18535277187245


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/clementi/clementi_opus36_4_3
# ALBENIZ: 20.149441679609886 BACH 21.95449840010015 CLEMENTI 20.0 HAYDN 19.390719429665317 SCHUMANN 19.44222209522358


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/clementi/clementi_opus36_3_3
# ALBENIZ: 13.45362404707371 BACH 16.34013463836819 CLEMENTI 13.0 HAYDN 12.609520212918492 SCHUMANN 12.206555615733702


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/clementi/clementi_opus36_2_3
# ALBENIZ: 8.94427190999916 BACH 13.19090595827292 CLEMENTI 9.16515138991168 HAYDN 8.12403840463596 SCHUMANN 8.0


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/clementi/clementi_opus36_5_1
# ALBENIZ: 58.309518948453004 BACH 58.90670590009256 CLEMENTI 58.240879114244144 HAYDN 58.01723881744115 SCHUMANN 58.034472514187634


# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/clementi/clementi_opus36_2_1 YES
# ALBENIZ: 10.954451150103322 BACH 13.711309200802088 CLEMENTI 10.488088481701515 HAYDN 9.16515138991168 SCHUMANN 9.591663046625438




# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/schumann/scn15_10
# ALBENIZ: 8.831760866327848 BACH 12.328828005937952 CLEMENTI 8.246211251235321 HAYDN 6.48074069840786 SCHUMANN 7.0710678118654755

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/schumann/scn15_4
# ALBENIZ: 7.937253933193772 BACH 11.958260743101398 CLEMENTI 7.416198487095663 HAYDN 5.916079783099616 SCHUMANN 5.744562646538029

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/schumann/scn15_9
# ALBENIZ: 10.723805294763608 BACH 14.035668847618199 CLEMENTI 10.344080432788601 HAYDN 9.539392014169456 SCHUMANN 9.433981132056603

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/schumann/scn15_1
# ALBENIZ: 8.12403840463596 BACH 12.328828005937952 CLEMENTI 7.0710678118654755 HAYDN 6.6332495807108 SCHUMANN 6.782329983125268

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/schumann/scn15_11
# ALBENIZ: 8.06225774829855 BACH 12.041594578792296 CLEMENTI 6.855654600401044 HAYDN 5.916079783099616 SCHUMANN 6.082762530298219

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/schumann/scn16_8
# ALBENIZ: 11.832159566199232 BACH 14.628738838327793 CLEMENTI 11.489125293076057 HAYDN 10.392304845413264 SCHUMANN 10.488088481701515

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/schumann/scn68_10
# ALBENIZ: 7.937253933193772 BACH 11.958260743101398 CLEMENTI 7.416198487095663 HAYDN 5.916079783099616 SCHUMANN 5.744562646538029

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/schumann/scn15_2
# ALBENIZ: 8.831760866327848 BACH 12.806248474865697 CLEMENTI 7.615773105863909 HAYDN 7.483314773547883 SCHUMANN 7.615773105863909

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/schumann/scn15_7 YES
# ALBENIZ: 8.426149773176359 BACH 12.288205727444508 CLEMENTI 7.937253933193772 HAYDN 6.557438524302 SCHUMANN 6.082762530298219

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/schumann/scn15_12 YESSSSS
# ALBENIZ: 8.602325267042627 BACH 12.649110640673518 CLEMENTI 8.246211251235321 HAYDN 7.0710678118654755 SCHUMANN 5.656854249492381

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/schumann/scn16_6 YES
# ALBENIZ: 9.433981132056603 BACH 13.0 CLEMENTI 8.888194417315589 HAYDN 7.681145747868608 SCHUMANN 7.14142842854285

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/schumann/scn15_13
# ALBENIZ: 8.06225774829855 BACH 12.041594578792296 CLEMENTI 6.855654600401044 HAYDN 5.916079783099616 SCHUMANN 6.082762530298219

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/schumann/scn15_8 YESSS
# ALBENIZ: 8.18535277187245 BACH 12.206555615733702 CLEMENTI 7.681145747868608 HAYDN 6.244997998398398 SCHUMANN 5.744562646538029

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/schumann/scn16_3
# ALBENIZ: 15.132745950421556 BACH 17.349351572897472 CLEMENTI 14.730919862656235 HAYDN 13.96424004376894 SCHUMANN 13.674794331177344

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/schumann/scn15_5
# ALBENIZ: 8.602325267042627 BACH 12.24744871391589 CLEMENTI 7.483314773547883 HAYDN 6.324555320336759 SCHUMANN 6.782329983125268

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/schumann/scn16_7
# ALBENIZ: 12.884098726725126 BACH 15.937377450509228 CLEMENTI 12.649110640673518 HAYDN 11.916375287812984 SCHUMANN 12.083045973594572

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/schumann/scn68_12
# ALBENIZ: 21.02379604162864 BACH 22.847319317591726 CLEMENTI 20.83266665599966 HAYDN 20.346989949375804 SCHUMANN 20.199009876724155

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/schumann/scn16_4
# ALBENIZ: 8.831760866327848 BACH 12.489995996796797 CLEMENTI 8.48528137423857 HAYDN 6.164414002968976 SCHUMANN 6.324555320336759

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/schumann/scn15_6
# ALBENIZ: 8.06225774829855 BACH 12.041594578792296 CLEMENTI 6.855654600401044 HAYDN 5.916079783099616 SCHUMANN 6.082762530298219

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/schumann/scn16_1
# ALBENIZ: 19.209372712298546 BACH 21.0 CLEMENTI 19.05255888325765 HAYDN 18.411952639521967 SCHUMANN 18.411952639521967

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/schumann/scn15_3
# ALBENIZ: 7.937253933193772 BACH 11.958260743101398 CLEMENTI 7.416198487095663 HAYDN 5.916079783099616 SCHUMANN 5.744562646538029

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/schumann/scn16_5 MAYBE
# ALBENIZ: 10.04987562112089 BACH 13.30413469565007 CLEMENTI 9.643650760992955 HAYDN 8.18535277187245 SCHUMANN 7.937253933193772




# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/haydn/haydn_43_3
# ALBENIZ: 16.703293088490067 BACH 18.841443681416774 CLEMENTI 16.46207763315433 HAYDN 15.588457268119896 SCHUMANN 15.652475842498529

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/haydn/haydn_33_3
# ALBENIZ: 14.0 BACH 16.492422502470642 CLEMENTI 13.711309200802088 HAYDN 12.806248474865697 SCHUMANN 12.489995996796797

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/haydn/haydn_8_3
# ALBENIZ: 8.06225774829855 BACH 12.041594578792296 CLEMENTI 6.855654600401044 HAYDN 5.916079783099616 SCHUMANN 6.082762530298219

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/haydn/haydn_33_1
# ALBENIZ: 21.77154105707724 BACH 23.49468024894146 CLEMENTI 21.633307652783937 HAYDN 20.97617696340303 SCHUMANN 21.071307505705477

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/haydn/haydn_7_2
# ALBENIZ: 9.695359714832659 BACH 13.114877048604 CLEMENTI 9.16515138991168 HAYDN 7.615773105863909 SCHUMANN 7.745966692414834

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/haydn/haydn_7_3
# ALBENIZ: 17.204650534085253 BACH 19.131126469708992 CLEMENTI 16.61324772583615 HAYDN 16.186414056238647 SCHUMANN 16.3707055437449

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/haydn/haydn_35_2
# ALBENIZ: 30.72458299147443 BACH 31.96873472629156 CLEMENTI 30.62678566222711 HAYDN 30.166206257996713 SCHUMANN 30.331501776206203

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/haydn/haydn_9_1
# ALBENIZ: 12.649110640673518 BACH 15.231546211727817 CLEMENTI 12.328828005937952 HAYDN 11.224972160321824 SCHUMANN 11.313708498984761

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/haydn/haydn_35_3 YESSSSS
# ALBENIZ: 8.717797887081348 BACH 12.328828005937952 CLEMENTI 7.615773105863909 HAYDN 5.291502622129181 SCHUMANN 6.928203230275509

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/haydn/haydn_7_1
# ALBENIZ: 9.591663046625438 BACH 12.884098726725126 CLEMENTI 8.602325267042627 HAYDN 7.483314773547883 SCHUMANN 8.0

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/haydn/hay_40_2
# ALBENIZ: 11.269427669584644 BACH 14.66287829861518 CLEMENTI 10.63014581273465 HAYDN 10.246950765959598 SCHUMANN 9.643650760992955

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/haydn/haydn_8_4
# ALBENIZ: 10.14889156509222 BACH 13.74772708486752 CLEMENTI 9.539392014169456 HAYDN 9.1104335791443 SCHUMANN 9.0

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/haydn/haydn_9_2 YES
# ALBENIZ: 10.954451150103322 BACH 14.0 CLEMENTI 10.583005244258363 HAYDN 9.16515138991168 SCHUMANN 9.695359714832659

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/haydn/haydn_9_3
# ALBENIZ: 9.899494936611665 BACH 13.490737563232042 CLEMENTI 9.591663046625438 HAYDN 8.48528137423857 SCHUMANN 8.48528137423857

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/haydn/haydn_8_2
# ALBENIZ: 7.937253933193772 BACH 11.958260743101398 CLEMENTI 7.280109889280518 HAYDN 5.744562646538029 SCHUMANN 5.744562646538029

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/haydn/haydn_33_2
# ALBENIZ: 15.7797338380595 BACH 18.24828759089466 CLEMENTI 15.588457268119896 HAYDN 15.0 SCHUMANN 14.594519519326424

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/haydn/haydn_8_1
# ALBENIZ: 13.19090595827292 BACH 15.620499351813308 CLEMENTI 12.806248474865697 HAYDN 11.832159566199232 SCHUMANN 11.575836902790225

# /homejonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/haydn/haydn_43_2
# ALBENIZ: 10.0 BACH 13.784048752090222 CLEMENTI 9.486832980505138 HAYDN 8.94427190999916 SCHUMANN 8.12403840463596



# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/granados/gra_esp_2
# ALBENIZ: 14.142135623730951 BACH 16.852299546352718 CLEMENTI 13.856406460551018 HAYDN 13.341664064126334 SCHUMANN 13.341664064126334

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/granados/gra_esp_4
# ALBENIZ: 8.888194417315589 BACH 12.767145334803704 CLEMENTI 8.54400374531753 HAYDN 7.280109889280518 SCHUMANN 6.082762530298219

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/granados/gra_esp_3
# ALBENIZ: 12.767145334803704 BACH 15.329709716755891 CLEMENTI 12.449899597988733 HAYDN 11.532562594670797 SCHUMANN 11.357816691600547


# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/brahms/brahms_opus117_2
# ALBENIZ: 11.0 BACH 14.035668847618199 CLEMENTI 10.816653826391969 HAYDN 9.433981132056603 SCHUMANN 9.433981132056603

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/brahms/br_im5 YES
# ALBENIZ: 8.54400374531753 BACH 12.68857754044952 CLEMENTI 8.426149773176359 HAYDN 7.280109889280518 SCHUMANN 7.0

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/brahms/brahms_opus117_1
# ALBENIZ: 8.54400374531753 BACH 12.449899597988733 CLEMENTI 7.681145747868608 HAYDN 6.708203932499369 SCHUMANN 7.0

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/brahms/br_im2
# ALBENIZ: 11.532562594670797 BACH 14.52583904633395 CLEMENTI 11.357816691600547 HAYDN 10.04987562112089 SCHUMANN 10.14889156509222

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/brahms/brahms_opus1_2
# ALBENIZ: 9.899494936611665 BACH 13.19090595827292 CLEMENTI 9.38083151964686 HAYDN 7.874007874011811 SCHUMANN 7.483314773547883




# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/albeniz/alb_esp3
# ALBENIZ: 17.52141546793523 BACH 19.467922333931785 CLEMENTI 17.11724276862369 HAYDN 16.522711641858304 SCHUMANN 16.703293088490067

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/albeniz/alb_se2
# ALBENIZ: 10.954451150103322 BACH 14.071247279470288 CLEMENTI 10.583005244258363 HAYDN 9.38083151964686 SCHUMANN 9.591663046625438

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/albeniz/alb_se4
# ALBENIZ: 11.874342087037917 BACH 14.730919862656235 CLEMENTI 11.269427669584644 HAYDN 9.9498743710662 SCHUMANN 10.63014581273465

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/albeniz/alb_se8
# ALBENIZ: 14.422205101855956 BACH 16.73320053068151 CLEMENTI 14.142135623730951 HAYDN 13.2664991614216 SCHUMANN 13.2664991614216

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/albeniz/alb_esp4
# ALBENIZ: 7.810249675906654 BACH 12.922847983320086 CLEMENTI 8.774964387392123 HAYDN 7.54983443527075 SCHUMANN 7.681145747868608

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/albeniz/alb_esp1
# ALBENIZ: 10.44030650891055 BACH 13.45362404707371 CLEMENTI 10.04987562112089 HAYDN 8.774964387392123 SCHUMANN 8.774964387392123

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/albeniz/alb_esp5
# ALBENIZ: 10.908712114635714 BACH 13.892443989449804 CLEMENTI 10.44030650891055 HAYDN 9.219544457292887 SCHUMANN 8.54400374531753

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/albeniz/alb_esp2
# ALBENIZ: 8.18535277187245 BACH 12.206555615733702 CLEMENTI 7.681145747868608 HAYDN 6.244997998398398 SCHUMANN 6.082762530298219

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/albeniz/alb_se5
# ALBENIZ: 52.592775169218825 BACH 53.291650377896914 CLEMENTI 52.459508194416 HAYDN 52.28766584960549 SCHUMANN 52.3450093132096

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/albeniz/alb_se7
# ALBENIZ: 13.892443989449804 BACH 16.34013463836819 CLEMENTI 13.601470508735444 HAYDN 12.529964086141668 SCHUMANN 12.68857754044952

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/albeniz/alb_se1
# ALBENIZ: 38.34057902536163 BACH 39.344631145812 CLEMENTI 38.2099463490856 HAYDN 37.94733192202055 SCHUMANN 37.97367509209505

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/albeniz/alb_se3
# ALBENIZ: 17.26267650163207 BACH 19.28730152198591 CLEMENTI 16.97056274847714 HAYDN 16.24807680927192 SCHUMANN 16.06237840420901

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/albeniz/alb_esp6
# ALBENIZ: 9.797958971132712 BACH 13.564659966250536 CLEMENTI 9.16515138991168 HAYDN 8.717797887081348 SCHUMANN 8.0


# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/bach/inventions_bwv-776
# ALBENIZ: 19.595917942265423 BACH 21.307275752662516 CLEMENTI 19.28730152198591 HAYDN 18.708286933869708 SCHUMANN 18.49324200890693

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/bach/bach_846
# ALBENIZ: 35.11409973215888 BACH 36.318039594669756 CLEMENTI 35.02855977627399 HAYDN 34.79942528261063 SCHUMANN 34.655446902326915

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/bach/inventions_bwv-781
# ALBENIZ: 13.892443989449804 BACH 16.278820596099706 CLEMENTI 13.152946437965905 HAYDN 12.609520212918492 SCHUMANN 12.84523257866513

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/bach/inventions_bwv-777
# ALBENIZ: 16.401219466856727 BACH 18.520259177452136 CLEMENTI 16.15549442140351 HAYDN 15.459624833740307 SCHUMANN 15.459624833740307

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/bach/inventions_bwv-779
# ALBENIZ: 16.0312195418814 BACH 18.411952639521967 CLEMENTI 15.459624833740307 HAYDN 15.198684153570664 SCHUMANN 15.264337522473747

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/bach/inventions_bwv-785
# ALBENIZ: 13.30413469565007 BACH 16.09347693943108 CLEMENTI 12.609520212918492 HAYDN 12.288205727444508 SCHUMANN 12.36931687685298

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/bach/bach_847
# ALBENIZ: 13.92838827718412 BACH 16.673332000533065 CLEMENTI 13.638181696985855 HAYDN 12.96148139681572 SCHUMANN 12.649110640673518

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/bach/inventions_bwv-775
# ALBENIZ: 18.894443627691185 BACH 21.18962010041709 CLEMENTI 18.894443627691185 HAYDN 18.466185312619388 SCHUMANN 18.520259177452136

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/bach/inventions_bwv-774
# ALBENIZ: 13.30413469565007 BACH 16.15549442140351 CLEMENTI 12.767145334803704 HAYDN 12.36931687685298 SCHUMANN 12.449899597988733

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/bach/inventions_bwv-786
# ALBENIZ: 11.489125293076057 BACH 14.628738838327793 CLEMENTI 11.045361017187261 HAYDN 10.295630140987 SCHUMANN 10.392304845413264

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/bach/bach_850
# ALBENIZ: 17.05872210923198 BACH 19.0 CLEMENTI 16.822603841260722 HAYDN 16.09347693943108 SCHUMANN 16.15549442140351

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/bach/inventions_bwv-784
# ALBENIZ: 11.789826122551595 BACH 14.52583904633395 CLEMENTI 11.357816691600547 HAYDN 9.9498743710662 SCHUMANN 10.535653752852738

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/bach/inventions_bwv-772
# ALBENIZ: 11.357816691600547 BACH 12.68857754044952 CLEMENTI 10.816653826391969 HAYDN 9.746794344808963 SCHUMANN 10.04987562112089

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/bach/inventions_bwv-780
# ALBENIZ: 12.0 BACH 14.696938456699069 CLEMENTI 11.489125293076057 HAYDN 10.392304845413264 SCHUMANN 10.770329614269007

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/bach/inventions_bwv-783
# ALBENIZ: 19.339079605813716 BACH 21.118712081942874 CLEMENTI 19.078784028338912 HAYDN 18.49324200890693 SCHUMANN 18.65475810617763

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/bach/inventions_bwv-782
# ALBENIZ: 15.066519173319364 BACH 17.291616465790582 CLEMENTI 14.730919862656235 HAYDN 13.892443989449804 SCHUMANN 13.892443989449804

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/bach/inventions_bwv-773
# ALBENIZ: 15.0 BACH 17.233687939614086 CLEMENTI 14.66287829861518 HAYDN 13.820274961085254 SCHUMANN 13.96424004376894

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/bach/inventions_bwv-778
# ALBENIZ: 22.0 BACH 23.748684174075834 CLEMENTI 21.817424229271428 HAYDN 21.354156504062622 SCHUMANN 21.400934559032695




# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/debussy/debussy_cc_4
# ALBENIZ: 45.552167895721496 BACH 46.44351407893249 CLEMENTI 45.50824101193101 HAYDN 45.28796749689701 SCHUMANN 45.17742799230607

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/debussy/deb_menu
# ALBENIZ: 15.620499351813308 BACH 17.776388834631177 CLEMENTI 15.297058540778355 HAYDN 14.560219778561036 SCHUMANN 14.628738838327793

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/debussy/deb_prel
# ALBENIZ: 14.7648230602334 BACH 17.08800749063506 CLEMENTI 14.491376746189438 HAYDN 13.638181696985855 SCHUMANN 13.416407864998739

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/debussy/debussy_cc_6
# ALBENIZ: 15.033296378372908 BACH 17.26267650163207 CLEMENTI 14.7648230602334 HAYDN 14.0 SCHUMANN 13.92838827718412

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/debussy/debussy_cc_1
# ALBENIZ: 18.49324200890693 BACH 20.591260281974 CLEMENTI 18.2208671582886 HAYDN 17.776388834631177 SCHUMANN 17.4928556845359

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/debussy/debussy_cc_3
# ALBENIZ: 31.28897569432403 BACH 32.449961479175904 CLEMENTI 31.192947920964443 HAYDN 30.773365106858236 SCHUMANN 30.83828789021855

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/debussy/debussy_cc_2
# ALBENIZ: 11.445523142259598 BACH 14.317821063276353 CLEMENTI 11.090536506409418 HAYDN 9.848857801796104 SCHUMANN 9.848857801796104



# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/liszt/liz_et4
# ALBENIZ: 19.697715603592208 BACH 21.633307652783937 CLEMENTI 19.390719429665317 HAYDN 19.026297590440446 SCHUMANN 19.026297590440446

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/liszt/liz_liebestraum
# ALBENIZ: 23.664319132398465 BACH 25.219040425836983 CLEMENTI 23.53720459187964 HAYDN 23.021728866442675 SCHUMANN 23.108440016582687

# /home/jonsuss/Ilana_Shapiro/constraints/classical_piano_midi_db/liszt/liz_et5
# ALBENIZ: 9.591663046625438 BACH 13.038404810405298 CLEMENTI 9.16515138991168 HAYDN 7.615773105863909 SCHUMANN 8.12403840463596