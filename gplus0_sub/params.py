# paths
edges_filename = 'undirected_edges.dat' 
attr_filename = 'node_attributes.csv'    

# attributes
use_context = True      # use the context embedding in addition to attribute similarity embeddings
predictor_attr_types = ['employer', 'major', 'places_lived', 'school']
nomination_attr_type = 'school'       # note: this automatically gets excluded from predictor_attr_types                          
nomination_attr_val = 'princeton university'  

# options
load_embeddings = True  # load pickled embeddings if they exist; otherwise, create them anew & save them
verbose = True          # show output
save_info = True        # save off info to .txt files, plots to .png files
rank_thresh = 100       # rank threshold for top-ranked attribute reports/plots
rng_seed = 123          # RNG seed for replicating experiments (random if = None)

# embedding
sim = 'NPMI1s'          # similarity measure ('NPMI1s', 'PMIs', 'conditional_prob')
delta = 0.0             # smoothing parameter for PMIs only (>= 0)
embedding = 'adj'       # embedding type ('adj', 'adj+diag', 'normlap', 'regnormlap')
max_eig = 200           # maximum number of eigenvectors to compute
which_elbow = 2         # which elbow of the scree plot to set threshold (if 0, keeps all eigenvectors)

# classification
sphere = True           # project embedding vectors to the unit sphere
classifier = 'logreg'   # supervised classifier ('logreg', 'naive_bayes', 'randfor', 'boost')
num_trees = 100         # number of decision trees to use if classifier is randfor or boost
cv_max = 100            # max number of seeds to sample for cross-validation
nominate_max = 500      # max number of unknown nodes to output in nomination list
