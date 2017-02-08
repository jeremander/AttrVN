# paths
nomination_path = 'selected_attrs.csv'  # filename of attributes to nominate
edges_filename = 'undirected_edges.dat' 
attr_filename = 'node_attributes.csv'    

# attributes
use_context = True      # use the context embedding in addition to attribute features
predictor_attr_types = {'employer' : str, 'major' : str, 'school' : str, 'places_lived' : str}
nomination_attr_type = 'places_lived'               
nomination_attr_val = 'baltimore, md' 

# options
vn_method = 'randomwalk' # how to perform VN ('embedding', 'randomwalk', 'diffusion')
#verbose = True
verbosity = 2           # show output
load_pfa = True         # load pickled PairwiseFreqAnalyzers if they already exist
load_embeddings = True  # load pickled embeddings if they already exist
save_info = True        # save off info to .txt files, plots to .png files
rank_thresh = 100       # rank threshold for top-ranked attribute reports/plots
nominate_max = 200      # max number of unknown nodes to output in nomination list
rng_seed = 123          # RNG seed for replicating experiments (random if None)
n_jobs = 4              # maximum number of jobs for parallel routines

########################################
# Options for embedding/classification #
########################################

# embedding
sim = 'NPMI1s'          # similarity measure ('NPMI1s', 'PMIs', 'conditional_prob')
delta = 0.0             # smoothing parameter for PMIs only (>= 0)
embedding = 'adj'       # embedding type ('adj', 'adj+diag', 'normlap', 'regnormlap')
max_eig = 200           # maximum number of eigenvectors to compute
which_elbow = 2         # which elbow of the scree plot to set threshold (if 0, keeps all eigenvectors)

# PCA
use_pca = False        # use PCA after stacking features
max_eig_pca = None      # maximum number of eigenvectors to compute in PCA (if None, compute all)
whiten = True           # whiten the PCA (ensure uncorrelated outputs with unit component-wise variances)

# feature handling
sphere_context = True   # project context embedding vectors to the unit sphere
sphere_content = False  # project content embedding vectors (text features) to the unit sphere
imputation = 'mean'     # strategy for imputation of missing numeric data ('mean' or 'median')

# classification
classifier = 'logreg'   # supervised classifier ('logreg', 'naive_bayes', 'randfor', 'boost', 'kde')
num_trees = 100         # number of decision trees to use if classifier is randfor or boost
cv_max = 10            # max number of seeds to sample for cross-validation (if 0, no cross-validation)

# KDE bandwidth selection
kde_cv_gridsize = 31 #101       # number of bandwidth values (centered around Silverman's approximation) to try for gridsearch
kde_cv_dynamic_range = 100#100  # max bandwidth / min bandwidth to try
kde_cv_folds = 10           # number of folds to use in bandwidth cross-validation


#######################################
# Options for random walk / diffusion #
#######################################

combination_style = 'fusion'   # how to combine attributes ('fusion', 'mean')
score_fusion_style = 'mean'  # if style is fusion, how to combine scores ('max', 'mean')

###########################
# Options for random walk #
###########################

randomwalk_steps = 8          # how many steps of random walk to perform
randomwalk_score_style = 'arith'  # whether to consider scores as probs ('arith') or log-probs ('geom')

#########################
# Options for diffusion #
#########################

diffusion_rate = .0001   # rate at which to update temperature every time step
diffusion_max_iters = 1000  # max number of diffusion iterations
diffusion_tol = 1e-8       # convergence threshold for temperatures
diffusion_bias = True      # weight pos. and neg. samples unequally (i.e. weight is proportional to 1 / #samples for each label)
