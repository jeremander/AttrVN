# Vertex Nomination with Attributes

This software performs vertex nomination (VN) on richly attributed graphs (RAGs). Currently, it can handle an arbitrary number of discrete or continuous attributes on an undirected graph.

## The Technique

Given a graph, a binary attribute of interest, and a specification of some "seed" vertices with or without this attribute, vertex nomination will rank the remaining (non-seed) vertices by descending probability of having the attribute.

This implementation takes a supervised learning approach, treating the seeds as labeled training samples for a binary classifier. Each vertex is assigned a vector of real-valued features based on "context" (position in the graph) as well as "content" (additional vertex attributes). 

Context features are created in the following way: the adjacency matrix is embedded in Euclidean space via spectral decomposition. Content features are created for each content attribute in one of two ways, depending on whether the attribute is discrete or numeric. 

If discrete, a similarity score (the normalized pointwise mutual information, or NPMI) is assigned to each pair of attributes occurring in an adjacent pair of vertices in the graph, so that a higher score implies greater co-occurrence of the attributes. These are then converted into a similarity matrix on vertices, which is then embedded with a spectral decomposition. Note: this procedure is robust to the case of missing or multiple attributes.

Numeric attributes, on the other hand, are taken as-is.

Once features are collected for each attribute type, the features are stacked. Various operations such as sphere-projecting and PCA are available here. After a classifier is trained on the seed vertices, it performs a soft (probabilistic) classification on all of the remaining nodes and ranks them by decreasing probability.


## API Reference

Requirements:
- Python3
- numpy, scipy
- matplotlib
- pandas
- scikit-learn

The top-level `AttrVN` directory contains all the libraries and scripts needed to run vertex nomination. To feed in the data and parameters, create a directory (in our included Google+ example, this is `gplus0_sub`), which should include three files:

1) An edge file (pairs of integer indices on each line, separated by whitespace).

2) An attribute file (three-column semicolon-delimited table whose first line is `node;attributeType;attributeVal` and each subsequent line is of the form `index;type;value`.

3) A file called `params.py` specifying all of the algorithm parameters. Crucially, this includes the data types for each attribute type, as well as the designation of one attribute type and attribute value to nominate. The VN "seeds" are considered to be all vertices in the graph for which this designated attribute type is specified.

Once such a directory exists containing the data and parameters, the vertex nomination algorithm can be run in two stages using the following scripts:

1) `embed.py`: Performs the embeddings on the graph and/or discrete attributes and saves off each feature matrix as a Python pickle file, along with other (optional) output. This procedure only needs to be done once.

Usage: `python3 embed.py gplus0_sub`

2) `nominate.py`: Stacks the feature matrices, performs any preprocessing, identifies the seed vertices and trains a binary classifier on these, performs cross-validation (optionally) to demonstrate performance of the classifier, and outputs a ranked nomination list.

Usage: `python3 nominate.py gplus0_sub`


## Example Dataset

The original dataset comes from a public crawl of Google+ in its early history. It contained about 4.7 million vertices and about 33 million edges. Each vertex possessed up to four free-form attribute types: `employer`, `major`, `places_lived`, and `school`. A user could provide multiple values for each of these attributes, or no value. In fact, at the time of the crawl, only about 20% of users included any attributes in their profile. For each attribute type, there were over 100,000 unique text strings provided in the dataset, and their distributions tended to obey a power law.

The graph embeddings each took several hours to perform on a MacBook Pro (such speed was enabled by sparse treatment of the data); supervised training and nomination took minutes. To make this process even faster in a demo setting, the `gplus0_sub` directory contains only a subgraph (50,000 vertices) of the original data set. This subgraph is fully connected, however.

