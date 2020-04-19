# Aladdin
Informaiton and code for Machine Learning MSc dissertation project at UCL, in collaboration with Aladdin Healthcare Technologies 


A list of possible useful resources:

1. [Drug-drug interaction prediction based on knowledge graph embeddings and convolutional-LSTM network](https://research.vu.nl/en/publications/drug-drug-interaction-prediction-based-on-knowledge-graph-embeddi)
2. [deep-DR (not plublished yet - this would be for drug repurposing)](https://github.com/ChengF-Lab/deepDR)
3. [HetioNet](https://elifesciences.org/articles/26726)
4. [Biomedical knowledge bases](https://openreview.net/pdf?id=B1gGyLFEDV)
5. [DTI](https://www.researchgate.net/publication/321428613_DDR_Efficient_computational_method_to_predict_drug-Target_interactions_using_graph_mining_and_machine_learning_approaches)
6. [dt12](https://www.frontiersin.org/articles/10.3389/fphar.2018.01134/full)
7. [Poster: Knowledge Graph Completion to Predict Polypharmacy Side Effects](https://arxiv.org/pdf/1810.09227.pdf)
8. [You CAN Teach an Old Dog New Tricks! On Training Knowledge Graph Embeddings](https://openreview.net/forum?id=BkxSmlBFvr)
9. [Predicting Adverse Drug-Drug Interactions with Neural Embedding of Semantic Predications](https://www.biorxiv.org/content/10.1101/752022v3.full)

- **Embedding Semantic Predications**: deriving distributed vector representations from structured biomedical knowledge
- Identifying side effects using link prediction problem as done by Decagon = *multi realtional link prediciton task*.
- not only predicts the **presence** of an interaction, but also the **nature** of the interaction (i.e. side effect) --> decagon was the first to model *different* polypharmacy side effects
- Decagon models drugs as vectors, side effects as matrices, hence they exist in a different space - making them challenging to reuse?
- ESP uses a **neural network** to generate biomedical concepts from concet-relation-concept triples that they call **predications**
- uses transient embeddings for **composite** concepts from their component vectors(?)
- during training embeddings are updated such that vectors for concepts appearing in predications with similar  predicate-argument pairs become more similar (closer in vector space) to each other. Dissimlar vectors will approach orthogonality. 
- binding operator used is elementwise XOR
- binary vectors? - **bit vectors**
- superposition vector between multiple vectors retains the most common element (1 or 0) amongst all vectors
- linearly decreasing learning rate
- non-negative normalised hamming distance
- optimisation objective is the cross-entropy function
- **question** - they are defining how updates are made to the weight vectors representing the subject, predicate and object. However they have an objective which they achieve through gradient descent - therefore the weigths should ideally be updated by backpropagation? - unless the update that is written is indeed tue to the backpropagation
- the **bound product** of the contexta and semantic embeddings of the two drugs invovled in polypharmacy side effect, should be similar to the side effect they produce
- they create semantic and context embeddings for each drug - i.e. depending wehtehr they appear as predicate or object in the triple



10. [Realistic Re-evaluation of Knowledge Graph Completion Methods: An Experimental Study](https://arxiv.org/abs/2003.08001)
11. [Knowledge Graph Completion to Predict Polypharmacy Side Effects](https://arxiv.org/abs/1810.09227)

12. Sameh's Thesis
- Study KGE and analyse their training pipeline + investigate effects of different trainign components (could take inspiration here to obtain better neural predictors). Different training objectives/negative sampling have different effects in accuracy and scalability 
- **TriVec** model: multiple vectors to model embedding interactions
- When comparing with other SOTA models he uses **MRR and Hits10** metrics. When comparing for the biological tasks, he uses the **AUC** metric.
- Suggests improvements have concentrated on scoring functions but not so much on loss functions, negative sampling etc
- KGE models are extremely sensitive to the different trainign parameters
- TriVec model extend tensor factorisation from ComplEX and DistMult - changes embedding representation and embedding interaction
- mostly inspired by ComplEX - but does not use complex parts - but rather embeddings of three different parts
- interaction of 3 part embeddings: one symmetric interaction, and two asymmetric ones
- showed that squared loss can enhance ComplEX performance on multiple tasks
- **for polypharmacy they use the held out test as defined by the decagon paper, and only report the held out test results which are comparable with state-of-the-art methods**



13. [Modelin Polypharmacy Side Effects Using Graph CNN (DECAGON PAPER)](https://academic.oup.com/bioinformatics/article/34/13/i457/5045770)

14. [Survey of different KG completion methods](https://persagen.com/files/misc/Wang2017Knowledge.pdf)
15. [Evaluation of knowledge graph embedding approaches for drug-drug interaction prediction in realistic settings](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3284-5) 
- DDI predictions report high accuracy however can be due to the systematic bias found in the network - so would report a high accuracy which however would not be realistic
- their method uses disjoint cross validation
- used embeddings from RDF2Vec, TransD and TransE as input to Log Reg, NB, RF etc on DrugBank - RDF2Vec was better
This could be interesting to read with respect to the realistic re-evaluation of kg datasets.
- *adresses evaluation biases by introducing drugwise and pairwise disjoint test classes*
- traditional CV: test pair may share components with training pairs. This could lead to overfitting
- disjoint cv = separates data into grouping according to 'first component of the pair'. Done such as to evaluate predictions on drugs that have no DDI information in the training set (cold start drugs)
- test both for prediciton of interaction between a cold start drug and existing drug, as well as interaction between two cold start drugs


- DrugBANK, KEGG, PharmaGKB were used as background KG. Removed DDI links.

16. [GrEDeL: A Knowledge Graph Embedding Based Method for Drug Discovery From Biomedical Literatures](GrEDeL: A Knowledge Graph Embedding Based Method for Drug Discovery From Biomedical Literatures)
17. [ELPKG: A High-Accuracy Link Prediction Approach for Knowledge Graph Completion](https://www.semanticscholar.org/paper/ELPKG%3A-A-High-Accuracy-Link-Prediction-Approach-for-Ma-Qiao/f9a26d39947c90f7ca432e79f4ce1668061197f9)
18. [Learning Hierarchy-Aware Knowledge Graph Embeddings for Link Prediction](https://arxiv.org/pdf/1911.09419.pdf)
19. [Neural networks for link prediction in realistic biomedical graphs: a multi-dimensional evaluation of graph embedding-based approaches](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2163-9)
- looks at evaluating how embeddings from DeepWalk, node2vec, LINE, etc perform as input to neural link predictor
- does so by evaluating on more *realistic* graphs - where in terms of *realistic settings* it introduces *time-slicing* - it's an interesting area of reasearch, but unsure how this would be relevnat to polypharmacy
- evaluates also on what it believe to be *better evaluation metrics*: weigh performance at each node equally as well as not, to illustrate the different aspects of predictor's performance
- from what I gather though its always a single relationship (homogenous graphs) rather than heteorgenous graphs
- metrics = AUPR - how increasing recall affects precision. Serves to help calculate *link-equality* metrics. AUC - also linke equality. Measures values across all recall levels (guess it means all thresholds). Precision@K (ercentage of true positives among only the top k ranked links.)- also link equality; *mean average precision* - given ranked list of predicted links for partiuclar node, calculates the precision after each true positive. The average of this gives average precision for a node. Take mean overa all nodes. *node-equality* measures; Average relevant precision (similar thing) - also node equality. 

20.[Systematic Biases in Link Prediction: comparing
heuristic and graph embedding based methods](https://arxiv.org/pdf/1811.12159.pdf) 
Potentially may be interesting for a deeper exmaination of the type of predictions created

21. [Canonical Tensor Decomposition in KC]()
- uses broaded hyperparams search dueto hyperparameter sensitivity
- evaluates effect of regularisation: introduce **nuclear-p norms** for more principled regularisers
- learns and predicts with the **inverse of predicates** and this seems to improve performance - don't think sameh tested this. Unsure also as to why they would test it this way - but then again somehow this may be affected by their leakage?
- **tensor representation**: N X P X N tensor with N being the number of entities, P the number of predicates. Its binary, with 1 where a certain relation exists. 
- **canonical decomposition tensors**
- both DistMult and ComplEX have the subject and object entities share the same factors (except with complex representing one as the complext conjugat of the other): slightly confused as to what **sharing the same factors** means.
- uses **full multi-class log-loss** rather than sampled multi-class or binary logistic regression
- trace norm has been proposed as a convex relaxation of the rank for matrix completion - when samples are not taken unifromly at random other norms may be preferable than the nuclear norm ''weighted trace norm reweights elements of the factorsbased on the marginal rows and columns sampling probabil-ities, which can improve sample complexity bounds whensampling is non-uniform''
- **tensor trace norm** - but costly with multiclass log loss. But because nuclear norms have been succesfull for matrices - they attempt to create similar ones for tensors
- they create a concatenation of the reciprocal of the predicate with the predicate **I don't understand why - is this not suggesting symmetric relationships/or assuming that predicate reciprocals acutally exist?** - but makes it invariant to existene of reciprocal in the KG or not


22. [Knowledge Graph Embedding: A Survey of
Approaches and Applications](https://persagen.com/files/misc/Wang2017Knowledge.pdf)
23. [Network embedding in biomedical data science 
](https://academic.oup.com/bib/article/21/1/182/5228144#198666838)
