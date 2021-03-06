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

24. [A Re-Evaluation of KGC methods](https://arxiv.org/pdf/1911.03903.pdf)
- evaluation protocol gives perfect score if they output a constant
- create protocol for comparison across **all score functions**
- highest conentration on NN embedding methods - they appear to be outputting the same score for a lot of valid and invalid triplets - due to the fact that aftet the ReLu activation function a lot of the neurons get set to 0 
25. [Toward Understanding The Effect Of Loss function On Then
Performance Of Knowledge Graph Embedding](https://arxiv.org/pdf/1909.00519.pdf)
- TransE has known to have limitation due to its inability to encode certian patterns e.g. symmetric, reflexive etc
- People have attempted to fix this by changing the **scoring** funciton and making variations of the model
- However this paper concentrates on improving it based on the **loss** function: selection of loss functions affect the boundary of the scorign function and hence the limitations that were mentioned in previous research could be deemed as inaccurate because they failed in taking the loss funciton into consideration
- They also suggest **TransComplEx** that translate head entities to the conjugate of the tail entity using relation in complex space - allows for less limitations in relation patterns; the conjugate tail vector allows for distinguishing between the role of an entity as subject or object. Appears to improve ComplEX method 

26. [Benchmark and Best Practices for Biomedical Knowledge Graph
Embeddings] (https://arxiv.org/pdf/2006.13774.pdf)
- suggests **mean quantile** as being more robust than MRR and MR but haven't seen this anywehre else (?)
-  mention *relation prediction* as a potential way of evaluating the model's relation representation directly - ComplEX and DistMult perfom worse thant SimplE and RotatE.
- encouage analysis beyond standard evaluation metrics - e.g. visualisations, reporting metrics for different relation groupings (e.g. many-to-many, many-to-one etc)



**Interpretability/explainability of knowledge graphs**
Believe this could be useful given the setting in which this link prediction etc is set. Would be nice to give, other than quantitative information, also qualitative information. 

1. [Investigating Robustness and Interpretability of Link Prediction
via Adversarial Modifications](https://arxiv.org/pdf/1905.00563.pdf)
2. [Modeling Paths for Explainable Knowledge Base Completion](https://pdfs.semanticscholar.org/6666/235c98c0926322ba501592fed30a585e6603.pdf)
3. [OXKBC: Outcome Explanation for Factorization Based Knowledge Base Completion](https://pdfs.semanticscholar.org/7cbc/9da236434bdbce325baed7a2806f248efa33.pdf)
3. [A Simple Approach to Case-Based Reaoning in Knowledge Bases] (https://openreview.net/pdf?id=AEY9tRqlU7)
- no training
- finds mutliple graph patterns that connect similar source entities through the given relation and looks for pattern matches starting from the query source
- CASE BASED REASONING IS DIVIDED IN:
    1. *retrieve*: retrieve similar cases to the given problem
    2. previous solutions are *reused* for problem at hand
    3. sometimes retrieves solutions can not be immediately reused, and therefore may have to be *revised*
    4. if solutions are useful they are *retained*
- How it works:
    1. Given an entity and query relation (e_q, r_q), it retrieves K entities which are similar to e_q and for which we observe the relation r_q. These entities are not restricted to be in the proximity of the query entity. 
    2. Similarity measured on relations that entities participate in 
    3. Find reasoning paths that connect similar entities to the query entity via the query relation r_q
    4. Check if similar reasoning paths exist starting from the query entity
    5. if similar paths exist, then answer is found by traversing the KG with such reaosning path 
    6. essentially - finds multiple reasoning paths that lead to different entities. **Ranks** such entities by the **number of reasoning paths that lead to them**. 
- Note that they are adding **inverse** relations - this tbh is something I don't understand why people tend to do. It is not true that a -> b, then b -> a... but for some reason appears to improve results?
- only stores a random sample of paths connecting two entities - storing all paths would be intractable. They looked at paths of length 3.
2. [Investigating Robustness and Interpretability of Link Prediction via Adversarial Modifications](https://www.aclweb.org/anthology/N19-1337.pdf)
- add in or remove links such that the prediction of the model changes
- helps identify the most important relations 
**Attacks**:
  - remove neighrbouring link from target, thus identifying most important related link
  - add new fake fact
  https://github.com/pouyapez/criage


