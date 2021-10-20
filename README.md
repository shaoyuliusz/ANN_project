# Advanced Algorithm Final Project

## Introduction

This is the course project GitHub repository for COMSW4995-008 Advanced Algorithm. We chose an implementation-based project, and applied Nearest Neighbor Search and Similarity Search algorithms to a real-world dataset. 

We plan to explore Scopus, the largest abstract and citation database provided by Elsevier in 2004. The interface of our algorithm would be to output N most similar articles on an input article, based on the similarity of their abstracts.

This repo provides helpful utility methods to create embeddings for words and sentences, either by directly loading pre-trained models or through training a fresh neural network from the ground. More importantly, it also provides implementation for the following algorithms: vanilla LSH via the FALCONN package, Neural LSH, Graph-based partitioning and Hierarchical NSW. For the course project, we implemented all four methods and compared their results.

## Authors
- Yuanchu Dang
- Yunfeng Guan
- Shaoyu Liu
- Chengrui Zhou

## Code structure
- **embeddings** contains utility functions to construct vectors out of sentences, from pretrained sentence embedding models based on SBERT, Reimers and Gurevych (2019).
- **falconn** contains the Falconn implementation of LSH as a baseline comparison.
- **hnsw** contains the implementation of Hierachical Navigable Small World (HNSW) algorithm based on Malkov and Yashunin (2020)
- **dong_prototype** contains the implementation of Neural LSH. 
- **data** contains a sample data csv. 
