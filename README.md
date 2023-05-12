# hic_analysis_LR

This repository contains the scripts used in the analysis of the long range interactions in Verticillium.
To run the analysis you need to provide a list of input files for metadata annotation of the genome (centromeres, AGR, core) location of the Hi-C bins on the genome and epigenetic marker peaks among others.
The long range colocalization data should be provided after normalization and correction following the HiC-explorer pipeline and exported as tsv/csv. 

The circos plot script is designed to create the circos plots that show the AGR association with LR interactions and the strong association with genomic duplications.
