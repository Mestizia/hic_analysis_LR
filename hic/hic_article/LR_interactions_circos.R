library(circlize)
library(dplyr)
library(tidyr)
circos.clear()

#This example script works for Verticillium dahliae JR2 genome, adapt it to your genome of interest (chromosome numbers, chromosome size etc..)
sectors = c('Chr1', 'Chr2', 'Chr3', 'Chr4', 'Chr5', 'Chr6', 'Chr7', 'Chr8')
chromosomes <- data.frame('Chr'=c('Chr1','Chr2','Chr3','Chr4','Chr5','Chr6','Chr7','Chr8'),
                          'start'=c(1,1,1,1,1,1,1,1),'end'=c(9275483,4277765,4168633,4086908,4171808,3530890,3277570,3361230),
                          'type'=c('chr','chr','chr','chr','chr','chr','chr','chr'))

circos.par(start.degree = 90)
circos.initialize(chromosomes$chr, xlim = cbind(rep(0, 8), chromosomes$sizes))
circos.genomicInitialize(chromosomes)

#load data for the edges
binID <- read.csv('/path/to/genome_BINS_annot.csv', sep='\t', header=1, colClasses = c(NA, 'NULL', NA, NA, NA, NA, 'NULL'), row.names='chr_start_stop')
centedges <- read.csv('/path/to/LR_interactions_including_centromeres_df.csv', sep='\t', colClasses = c(NA, NA, NA))
edges <- read.csv('/path/to/LR_interactions_excluding_centromeres_df.csv', sep='\t', colClasses = c('NULL', NA, NA, NA, NA))
duplications <-  read.csv('/path/to/genomic_duplications.csv', sep='\t', colClasses = c('NULL',NA,NA,NA,NA,NA,NA,NA,NA), header=1)

# make mixed circle with AGR and centromeres
AGR <- read.csv('/path/to/agr_locations.csv', sep='\t')
cent <- read.csv('/path/to/centromeres_locations.csv', sep='\t')
agr_cent <- list(AGR, cent)
col_fun = colorRamp2(c(1, 2), c("#0055b3", "#FED000"))
circos.genomicTrackPlotRegion(agr_cent,ylim=c(0, 1),bg.col="#FFFFFF",track.height=0.07, track.margin=c(0,0),
                              cell.padding=c(0,0,0,0),
                              panel.fun= function(agr_cent,value, ...){
                                i = getI(...)
                                circos.genomicRect(agr_cent,value,col=col_fun(i), border=col_fun(i), ...)
                              })
#load and plot metadata on outer circle
H3K27AC <-read.csv('/path/to/metadata_peaks/H3K27AC.PEAKS.bed', sep='\t', header=0, colClasses = c(NA, NA, NA, "NULL", "NULL", "NULL"))
H3K27M3 <-read.csv('/path/to/metadata_peaks/H3K27M3.PEAKS.bed', sep='\t', header=0, colClasses = c(NA, NA, NA, "NULL", "NULL", "NULL"))
H3K4M2 <-read.csv('/path/to/metadata_peaks/H3K4M2.PEAKS.bed', sep='\t', header=0, colClasses = c(NA, NA, NA, "NULL", "NULL", "NULL"))
H3K9M3 <-read.csv('/path/to/metadata_peaks/H3K9M3.PEAKS.bed', sep='\t', header=0, colClasses = c(NA, NA, NA, "NULL", "NULL", "NULL"))
circos.genomicDensity(H3K27AC, col = c("#FF000080"), track.height = 0.05)
circos.genomicDensity(H3K27M3, col = c("#FF000080"), track.height = 0.05)
circos.genomicDensity(H3K4M2, col = c("#FF000080"), track.height = 0.05)
circos.genomicDensity(H3K9M3, col = c("#FF000080"), track.height = 0.05)

#ADD AGR to outer circle
AGR <- read.csv("/path/to/AGR/regions/saturatedLearning.consensus20K.bed", sep='\t')
circos.genomicTrackPlotRegion(AGR,ylim=c(0,1),bg.col="#D3D3D3",track.height=0.07, track.margin=c(0,0),
                              cell.padding=c(0,0,0,0),
                              panel.fun= function(AGR,value, ...){
                                circos.genomicRect(AGR,value,col='#0055b3',border='#0055b3', ...)})

#add outer circles with genes and tes
genes <- read.csv('/path/to/metadata/JR2_cog_genes.csv', sep=',', header=1, colClasses = c("NULL", "NULL", NA, NA, NA, "NULL", "NULL", "NULL","NULL", "NULL", "NULL", "NULL", "NULL"))
circos.genomicDensity(genes, col = c("#000000"), track.height = 0.07)
te <- read.csv('/path/to/metadata/JR2_TE.bed', sep='\t', header=0, colClasses = c(NA, NA, NA, "NULL", "NULL", "NULL", NA))
circos.genomicDensity(te, col = c("#000000"), track.height = 0.07)

#LR interactions without centromeres
for (edge in row.names(edges)) {
  snode <- as.matrix(edges$level_0)[as.numeric(edge)]
  schr <- as.matrix(binID[snode, ])[1]
  sloc  <- as.integer(as.matrix(binID[snode, ])[2])
  tnode <- as.matrix(edges$level_1)[as.numeric(edge)]
  tchr <- as.matrix(binID[tnode, ])[1]
  tloc  <- as.integer(as.matrix(binID[tnode, ])[2])
  if (as.matrix(binID[snode, ][4]) != 'centromere' & as.matrix(binID[tnode, ][4]) != 'centromere') {
    if (as.matrix(binID[snode, ][4]) == 'core' | as.matrix(binID[tnode, ][4]) == 'core') {edge_color= "#4d4d4d"}
    if (as.matrix(binID[snode, ][4]) == 'AGR' & as.matrix(binID[tnode, ][4]) == 'AGR') {edge_color = "#0055b3"}
    if (as.matrix(binID[snode, ][4]) == 'AGR' | as.matrix(binID[tnode, ][4]) == 'AGR') {edge_color= "#0055b3"}
    tran <- 1-as.matrix(edges$X0)[as.numeric(edge)]/max(as.numeric(centedges$X0))
    circos.link(schr, sloc, tchr, tloc, col=(add_transparency(edge_color, tran)))}
}

#LR interactions with centromeres
for (edge in row.names(centedges)) {
  snode <- as.matrix(centedges$level_0)[as.numeric(edge)]
  schr <- as.matrix(binID[snode, ])[1]
  sloc  <- as.integer(as.matrix(binID[snode, ])[2])
  tnode <- as.matrix(centedges$level_1)[as.numeric(edge)]
  tchr <- as.matrix(binID[tnode, ])[1]
  tloc  <- as.integer(as.matrix(binID[tnode, ])[2])
  edge_color = "#FED000"
  if (as.matrix(binID[snode, ][4]) == 'core' | as.matrix(binID[tnode, ][4]) == 'core') {edge_color= "#4d4d4d"}
  if (as.matrix(binID[snode, ][4]) == 'AGR' & as.matrix(binID[tnode, ][4]) == 'AGR') {edge_color = "#0055b3"}
  if (as.matrix(binID[snode, ][4]) == 'AGR' | as.matrix(binID[tnode, ][4]) == 'AGR') {edge_color= "#0055b3"}
  tran <- 1-as.matrix(edges$X0)[as.numeric(edge)]/max(as.numeric(centedges$X0))
  circos.link(schr, sloc, tchr, tloc, col=(add_transparency(edge_color, tran)))
}

#plot duplications on circos
for (dup in row.names(duplications)) {
  schr <-as.matrix(duplications[dup, ])[3]
  sloc1 <-as.integer(as.matrix(duplications[dup, ])[1])
  sloc2 <-as.integer(as.matrix(duplications[dup, ])[2])
  echr <-as.matrix(duplications[dup, ])[6]
  eloc1 <-as.integer(as.matrix(duplications[dup, ])[4])
  eloc2 <-as.integer(as.matrix(duplications[dup, ])[5])
  edge_color = '#4d4d4d'
  circos.link(schr, c(sloc1, sloc2), echr, c(eloc1, eloc2), col=edge_color)
}