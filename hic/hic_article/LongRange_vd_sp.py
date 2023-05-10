#long list of imports
import os
import time
import scipy
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import MultiIndex
import subprocess
import random
import statsmodels.stats.multitest
from scipy.stats import fisher_exact
from scipy.stats import chisquare
from scipy.stats import ranksums
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
from scipy.stats import ttest_1samp
from statsmodels.stats.weightstats import ztest
import argparse

def make_2d(df, bins, binID, outputDir, referenceSampleName):
    # create a dictionary to store the data
    df_dict = {}
    # create a list to store the bin order
    bin_order = []
    # loop over the dataframe
    for i in df.index:
        # get the coordinates for the first bin
        r1 = df.loc[i, ['chrom1', 'start1', 'end1']]
        # get the coordinates for the second bin
        r2 = df.loc[i, ['chrom2', 'start2', 'end2']]
        # combine the coordinates into a string
        R1 = '_'.join(str(r) for r in r1)
        R2 = '_'.join(str(r) for r in r2)
        # if the first bin is not in the order list then add it
        if not R1 in bin_order:
            bin_order.append(R1)
        # add the data to the dictionary
        df_dict[(R1, R2)] = df.loc[i, 'count']
    # create a series from the dictionary
    s = pd.Series(df_dict, index=MultiIndex.from_tuples(df_dict))
    # create a dataframe from the series
    df_dict = s.unstack()
    # make the dataframe symmetrical
    DF = df_dict.combine_first(df_dict.T).fillna(0)
    # sort the dataframe by the bin order
    DF = DF.loc[bin_order, bin_order]
    # save the dataframe as a tsv file
    DF.to_csv('{}df_{}.tsv'.format(outputDir, referenceSampleName), sep='\t')
    # return the dataframe
    return DF

def get_LR_bins(bins):
    # for each bin create list of LR cis bins and trans bins
    bins = list(bins)
    chr = ''
    cis_LR_dict = {}
    trans_LR_dict = {}
    for bin1 in bins:
        # for each bin, get start and end position
        binstart = int(bin1.split('_')[1])
        binend = int(bin1.split('_')[-1])
        if bin1.split('_')[0] != chr:
            # if it's a new chromosome, update chr variable and create chromosome bin list
            chr = bin1.split('_')[0]
            chrom_bins = [x for x in bins if x.split('_')[0] == chr]
            trans_bins = [x for x in bins if x not in chrom_bins]
        # in the dictionary with the bins in trans, add all bins on other chr
        trans_LR_dict[bin1] = trans_bins
        cis_LR_list = []
        for bin2 in chrom_bins:
            # check if other bins on the chromosome are far enough [threshold=3000]. If so, classify they as long range
            bin2start = int(bin2.split('_')[1])
            bin2end = int(bin2.split('_')[-1])
            if abs(binstart - bin2end) > 30000:
                cis_LR_list.append(bin2)
            elif abs(binend - bin2start) > 30000:
                cis_LR_list.append(bin2)
        cis_LR_dict[bin1] = cis_LR_list

    return cis_LR_dict, trans_LR_dict

def count_LR_obs_exp(obsexp, cis, trans, annot, _binID):
    #counts the LR obs/exp for each bin
    tot_df = pd.DataFrame(0, index=annot.index, columns=['cis', 'trans'])

    for bin in obsexp.index:
        cis_counts = obsexp.loc[bin, cis[bin]]
        trans_counts = obsexp.loc[bin, trans[bin]]
        tot_df.loc[_binID.loc[bin, 3], 'cis'] = cis_counts.sum()
        tot_df.loc[_binID.loc[bin, 3], 'trans'] = trans_counts.sum()

    new_index = [_binID.loc[x, 3] for x in obsexp.index]
    obsexp.columns = new_index
    obsexp.index = new_index
    annot.dropna(how='any', inplace=True)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    sns.histplot(data=tot_df.loc[annot.index], x='cis', hue=annot.loc[:, 'bintype'], ax=ax1, multiple="stack")
    sns.histplot(data=tot_df.loc[annot.index], x='trans', hue=annot.loc[:, 'bintype'], ax=ax2, multiple="stack")
    plt.show()

def show_bins(bins):
    # create mapping between bins_id and genome locations
    _df = pd.DataFrame(0, index=bins.index, columns=['chr_start_stop'])
    for bin in bins.index:
        _df.loc[bin, 'chr_start_stop'] = '{}_{}_{}'.format(bin.split('_')[-1], bins.loc[bin, 1], bins.loc[bin, 2])
    _df.reset_index(inplace=True)
    _df.set_index('chr_start_stop', inplace=True)
    return bins, _df

def annotate_bins(binID, bins, centromeres, agr, chr):
    #annotate bins as core, AGR or centromere
    annot = pd.DataFrame(index=binID.index, columns=['chr', 'start', 'stop', 'bintype'])
    for bin in binID.index:
        bindata = binID.loc[bin]
        AGR = isAGR(bin, agr, bindata, chr, annot, binID)
        if not AGR:
            centromere = isCentromere(bin, centromeres, bindata, chr, annot)
            if not centromere:
                annot.loc[bin, ['chr', 'start', 'stop', 'bintype']] = [binID.loc[bin, 0], binID.loc[bin, 1], binID.loc[bin, 2], 'core']

    for chr in set(annot.loc[:, 'chr']):
        _annot = annot[annot.loc[:, 'chr'] == chr]
        if commandLineArgs['verbose']:
            print(list(_annot.loc[:, 'bintype'].values))

    annot.to_csv('{}{}_bintype_annot.csv'.format(outputDir, sample_name), sep='\t')
    return annot

def annotate_centromere(binID, centromeres, annot):
    #annotate bins as centromere
    for bin in binID.index:
        bindata = binID.loc[bin]
        chr = binID.loc[bin, 'Chr']
        centromere = isCentromere(bin, centromeres, bindata, chr, annot)
        if not centromere:
            if annot.loc[bin, 'bintype'] != 'AGR_l':
                annot.loc[bin, ['chr', 'start', 'stop', 'bintype']] = [binID.loc[bin, 'Chr'], binID.loc[bin, 'start'], binID.loc[bin, 'stop'], 'core']
        else:
            annot.loc[bin, ['chr', 'start', 'stop', 'bintype']] = [binID.loc[bin, 'Chr'], binID.loc[bin, 'start'], binID.loc[bin, 'stop'], 'centromere']
    return annot

def isAGR(bin, agr, bindata, chr, annot, binID):
    #check if bin is in AGRegion
    AGR = agr[agr.loc[:, 'Chrom'] == bindata[0]]
    for region in AGR.index:
        regionstart = AGR.loc[region, 'start']
        regionend = AGR.loc[region, 'end']
        if bindata[1] >= regionstart and bindata[1] <= regionend:
            annot.loc[bin, ['chr', 'start', 'stop', 'bintype']] = [binID.loc[bin, 0], binID.loc[bin, 1], binID.loc[bin, 2], 'AGR']
            return True
        elif bindata[2] <= regionend and bindata[2] >= regionstart:
            annot.loc[bin, ['chr', 'start', 'stop', 'bintype']] = [binID.loc[bin, 0], binID.loc[bin, 1], binID.loc[bin, 2], 'AGR']
            return True
    return False

def isCentromere(bin, centromeres, bindata, chr, annot):
    #check if bin is in centromere
    centromere_start = centromeres.loc[chr, 1]
    centromere_end = centromeres.loc[chr, 2]
    if bindata[1] >= centromere_start and bindata[1] <= centromere_end:
        return True
    elif bindata[2] < centromere_end and bindata[2] > centromere_start:
        return True
    return False

def make_nw(obsexp, pairwise_edge_type, _binID, annot, cis_LR_dict, trans_LR_dict, distance_from_centromere):
    #make network
    unstack = obsexp.unstack()
    _IDbin = _binID.reset_index()
    _IDbin.set_index('chr_start_stop', inplace=True)
    _annot = annot.reset_index()
    _annot.index = _IDbin.index
    _annot.columns = ['binID', 'chr', 'start', 'stop', 'bintype', 'distance_from_centromere']
    _annot.iloc[1:].to_csv('{}{}_BINS_annot.csv'.format(outputDir, sample_name), sep='\t')
    top_df = unstack > 1
    _top_df = unstack.loc[top_df]
    trans = {};
    cis = {}
    for i in _top_df.index:
        if i[0].split('_')[0] != i[1].split('_')[0]:
            trans[(i[0], i[1])] = obsexp.loc[i]
        else:
            if i[1] in cis_LR_dict[i[0]]:
                cis[(i[0], i[1])] = obsexp.loc[i]

    cis_df = pd.DataFrame.from_dict(cis, orient="index")
    trans_df = pd.DataFrame.from_dict(trans, orient='index')
    cis_index = pd.MultiIndex.from_tuples(cis_df.index, names=["first", "second"])
    trans_index = pd.MultiIndex.from_tuples(trans_df.index, names=["first", "second"])
    _cis_df = pd.Series(cis_df.values.flatten(), index=cis_index)
    _trans_df = pd.Series(trans_df.values.flatten(), index=trans_index)
    cis_annot = pd.Series(pairwise_edge_type.loc[cis_index].values.flatten(), index=cis_index)
    trans_annot = pd.Series(pairwise_edge_type.loc[trans_index].values.flatten(), index=trans_index)

    _cis = pd.concat([_cis_df, cis_annot], axis=1)
    _trans = pd.concat([_trans_df, trans_annot], axis=1)

    cis_threshold = sorted(_cis.loc[:, 0].values.flatten(), reverse=True)[:int(len(_cis) * 0.1)][-1]
    trans_threshold = sorted(_trans.loc[:, 0].values.flatten(), reverse=True)[:int(len(_trans) * 0.1)][-1]

    _cis.sort_values(by=0, ascending=False, inplace=True)
    _trans.sort_values(by=0, ascending=False, inplace=True)
    df = pd.concat([_cis, _trans], axis=0)
    if commandLineArgs['verbose']:
        print(df)
    df.sort_values(by=0, ascending=False, inplace=True)

    filt_cis = _cis[_cis.loc[:, 0] >= cis_threshold]
    filt_trans = _trans[_trans.loc[:, 0] >= trans_threshold]
    if commandLineArgs['verbose']:
        print(filt_trans)
        print(filt_cis)

    if not os.path.isfile('{}{}_cis_with_dist.csv'.format(outputDir, sample_name)):
        distance_from_centromere.index = _IDbin.index[1:]
        cis_dist = pd.DataFrame(0, index=filt_cis.index, columns=['cis_counts', 'min_distance', 'edge_type'])
        cis_df.index = pd.MultiIndex.from_tuples(cis_df.index)
        for i in cis_dist.index:
            cis_dist.loc[i, 'min_distance'] = min(distance_from_centromere.loc[i[0]],
                                                  distance_from_centromere.loc[i[1]])
            cis_dist.loc[i, 'cis_counts'] = cis_df.loc[i, 0]
            cis_dist.loc[i, 'edge_type'] = filt_cis.loc[i, 1]
        cis_dist.to_csv('{}{}_cis_with_dist.csv'.format(outputDir, sample_name), sep='\t')
    else:
        cis_dist = pd.read_csv('{}{}_cis_with_dist.csv'.format(outputDir, sample_name), sep='\t', index_col=[0, 1])

    cis_out = open('{}{}_cis_cytoscape.csv'.format(outputDir, sample_name), 'w')
    cis_txt = 'bin_1\tbin_2\tcounts\tedge_type\tdist2centromere\n'
    for I in cis_dist.index:
        cis_txt += '{}\t{}\t{}\t{}\t{}\n'.format(I[0], I[1], cis_dist.loc[I, 'cis_counts'], cis_dist.loc[I, 'edge_type'], cis_dist.loc[I, 'min_distance'])
    cis_out.write(cis_txt)

    trans_out = open('{}{}_trans_cytoscape.csv'.format(outputDir, sample_name), 'w')
    trans_txt = 'bin_1\tbin_2\tcounts\tedge_type\n'
    for I in filt_trans.index:
        trans_txt += '{}\t{}\t{}\t{}\n'.format(I[0], I[1], filt_trans.loc[I, 0], filt_trans.loc[I, 1])
    trans_out.write(trans_txt)
    return cis_dist, filt_trans

def check_for_duplication(df, T):
    #checks for duplication
    DUP = 0
    QUE = 0
    matches_dict = {}
    dup_dict = {}
    dup_qual = []

    _DUP = 0
    _QUE = 0
    _matches_dict = {}
    _dup_dict = {}
    ref_dup_qual = []

    for i in df.index.get_level_values('bin_1'):
        matches = []
        _df = df.loc[[i]]
        que_size = 0
        if not os.path.isdir('{}dup/{}/{}'.format(commandLineArgs['outputDir'], T, i)):
            os.mkdir('{}dup/{}/{}'.format(commandLineArgs['outputDir'], T, i))
        if not os.path.isdir('{}dup/ref/{}'.format(commandLineArgs['outputDir'], i)):
            os.mkdir('{}dup/ref/{}'.format(commandLineArgs['outputDir'], i))
        if not os.path.isfile('{}dup/{}/{}/reference-region.txt'.format(commandLineArgs['outputDir'], T, i)):
            ref_file = open('{}dup/{}/{}/reference-region.txt'.format(commandLineArgs['outputDir'], T, i), 'w')
            ref_txt = '{}:{}-{}'.format(i.split('_')[0], i.split('_')[1], i.split('_')[2])
            ref_file.write(ref_txt)
        if not os.path.isfile('{}dup/{}/{}/query-region.txt'.format(commandLineArgs['outputDir'], T, i)):
            query_file = open('{}dup/{}/{}/query-region.txt'.format(commandLineArgs['outputDir'], T, i), 'w')
            query_txt = ''
            for i2 in _df.index.get_level_values('bin_2'):
                matches.append(i2)
                que_size = int(i2.split('_')[2]) - int(i2.split('_')[1])
                query_txt += '{}:{}-{}\n'.format(i2.split('_')[0], i2.split('_')[1], i2.split('_')[2])
            query_file.write(query_txt)
        else:
            for i2 in _df.index.get_level_values('bin_2'):
                que_size += int(i2.split('_')[2]) - int(i2.split('_')[1])

        if not os.path.isfile('{}dup/{}/{}/nucmer.coords'.format(commandLineArgs['outputDir'], T, i)):
            if not os.path.isfile('{}dup/{}/{}/reference-region.fasta'.format(commandLineArgs['outputDir'], T, i)):
                ref_fasta_cmd = 'samtools faidx work/vert/pacbio/GCA_000400815.2_VDAG_JR2v.4.0_genomic.fna -r {}dup/{}/{}/reference-region.txt -o {}dup/{}/{}/reference-region.fasta'.format(commandLineArgs['outputDir'], T, i, commandLineArgs['outputDir'], T, i)
                if commandLineArgs['verbose']:
                    print(ref_fasta_cmd)
                subprocess.check_call(ref_fasta_cmd, shell=True)
                time.sleep(0.1)

            if not os.path.isfile('{}dup/{}/{}/query-region.fasta'.format(commandLineArgs['outputDir'], T, i)):
                query_fasta_cmd = 'samtools faidx work/vert/pacbio/GCA_000400815.2_VDAG_JR2v.4.0_genomic.fna -r {}dup/{}/{}/query-region.txt -o {}dup/{}/{}/query-region.fasta'.format(commandLineArgs['outputDir'], T, i, commandLineArgs['outputDir'], T, i)
                if commandLineArgs['verbose']:
                    print(query_fasta_cmd)
                subprocess.check_call(query_fasta_cmd, shell=True)
                time.sleep(0.1)

            if not os.path.isfile('{}dup/{}/{}/nucmer.delta'.format(commandLineArgs['outputDir'], T, i)):
                nucmer_cmd = 'nucmer --maxmatch --nosimplify --prefix={}dup/{}/{}/nucmer {}dup/{}/{}/reference-region.fasta {}dup/{}/{}/query-region.fasta'.format(outputDir, T, i, outputDir, T, i, outputDir, T, i)
                if commandLineArgs['verbose']:
                    print(nucmer_cmd)
                subprocess.check_call(nucmer_cmd, shell=True)
                time.sleep(0.1)

            if not os.path.isfile('{}dup/{}/{}/nucmer.coords'.format(commandLineArgs['outputDir'], T, i)):
                show_coords_cmd = 'show-coords -r {}dup/{}/{}/nucmer.delta > {}dup/{}/{}/nucmer.coords'.format(commandLineArgs['outputDir'], T, i, commandLineArgs['outputDir'], T, i)
                if commandLineArgs['verbose']:
                    print(show_coords_cmd)
                subprocess.check_call(show_coords_cmd, shell=True)
                time.sleep(0.1)

        if os.path.isfile('{}dup/{}/{}/nucmer.coords'.format(commandLineArgs['outputDir'], T, i)):
            deltafile = open('{}dup/{}/{}/nucmer.coords'.format(commandLineArgs['outputDir'], T, i)).read()
            if len(deltafile.split('NUCMER')[1]) > 170:
                duplicated_edges = parse_deltafile(deltafile)
                matches_dict[i] = duplicated_edges
                for k in duplicated_edges.keys():
                    DUP += len(duplicated_edges[k])
                dup_dict[i] = list(duplicated_edges.keys())
                for k in duplicated_edges.keys():
                    dup_qual.append(float(duplicated_edges[k][0][1]))
        QUE += que_size

    for i in set(df.index.get_level_values('bin_1')):
        if not os.path.isfile('{}dup/ref/{}/nucmer.coords'.format(commandLineArgs['outputDir'], i)):
            if not os.path.isfile('{}dup/ref/{}/nucmer.delta'.format(commandLineArgs['outputDir'], i)):
                nucmer_cmd = 'nucmer --maxmatch --nosimplify --prefix={}dup/ref/{}/nucmer {}dup/{}/{}/reference-region.fasta work/vert/pacbio/GCA_000400815.2_VDAG_JR2v.4.0_genomic.fna'.format(commandLineArgs['outputDir'], i, commandLineArgs['outputDir'], T, i)
                if commandLineArgs['verbose']:
                    print(nucmer_cmd)
                subprocess.check_call(nucmer_cmd, shell=True)

            if not os.path.isfile('{}dup/ref/{}/nucmer.coords'.format(commandLineArgs['outputDir'], T, i)):
                show_coords_cmd = 'show-coords -r {}dup/ref/{}/nucmer.delta > {}dup/ref/{}/nucmer.coords'.format(commandLineArgs['outputDir'], i, commandLineArgs['outputDir'], i)
                if commandLineArgs['verbose']:
                    print(show_coords_cmd)
                subprocess.check_call(show_coords_cmd, shell=True)

        if os.path.isfile('{}dup/ref/{}/nucmer.coords'.format(commandLineArgs['outputDir'], i)):
            deltafile = open('{}dup/ref/{}/nucmer.coords'.format(commandLineArgs['outputDir'], i)).read()
            if len(deltafile.split('NUCMER')[1]) > 170:
                _duplicated_edges = parse_ref_deltafile(deltafile)
                _matches_dict[i] = _duplicated_edges
                _DUP += len(_duplicated_edges.keys()) - 1
                _dup_dict[i] = list(_duplicated_edges.keys())
                for k in duplicated_edges.keys():
                    ref_dup_qual.append(float(_duplicated_edges[k][0][1]))
        # size of the genome, not final value, may need to adjust. I don't feel like doing it now... it works!
        _QUE += 36602118

    if commandLineArgs['verbose']:
        print('{} duplication events in {} bp in interacting set'.format(DUP, QUE))
        print('{} duplication/Kb'.format((DUP / QUE) * 1000))
        print('{} duplication events in {} bp in non-interacting set'.format(_DUP, _QUE))
        print('{} duplication/Kb'.format((_DUP / _QUE) * 1000))

    return matches_dict, _matches_dict, dup_dict, _dup_dict, dup_qual, ref_dup_qual

def _check_for_duplication(df, assembly_dir, sample_name):
    #second function to check for duplication
    df = df.unstack()
    matches_dict = {}
    dup_dict = {}
    dup_qual = []

    _matches_dict = {}
    _dup_dict = {}
    ref_dup_qual = []
    for i in df.index.get_level_values('bin_1'):
        matches = []
        _df = df.loc[[i]]
        que_size = 0
        if not os.path.isdir('{}dup/{}/{}'.format(outputDir, sample_name, i)):
            os.mkdir('{}dup/{}/{}'.format(outputDir, sample_name, i))
        if not os.path.isdir('{}dup/ref/{}'.format(outputDir, i)):
            os.mkdir('{}dup/ref/{}'.format(outputDir, i))
        if not os.path.isfile('{}dup/{}/{}/reference-region.txt'.format(outputDir, sample_name, i)):
            ref_file = open('{}dup/{}/{}/reference-region.txt'.format(outputDir, sample_name, i), 'w')
            ref_txt = '{}:{}-{}'.format(i.split('_')[0], i.split('_')[1], i.split('_')[2])
            ref_file.write(ref_txt)
        if not os.path.isfile('{}dup/{}/{}/query-region.txt'.format(outputDir, sample_name, i)):
            query_file = open('{}dup/{}/{}/query-region.txt'.format(outputDir, sample_name, i), 'w')
            query_txt = ''
            for i2 in _df.index.get_level_values('bin_2'):
                matches.append(i2)
                que_size = int(i2.split('_')[2]) - int(i2.split('_')[1])
                query_txt += '{}:{}-{}\n'.format(i2.split('_')[0], i2.split('_')[1], i2.split('_')[2])
            query_file.write(query_txt)
        else:
            for i2 in _df.index.get_level_values('bin_2'):
                que_size += int(i2.split('_')[2]) - int(i2.split('_')[1])

        if not os.path.isfile('{}dup/{}/{}/nucmer.coords'.format(outputDir, sample_name, i)):
            if not os.path.isfile('{}dup/{}/{}/reference-region.fasta'.format(outputDir, sample_name, i)):
                ref_fasta_cmd = 'samtools faidx {}{}_HiC-improved.final.fasta.masked -r {}dup/{}/{}/reference-region.txt -o {}dup/{}/{}/reference-region.fasta'.format(assembly_dir, sample_name, outputDir, sample_name, i, outputDir, sample_name, i)
                if commandLineArgs['verbose']:
                    print(ref_fasta_cmd)
                subprocess.check_call(ref_fasta_cmd, shell=True)
                time.sleep(0.1)

            if not os.path.isfile('{}dup/{}/{}/query-region.fasta'.format(outputDir, sample_name, i)):
                query_fasta_cmd = 'samtools faidx {}{}_HiC-improved.final.fasta.masked -r {}dup/{}/{}/query-region.txt -o {}dup/{}/{}/query-region.fasta'.format(assembly_dir, sample_name, outputDir, sample_name, i, outputDir, sample_name, i)
                if commandLineArgs['verbose']:
                    print(query_fasta_cmd)
                subprocess.check_call(query_fasta_cmd, shell=True)
                time.sleep(0.1)

            if not os.path.isfile('{}dup/{}/{}/nucmer.delta'.format(outputDir, sample_name, i)):
                nucmer_cmd = 'nucmer --maxmatch --nosimplify --prefix=dup/{}/{}/nucmer {}dup/{}/{}/reference-region.fasta {}dup/{}/{}/query-region.fasta'.format(outputDir, sample_name, i, outputDir, sample_name, i, outputDir, sample_name, i)
                if commandLineArgs['verbose']:
                    print(nucmer_cmd)
                subprocess.check_call(nucmer_cmd, shell=True)
                time.sleep(0.1)

            if not os.path.isfile('{}dup/{}/{}/nucmer.coords'.format(outputDir, sample_name, i)):
                show_coords_cmd = 'show-coords -r {}dup/{}/{}/nucmer.delta > {}dup/{}/{}/nucmer.coords'.format(outputDir, sample_name, i, outputDir, sample_name, i)
                if commandLineArgs['verbose']:
                    print(show_coords_cmd)
                subprocess.check_call(show_coords_cmd, shell=True)
                time.sleep(0.1)

        if os.path.isfile('{}dup/{}/{}/nucmer.coords'.format(outputDir, sample_name, i)):
            deltafile = open('{}dup/{}/{}/nucmer.coords'.format(outputDir, sample_name, i)).read()
            if len(deltafile.split('NUCMER')[1]) > 170:
                duplicated_edges = parse_deltafile(deltafile)
                matches_dict[i] = duplicated_edges
                dup_dict[i] = list(duplicated_edges.keys())
                for k in duplicated_edges.keys():
                    dup_qual.append(float(duplicated_edges[k][0][1]))

    for i in set(df.index.get_level_values('bin_1')):
        if not os.path.isfile('{}dup/{}/{}/nucmer.coords'.format(outputDir, sample_name, i)):
            if not os.path.isfile('{}dup/{}/{}/nucmer.delta'.format(outputDir, sample_name, i)):
                nucmer_cmd = 'nucmer --maxmatch --nosimplify --prefix={}dup/ref/{}/nucmer {}dup/{}/{}/reference-region.fasta {}{}_HiC-improved.final.fasta.masked'.format(outputDir, i, sample_name, outputDir, i, assembly_dir, sample_name)
                if commandLineArgs['verbose']:
                    print(nucmer_cmd)
                subprocess.check_call(nucmer_cmd, shell=True)

            if not os.path.isfile('{}dup/{}/{}/nucmer.coords'.format(outputDir, sample_name, i)):
                show_coords_cmd = 'show-coords -r {}dup/{}/{}/nucmer.delta > {}dup/{}/{}/nucmer.coords'.format(outputDir, sample_name, i, outputDir, sample_name, i)
                if commandLineArgs['verbose']:
                    print(show_coords_cmd)
                subprocess.check_call(show_coords_cmd, shell=True)

        if os.path.isfile('{}dup/ref/{}/nucmer.coords'.format(outputDir, i)):
            deltafile = open('{}dup/ref/{}/nucmer.coords'.format(outputDir, i)).read()
            if len(deltafile.split('NUCMER')[1]) > 170:
                _duplicated_edges = parse_ref_deltafile(deltafile)
                _matches_dict[i] = _duplicated_edges
                _dup_dict[i] = list(_duplicated_edges.keys())
                for k in duplicated_edges.keys():
                    ref_dup_qual.append(float(_duplicated_edges[k][0][1]))

    return matches_dict, _matches_dict, dup_dict, _dup_dict, dup_qual, ref_dup_qual

def make_x(_df, all_info_df, interactive_nodes):
    #makes x square table and checks for statistical significance of association between interacting bins and epigenetics
    if commandLineArgs['verbose']:
        print(_df)
        print(interactive_nodes)
    non_centromere_interacting_nodes = [x for x in interactive_nodes if _df.loc[x, 'edge_type'] != 'centromere_centromere']

    for i in _df.index:
        if i in non_centromere_interacting_nodes:
            if _df.loc[i, 'bintype'] == 'AGR':
                _df.loc[i, 'bin_class'] = 'AGR_in_network'
            elif _df.loc[i, 'bintype'] == 'centromere':
                _df.loc[i, 'bin_class'] = 'centromere_in_network'
            else:
                _df.loc[i, 'bin_class'] = 'core_in_network'
        elif _df.loc[i, 'bintype'] == 'AGR':
            _df.loc[i, 'bin_class'] = 'AGR'
        elif _df.loc[i, 'bintype'] == 'centromere':
            _df.loc[i, 'bin_class'] = 'centromere'
        else:
            _df.loc[i, 'bin_class'] = 'core'

    core_in = len([x for x in all_info_df.loc[non_centromere_interacting_nodes, 'bintype'].values if x != 'AGR'])
    agr_in = len([x for x in all_info_df.loc[non_centromere_interacting_nodes, 'bintype'].values if x == 'AGR'])
    core_out = len(_df[_df.loc[:, 'bin_class'] == 'core'])
    agr_out = len(_df[_df.loc[:, 'bintype'] == 'AGR']) - agr_in
    if commandLineArgs['verbose']:
        print([[agr_in, core_in], [agr_out, core_out]])
    odds, p = scipy.stats.fisher_exact([[agr_in, core_in], [agr_out, core_out]])
    if commandLineArgs['verbose']:
        print(odds, p)

    agr_in_nodes = [x for x in interactive_nodes if all_info_df.loc[x, 'bintype'] == 'AGR']
    core_in_nodes = [x for x in interactive_nodes if
                     (all_info_df.loc[x, 'bintype'] != 'AGR' and x in non_centromere_interacting_nodes)]
    agr_out_nodes = set(_df[_df.loc[:, 'bintype'] == 'AGR'].index) - set(agr_in_nodes)
    core_out_nodes = _df[_df.loc[:, 'bin_class'] == 'core'].index
    core_nodes = list(set(core_out_nodes).union(core_in_nodes))
    network_nodes = list(set(core_in_nodes).union(set(agr_in_nodes)))
    non_centromere_nodes = [x for x in all_info_df.index if all_info_df.loc[x, 'bintype'] != 'centromere']

    for meta in ['H3K27AC1', 'H3K27M3', 'H3K4M2', 'H3K9M3', 'ATAC', 'insulation']:
        sns.boxplot(data=_df.loc[non_centromere_nodes], y=meta, x='bin_class',
                    order=['AGR', 'AGR_in_network', 'core', 'core_in_network'])
        plt.title('{} network'.format(meta))
        if commandLineArgs['verbose']:
            print(meta)
            print(scipy.stats.ranksums(_df.loc[agr_in_nodes, meta].values, _df.loc[agr_out_nodes, meta].values))
            print(scipy.stats.ttest_ind(_df.loc[agr_in_nodes, meta].values, _df.loc[agr_out_nodes, meta].values))
            print(scipy.stats.ranksums(_df.loc[core_in_nodes, meta].values, _df.loc[core_out_nodes, meta].values))
            print(scipy.stats.ttest_ind(_df.loc[core_in_nodes, meta].values, _df.loc[core_out_nodes, meta].values))
        plt.savefig('{}{}_{}_agr_core.png'.format(outputDir, sample_name, meta), dpi=300)
        plt.show()
        plt.close()

    return _df

def find_edges_with_duplications(active, cis_dup_dict, trans_dup_dict):
    #finds edges with duplications
    edges_with_dup = []
    for i in active.index.get_level_values('bin_1'):
        try:
            if cis_dup_dict[i]:
                for i2 in active.loc[[i]].index:
                    _i = i2[1]
                    if _i in cis_dup_dict[i]:
                        edges_with_dup.append((i, _i))
                        # print(active.loc[[i]])
        except KeyError:
            continue

        try:
            if trans_dup_dict[i]:
                for i2 in active.loc[[i]].index:
                    _i = i2[1]
                    if _i in trans_dup_dict[i]:
                        edges_with_dup.append((i, _i))
                # print(active.loc[[i]])
        except KeyError:
            continue
    return edges_with_dup

def individual_LR_obs_exp(obsexp, annot, _binID, output_dir):
    #finds the observed and expected number of edges between each bin type
    if not os.path.isfile('{}{}_edgetype.df'.format(output_dir, sample_name)):
        undf= obsexp.unstack()
        edgetype = pd.Series(0, index=undf.index)
        if commandLineArgs['verbose']:
            print (_binID)
            print (annot)
        for i in undf.index:
            t1 = annot.loc[i[0], 'bintype']
            t2 = annot.loc[i[1], 'bintype']
            if t1 == 'AGR':
                if t2 == 'AGR':
                    edgetype[i] = 'AGR_AGR'
                elif t2 == 'core':
                    edgetype[i] = 'AGR_core_Hybrid'
                else:
                    edgetype[i] = 'AGR_centromere_Hybrid'
            elif t1 == 'core':
                if t2 == 'AGR':
                    edgetype[i] = 'AGR_core_Hybrid'
                elif t2 == 'core':
                    edgetype[i] = 'core_core'
                else:
                    edgetype[i] = 'core_centromere_Hybrid'
            else:
                if t2 == 'AGR':
                    edgetype[i] = 'AGR_centromere_Hybrid'
                elif t2 == 'core':
                    edgetype[i] = 'core_centromere_Hybrid'
                else:
                    edgetype[i] = 'centromere_centromere'
        edgetype.to_csv('{}{}_edgetype.df'.format(output_dir, sample_name),sep='\t')
        sns.histplot(data=edgetype, x='counts', hue='bintype')
        plt.show()
    else:
        edgetype = pd.read_csv('{}{}_edgetype.df'.format(output_dir, sample_name), sep='\t', index_col=[0,1])
    return edgetype

def get_duplications(active, cis_matches_dict, trans_matches_dict, ):
    #finds the number of duplications in the genome
    interactingdups = 0
    cisdups = 0
    transdups = 0
    cis_edges = 0
    trans_edges = 0
    LRdups = 0
    ref_dups = 0
    interactions = 0
    counter = 0

    for bin in active.index.get_level_values('bin_1'):
        counter += 1
        edges = active.loc[[bin]]
        connectsto = [x[1] for x in active.loc[[bin]].index]
        # print (bin)
        try:
            connectsto_cis = [x for x in connectsto if x.split('_')[0] == bin.split('_')[0]]
            bin_dup_in_edges_cis = [x for x in cis_matches_dict[bin].keys() if x in connectsto_cis]
            cisdups += len(bin_dup_in_edges_cis)
            LRdups += len(bin_dup_in_edges_cis)
            cis_edges += len(connectsto_cis)
            if commandLineArgs['verbose']:
                print('cis')
                print(len(connectsto_cis))
                print(len(bin_dup_in_edges_cis))
            # print ('{} duplications in cis out of {} interactions'.format(len(bin_dup_in_edges_cis), len(edges)))
        except KeyError:
            bin_dup_in_edges_cis = []
        try:
            connectsto_trans = [x for x in connectsto if x.split('_')[0] != bin.split('_')[0]]
            bin_dup_in_edges_trans = [x for x in trans_matches_dict[bin].keys() if x in connectsto_trans]
            transdups += len(bin_dup_in_edges_trans)
            LRdups += len(bin_dup_in_edges_trans)
            trans_edges += len(connectsto_trans)
            if commandLineArgs['verbose']:
                print('trans')
                print(len(connectsto_trans))
                print(len(bin_dup_in_edges_trans))
            # print ('{} duplications in cis out of {} interactions'.format(len(bin_dup_in_edges_trans), len(edges)))
        except KeyError:
            bin_dup_in_edges_trans = []
        interactions += len(edges)
        # print ('{} duplications {} cis and {} trans out of {} interacting bins'.format(len(bin_dup_in_edges_cis)+len(bin_dup_in_edges_trans), len(bin_dup_in_edges_cis), len(bin_dup_in_edges_trans), len(edges)))
        try:
            # print ('{} duplications in reference'.format(len(ref_matches_dict[bin])))
            ref_dups += len(ref_matches_dict[bin])
        except KeyError:
            if commandLineArgs['verbose']:
                print('No duplication in reference')

    if commandLineArgs['verbose']:
        print('there are {} duplications in {} interacting cis regions'.format(cisdups, cis_edges))
        print('there are {} duplications in {} interacting trans regions'.format(transdups, trans_edges))
        print('there are {} duplications events in interacting regions out of {} duplications'.format(cisdups + transdups,
                                                                                                      ref_dups))
        print(active)

def _get_duplications(active, cis_matches_dict, trans_matches_dict, ):
    #finds the number of duplications in the genome [second function]
    counted = []
    cisdups = 0
    transdups = 0
    cis_edges = 0
    trans_edges = 0
    LRdups = 0
    ref_dups = 0
    counter = 0

    _active = active.reset_index()
    print(_active)

    for bin_i in _active.index:
        counter += 1
        bin1 = _active.loc[bin_i, 'bin_1']
        bin2 = _active.loc[bin_i, 'bin_2']
        print(bin1, bin2)
        # print (bin)

        if bin1.split("_")[0] == bin2.split("_")[0]:
            cis_edges += 1
            try:
                if bin2 in cis_matches_dict[bin1].keys():
                    cisdups += 1
                    LRdups += 1
            except KeyError:
                continue
        elif bin1.split("_")[0] != bin2.split("_")[0]:
            trans_edges += 1
            try:
                if bin2 in trans_matches_dict[bin1].keys():
                    transdups += 1
                    LRdups += 1
            except KeyError:
                continue
        try:
            if bin1 not in counted:
                ref_dups += len(ref_matches_dict[bin1].keys())
                counted.append(bin1)
        except KeyError:
            continue

    print('there are {} duplications in {} interacting cis regions'.format(cisdups, cis_edges))
    print('there are {} duplications in {} interacting trans regions'.format(transdups, trans_edges))
    print('there are {} duplications events in interacting regions out of {} duplications'.format(cisdups + transdups,
                                                                                                  ref_dups))

def get_bin(chr, st, en, bins):
    #finds the bin that a TE is in [first function]
    b = bins[bins.loc[:, 'Chr'] == chr]
    _i_list = []
    multi_bin = False
    for _i in b.index:
        if st >= b.loc[_i, 'start'] and en <= b.loc[_i, 'stop']:
            # print ('TE fits in bin {} {} is contained in {} {}'.format(st, en,  b.loc[_i, 1],  b.loc[_i, 2]))
            return _i
        elif st >= b.loc[_i, 'start'] and st <= b.loc[_i, 'stop']:
            # print ('start position of TE is past the start of the bin {} > {}\nBut the end is not {} > {} this is a multibin TE'.format(st, b.loc[_i, 1], en, b.loc[_i, 2]))
            multi_bin = True
        if multi_bin:
            _i_list.append(_i)
            if en <= b.loc[_i, 'stop']:
                # print ('multibin TE ends. TE end position is smaller than end position of current bin {} < {}'.format(en, b.loc[_i, 2]))
                return _i_list
    #print ('No bin found starting in {} and ending {} not found in index for {}'.format(st, en, chr))
    return

def _get_bin(chr, st, en, bins):
    #finds the bin that a TE is in [second function]
    b = bins[bins.loc[:, 'Chr'] == chr]
    _i_list = []
    multi_bin = False
    missing = check_if_missing(chr, st, en)
    if missing:
        for _i in b.index:
            if st >= b.loc[_i, 'start'] and en <= b.loc[_i, 'stop']:
                overlap = en - st
                return [_i, overlap]
            elif st >= b.loc[_i, 'start'] and st <= b.loc[_i, 'stop'] and not multi_bin:
                multi_bin = True
                overlap = []
            if multi_bin:
                _i_list.append(_i)
                if en <= b.loc[_i, 'stop']:
                    overlap.append(en - b.loc[_i, 'start'])
                    return [_i_list, overlap]
                overlap.append(b.loc[_i, 'stop'] - b.loc[_i, 'start'])

    for _i in b.index:
        if st >= b.loc[_i, 'start'] and en <= b.loc[_i, 'stop']:
            overlap = en-st
            return [_i, overlap]
        elif st >= b.loc[_i, 'start'] and st <= b.loc[_i, 'stop'] and not multi_bin:
            multi_bin = True
            overlap = []
        if multi_bin:
            _i_list.append(_i)
            if en <= b.loc[_i, 'stop']:
                overlap.append(en - b.loc[_i, 'start'])
                return [_i_list, overlap]
            overlap.append(b.loc[_i, 'stop'] - b.loc[_i, 'start'])

    #print ('No bin found starting in {} and ending {} not found in index for {} ##'.format(st, en, chr))
    return

def _get_bin_(chr, st, en, bins):
    #finds the bin that a TE is in [third function]
    b = bins[bins.loc[:, 'Chr'] == chr]
    _i_list = []
    multi_bin = False
    missing = check_if_missing(chr, st, en)
    if missing:
        for _i in b.index:
            if st >= b.loc[_i, 'start'] and en <= b.loc[_i, 'stop']:
                overlap = en - st
                # print ('TE fits in bin {} {} is contained in {} {}'.format(st, en,  b.loc[_i, 1],  b.loc[_i, 2]))
                return [_i, overlap]
            elif st >= b.loc[_i, 'start'] and st <= b.loc[_i, 'stop'] and not multi_bin:
                # print ('start position of TE is past the start of the bin {} > {}\nBut the end is not {} > {} this is a multibin TE'.format(st, b.loc[_i, 1], en, b.loc[_i, 2]))
                multi_bin = True
                overlap = []
            if multi_bin:
                _i_list.append(_i)
                if en <= b.loc[_i, 'stop']:
                    # print ('multibin TE ends. TE end position is smaller than end position of current bin {} < {}'.format(en, b.loc[_i, 2]))
                    overlap.append(en - b.loc[_i, 'start'])
                    return [_i_list, overlap]
                overlap.append(b.loc[_i, 'stop'] - b.loc[_i, 'start'])

    for _i in b.index:
        if st >= b.loc[_i, 'start'] and en <= b.loc[_i, 'stop']:
            overlap = en-st
            # print ('TE fits in bin {} {} is contained in {} {}'.format(st, en,  b.loc[_i, 1],  b.loc[_i, 2]))
            return [_i, overlap]
        elif st >= b.loc[_i, 'start'] and st <= b.loc[_i, 'stop'] and not multi_bin:
            # print ('start position of TE is past the start of the bin {} > {}\nBut the end is not {} > {} this is a multibin TE'.format(st, b.loc[_i, 1], en, b.loc[_i, 2]))
            multi_bin = True
            overlap = []
        if multi_bin:
            _i_list.append(_i)
            if en <= b.loc[_i, 'stop']:
                # print ('multibin TE ends. TE end position is smaller than end position of current bin {} < {}'.format(en, b.loc[_i, 2]))
                overlap.append(en - b.loc[_i, 'start'])
                return [_i_list, overlap]
            overlap.append(b.loc[_i, 'stop'] - b.loc[_i, 'start'])

    #print ('No bin found starting in {} and ending {} not found in index for {} ##'.format(st, en, chr))
    return

def __get_bin(chr, st, en, bins):
    #finds the bin that a TE is in [fourth function]
    b = bins[bins.loc[:, 'Chr'] == chr]
    _i_list = []
    multi_bin = False
    missing = check_if_missing(chr, st, en)
    if missing:
        for x in range(len(b.index)):
            _i = b.index[x]
            if st >= b.loc[_i, 'start'] and en <= b.loc[_i, 'stop']:
                overlap = en - st
                # print ('TE fits in bin {} {} is contained in {} {}'.format(st, en,  b.loc[_i, 1],  b.loc[_i, 2]))
                if x > 3:
                    if x < len(b.index)-3:
                        return [[b.index[_x] for _x in range(x-5, x+5)], overlap]
                    else:
                        return [[b.index[_x] for _x in range(x-5, len(b.index))], overlap]
                else:
                    return [[b.index[_x] for _x in range(0, x+5)], overlap]
                #return [_i, overlap]
            elif st >= b.loc[_i, 'start'] and st <= b.loc[_i, 'stop'] and not multi_bin:
                # print ('start position of TE is past the start of the bin {} > {}\nBut the end is not {} > {} this is a multibin TE'.format(st, b.loc[_i, 1], en, b.loc[_i, 2]))
                multi_bin = True
                overlap = []
                if x<5:
                    start_x = 0
                else:
                    start_x = x
            if multi_bin:
                _i_list.append(_i)
                if en <= b.loc[_i, 'stop']:
                    # print ('multibin TE ends. TE end position is smaller than end position of current bin {} < {}'.format(en, b.loc[_i, 2]))
                    overlap.append(en - b.loc[_i, 'start'])
                    if x < len(b.index)-5:
                        return [[b.index[_x] for _x in range(start_x, x+5)], overlap]
                    else:
                        return [[b.index[_x] for _x in range(start_x, len(b.index))], overlap]
                    #return [_i_list, overlap]
                overlap.append(b.loc[_i, 'stop'] - b.loc[_i, 'start'])

    for x in range(len(b.index)):
        _i = b.index[x]
        if st >= b.loc[_i, 'start'] and en <= b.loc[_i, 'stop']:
            overlap = en-st
            # print ('TE fits in bin {} {} is contained in {} {}'.format(st, en,  b.loc[_i, 1],  b.loc[_i, 2]))
            if x > 5:
                if x < len(b.index) - 5:
                    return [[b.index[_x] for _x in range(x - 5, x + 5)], overlap]
                else:
                    return [[b.index[_x] for _x in range(x - 5, len(b.index))], overlap]
            else:
                return [[b.index[_x] for _x in range(0, x + 5)], overlap]
            #return [_i, overlap]
        elif st >= b.loc[_i, 'start'] and st <= b.loc[_i, 'stop'] and not multi_bin:
            # print ('start position of TE is past the start of the bin {} > {}\nBut the end is not {} > {} this is a multibin TE'.format(st, b.loc[_i, 1], en, b.loc[_i, 2]))
            multi_bin = True
            overlap = []
            if x < 3:
                start_x = 0
            else:
                start_x = x
        if multi_bin:
            _i_list.append(_i)
            if en <= b.loc[_i, 'stop']:
                # print ('multibin TE ends. TE end position is smaller than end position of current bin {} < {}'.format(en, b.loc[_i, 2]))
                overlap.append(en - b.loc[_i, 'start'])
                if x < len(b.index) - 5:
                    return [[b.index[_x] for _x in range(start_x, x + 5)], overlap]
                else:
                    return [[b.index[_x] for _x in range(start_x, len(b.index))], overlap]
                #return [_i_list, overlap]
            overlap.append(b.loc[_i, 'stop'] - b.loc[_i, 'start'])

    #print ('No bin found starting in {} and ending {} not found in index for {} ##'.format(st, en, chr))
    return

def make_duplication_edgelist(all_duplications, complete_metadata, _IDbin):
    #makes the duplication edgelist
    dup_file_txt = 'bin_1\tbin_2\tedge_type\tactivity\n'
    for dup1 in all_duplications.keys():
        for dup2 in all_duplications[dup1]:
            Dup1 = get_bin(dup1.split('_')[0], int(dup1.split('_')[1]), int(dup1.split('_')[2]))[0]
            Dup2 = get_bin(dup2.split('_')[0], int(dup2.split('_')[1]), int(dup2.split('_')[2]))[0]
            if type(Dup1) == type('str'):
                _Dup1 = _IDbin.loc[Dup1, 'chr_start_stop']
                if type(Dup2) == type('str'):
                    _Dup2 = _IDbin.loc[Dup2, 'chr_start_stop']
                    dup_file_txt += '{}\t{}\t{}_{}\t{}_{}\n'.format(_Dup1, _Dup2,
                                                                    complete_metadata.loc[Dup1, 'bintype'],
                                                                    complete_metadata.loc[Dup2, 'bintype'],
                                                                    complete_metadata.loc[Dup1, 'interacting'],
                                                                    complete_metadata.loc[Dup2, 'interacting'])
                elif type(Dup2) == type(['list']):
                    for _dup2 in Dup2:
                        _Dup2 = _IDbin.loc[_dup2, 'chr_start_stop']
                        dup_file_txt += '{}\t{}\t{}_{}\t{}_{}\n'.format(_Dup1, _Dup2,
                                                                        complete_metadata.loc[Dup1, 'bintype'],
                                                                        complete_metadata.loc[_dup2, 'bintype'],
                                                                        complete_metadata.loc[Dup1, 'interacting'],
                                                                        complete_metadata.loc[_dup2, 'interacting'])
            elif type(Dup2) == type('str'):
                _Dup2 = _IDbin.loc[Dup2, 'chr_start_stop']
                for _dup1 in Dup1:
                    _Dup1 = _IDbin.loc[_dup1, 'chr_start_stop']
                    dup_file_txt += '{}\t{}\t{}_{}\t{}_{}\n'.format(_Dup1, _Dup2,
                                                                    complete_metadata.loc[_dup1, 'bintype'],
                                                                    complete_metadata.loc[Dup2, 'bintype'],
                                                                    complete_metadata.loc[_dup1, 'interacting'],
                                                                    complete_metadata.loc[Dup2, 'interacting'])
            elif type(Dup2) == type(['list']):
                for _dup1 in Dup1:
                    for _dup2 in Dup2:
                        _Dup1 = _IDbin.loc[_dup1, 'chr_start_stop']
                        _Dup2 = _IDbin.loc[_dup2, 'chr_start_stop']
                        dup_file_txt += '{}\t{}\t{}_{}\t{}_{}\n'.format(_Dup1, _Dup2,
                                                                        complete_metadata.loc[_dup1, 'bintype'],
                                                                        complete_metadata.loc[_dup2, 'bintype'],
                                                                        complete_metadata.loc[_dup1, 'interacting'],
                                                                        complete_metadata.loc[_dup2, 'interacting'])

    outfile = open('duplications_edgelist.tsv', 'w')
    outfile.write(dup_file_txt)
    outfile.close()

def make_bed_genome(unmasked_genome_dir, genome):
    #makes a bed file from a fasta file
    if not os.path.isfile('{}{}.bed'.format(unmasked_genome_dir, genome.split('.')[0])):
        fasta_file = open('{}{}'.format(unmasked_genome_dir, genome)).read()
        fasta = fasta_file.split('>')[1:]
        bed = ''
        old_chr_size = 0
        for chr in fasta:
            header = chr.split('\n')[0]
            sequence = ''.join(chr.split('\n')[1:])
            new_chr_len = len(sequence)
            bed += '{}\t{}\t{}\n'.format(header, old_chr_size, old_chr_size+new_chr_len)
            old_chr_size = old_chr_size+new_chr_len
        outfile = open('{}{}.bed'.format(unmasked_genome_dir, genome.split('.')[0]), 'w')
        outfile.write(bed)
        outfile.close()

    return pd.read_csv('{}{}.bed'.format(unmasked_genome_dir, genome.split('.')[0]), sep='\t', index_col=0, header=None)

def remove_UnplacedScaffolds(bins):
    #removes bins that are not on a chromosome
    valid_index = []
    for i in bins.index:
        if i.startswith('Chr'):
            valid_index.append(True)
        else:
            valid_index.append(False)
    return bins.loc[bins.index[valid_index]]

def get_centromere_interactions(obs_exp_df, bin_annot):
    #gets the interactions between centromeres
    centromere_bins = bin_annot[bin_annot.loc[:, 'bintype'] == 'centromere']
    centromere_counts = []
    for chr1 in list(set(centromere_bins.loc[:, 'chr'])):
        chr1_bins = centromere_bins[centromere_bins.loc[:, 'chr'] == chr1]
        for chr2 in [x for x in list(set(centromere_bins.loc[:, 'chr'])) if x != chr1]:
            chr2_bins = centromere_bins[centromere_bins.loc[:, 'chr'] == chr2]
            centromere_counts.extend(obs_exp_df.loc[chr1_bins.index, chr2_bins.index].values.flatten())
    centromere_counts = np.array(centromere_counts)
    #nonzero = centromere_counts>0.01
    #print (np.mean(centromere_counts[nonzero]))
    #sns.histplot(centromere_counts[nonzero])
    #plt.axvline(x = np.mean(centromere_counts[nonzero]), color = 'b')
    #plt.show()
    return centromere_counts

def filter_df(filt_df, cis_LR_dict, out_tsv):
    #filters the dataframe to remove cis interactions that are not in the cis_LR_dict
    I = filt_df.to_numpy().nonzero()
    pos2drop = []
    for i in range(len(I[0])):
        bin1 = filt_df.index[I[0][i]]
        bin2 = filt_df.index[I[1][i]]
        #print (bin1, bin2)
        if bin1.split('_')[0] == bin2.split('_')[0]:
            if bin2 not in cis_LR_dict[bin1] and bin1 not in cis_LR_dict[bin2]:
                pos2drop.append((I[0][i], I[1][i]))
    for duo in pos2drop:
        filt_df.iloc[duo[0], duo[1]] = 0
        filt_df.iloc[duo[1], duo[0]] = 0
    I = filt_df.index
    nonzero = filt_df.to_numpy().nonzero()
    out_txt = 'level_0\tlevel_1\t0\n'
    for i in range(len(nonzero[0])):
        out_txt += '{}\t{}\t{}\n'.format(I[nonzero[0][i]], I[nonzero[1][i]], filt_df.loc[I[nonzero[0][i]], I[nonzero[1][i]]])
    out_file = open('{}{}_filtered_df.csv'.format(tsv_dir, sample_name), 'w')
    out_file.write(out_txt)
    out_file.close()
    unstacked_filt_df = pd.read_csv('{}{}_filtered_df.csv'.format(tsv_dir, sample_name), sep='\t')
    return unstacked_filt_df

def parse_deltafile(deltafile, out_dir, sample_name):
    #parses the deltafile to get the duplications
    #print ('{}{}_duplications_100.csv'.format(out_dir, sample_name))
    if not os.path.isfile('{}{}_duplications_100.csv'.format(out_dir, sample_name)):
        deltafile = deltafile.split('=====================================================================================\n')[1]
        deltafile = deltafile.split('\n')[:-1]
        matches = pd.DataFrame(0, columns=['ref_start', 'ref_stop', 'ref_chr', 'que_start', 'que_stop', 'que_chr', 'LEN', 'ID'], index=[])
        counter = 0
        for align in deltafile:
            #print (counter, len(deltafile))
            ID = align.split('|')[3].replace(' ', '')
            #if ID != '100.00':
            if float(ID) >= 85.00:
                lens = [x for x in align.split('|')[2].split(' ') if x]
                LEN = (int(lens[0]) + int(lens[1]))/2
                if LEN > 1000:
                    #ref = align.split('|')[-1].replace(' ', '').split('\t')[0]
                    que = align.split('|')[-1].replace(' ', '').split('\t')[0]
                    ref = align.split('|')[-1].replace(' ', '').split('\t')[1]

                    que_start = [x for x in align.split('|')[0].split(' ') if x][0]
                    que_stop = [x for x in align.split('|')[0].split(' ') if x][1]

                    ref_start = [x for x in align.split('|')[1].split(' ') if x][0]
                    ref_stop = [x for x in align.split('|')[1].split(' ') if x][1]

                    if que != 'UnplacedScaffolds' and ref != 'UnplacedScaffolds':
                        if que != ref:
                            matches.loc[counter, ['ref_start', 'ref_stop', 'ref_chr', 'que_start', 'que_stop', 'que_chr', 'LEN', 'ID']] = [ref_start, ref_stop, ref, que_start, que_stop, que, LEN, ID]
                            counter+=1
                        elif abs(int(ref_start)-int(que_start)) > 30000 and abs(int(ref_stop)-int(que_stop)) > 30000:
                            matches.loc[counter, ['ref_start', 'ref_stop', 'ref_chr', 'que_start', 'que_stop', 'que_chr', 'LEN', 'ID']] = [ref_start, ref_stop, ref, que_start, que_stop, que, LEN, ID]
                            counter += 1
        matches.to_csv('{}{}_duplications_100.csv'.format(out_dir, sample_name),sep='\t')
        matches = pd.read_csv('{}{}_duplications_100.csv'.format(out_dir, sample_name), sep='\t')
    else:
        matches = pd.read_csv('{}{}_duplications_100.csv'.format(out_dir, sample_name), sep='\t')
    return matches

def _parse_deltafile(deltafile, out_dir, sample_name):
    #parses the deltafile to get the duplications [second version]
    #do not remove 100 id dups
    print ('{}{}_duplications_100.csv'.format(out_dir, sample_name))
    if not os.path.isfile('{}{}_duplications_100.csv'.format(out_dir, sample_name)):
        deltafile = deltafile.split('=====================================================================================\n')[1]
        deltafile = deltafile.split('\n')[:-1]
        matches = pd.DataFrame(0, columns=['ref_start', 'ref_stop', 'ref_chr', 'que_start', 'que_stop', 'que_chr', 'LEN', 'ID'], index=[])
        counter = 0
        for align in deltafile:
            #print (counter, len(deltafile))
            ID = align.split('|')[3].replace(' ', '')
            #if ID != '100.00':
            if float(ID) >= 85.00:
                lens = [x for x in align.split('|')[2].split(' ') if x]
                LEN = (int(lens[0]) + int(lens[1]))/2
                if LEN > 1000:
                    #ref = align.split('|')[-1].replace(' ', '').split('\t')[0]
                    que = align.split('|')[-1].replace(' ', '').split('\t')[0]
                    ref = align.split('|')[-1].replace(' ', '').split('\t')[1]

                    que_start = [x for x in align.split('|')[0].split(' ') if x][0]
                    que_stop = [x for x in align.split('|')[0].split(' ') if x][1]

                    ref_start = [x for x in align.split('|')[1].split(' ') if x][0]
                    ref_stop = [x for x in align.split('|')[1].split(' ') if x][1]

                    if que != 'UnplacedScaffolds' and ref != 'UnplacedScaffolds':
                        if que != ref:
                            matches.loc[counter, ['ref_start', 'ref_stop', 'ref_chr', 'que_start', 'que_stop', 'que_chr', 'LEN', 'ID']] = [ref_start, ref_stop, ref, que_start, que_stop, que, LEN, ID]
                            counter+=1
                        elif abs(int(ref_start)-int(que_start)) > 30000 and abs(int(ref_stop)-int(que_stop)) > 30000:
                            matches.loc[counter, ['ref_start', 'ref_stop', 'ref_chr', 'que_start', 'que_stop', 'que_chr', 'LEN', 'ID']] = [ref_start, ref_stop, ref, que_start, que_stop, que, LEN, ID]
                            counter += 1
        matches.to_csv('{}{}_duplications_100.csv'.format(out_dir, sample_name),sep='\t')
        matches = pd.read_csv('{}{}_duplications_100.csv'.format(out_dir, sample_name), sep='\t')
    else:
        matches = pd.read_csv('{}{}_duplications_100.csv'.format(out_dir, sample_name), sep='\t')
    return matches

def genome2genome(masked_genome_dir, sample_name, out_dir):
    #aligns the genome to itself to find duplications
    genome_path = '{}{}_HiC-improved.final.fasta.masked'.format(masked_genome_dir, sample_name)
    #print(genome_path)
    if not os.path.isfile('{}self_align_{}.delta'.format(out_dir, sample_name)):
        self_map_cmd = 'nucmer --maxmatch --nosimplify --prefix {}self_align_{} {} {}'.format(out_dir, sample_name, genome_path, genome_path)
        subprocess.check_call(self_map_cmd, shell=True)

    if not os.path.isfile('{}self_align_{}.coords'.format(out_dir, sample_name)):
        coord_cmd = 'show-coords -r {}self_align_{}.delta > {}self_align_{}.coords'.format(out_dir, sample_name, out_dir, sample_name)
        subprocess.check_call(coord_cmd, shell=True)

    deltafile = open('{}self_align_{}.coords'.format(out_dir, sample_name)).read()
    duplications = parse_deltafile(deltafile, out_dir, sample_name)
    return duplications

def _genome2genome(masked_genome_dir, sample_name, out_dir):
    #aligns the genome to itself to find duplications [second version]
    genome_path = 'work/genomes/JR2/JR2_masked/Verticillium_dahliae_JR2.fasta.masked'
    #print(genome_path)
    if not os.path.isfile('{}self_align_{}.delta'.format(out_dir, sample_name)):
        self_map_cmd = 'nucmer --maxmatch --nosimplify --prefix {}self_align_{} {} {}'.format(out_dir, sample_name, genome_path, genome_path)
        subprocess.check_call(self_map_cmd, shell=True)

    if not os.path.isfile('{}self_align_{}.coords'.format(out_dir, sample_name)):
        coord_cmd = 'show-coords -r {}self_align_{}.delta > {}self_align_{}.coords'.format(out_dir, sample_name, out_dir, sample_name)
        subprocess.check_call(coord_cmd, shell=True)

    deltafile = open('{}self_align_{}.coords'.format(out_dir, sample_name)).read()
    duplications = _parse_deltafile(deltafile, out_dir, sample_name)
    return duplications

def find_dup_bins(duplications, bins, cis_LR_dict):
    #finds the bins that are duplicated
    missing_duplication = 0;    found_duplication = 0;  unexplained_missing_duplication = 0
    dup_dict = {}
    dup_len=[]
    for x in duplications.index:
        dup = duplications.loc[x]
        ref_chr = dup.loc['ref_chr']
        ref_start = dup.loc['ref_start']
        ref_stop = dup.loc['ref_stop']
        que_chr = dup.loc['que_chr']
        que_start = dup.loc['que_start']
        que_stop = dup.loc['que_stop']
        #print (ref_chr, ref_start, ref_stop)
        ref_bins = _get_bin(ref_chr, ref_start, ref_stop, bins)
        que_bins = _get_bin(que_chr, que_start, que_stop, bins)
        dup_len.append(abs(ref_start - ref_stop))
        if ref_bins and que_bins:
            found_duplication +=1
            if type(ref_bins[0]) == type(['list']) and type(que_bins[0]) == type(['list']):
                for ref_bin in ref_bins[0]:
                    for que_bin in que_bins[0]:
                        if ref_bin in dup_dict.keys():
                            if que_bin in cis_LR_dict[ref_bin]:
                                dup_dict[ref_bin].append(que_bin)
                            elif ref_bin.split('_')[0] != que_bin.split('_')[0]:
                                dup_dict[ref_bin].append(que_bin)
                        else:
                            if que_bin in cis_LR_dict[ref_bin]:
                                dup_dict[ref_bin] = [que_bin]
                            elif ref_bin.split('_')[0] != que_bin.split('_')[0]:
                                dup_dict[ref_bin] = [que_bin]

            elif type(ref_bins[0]) == type(['list']) and type(que_bins[0]) == type('str'):
                for ref_bin in ref_bins[0]:
                    if ref_bin in dup_dict.keys():
                        if que_bins[0] in cis_LR_dict[ref_bin]:
                            dup_dict[ref_bin].append(que_bins[0])
                        elif ref_bin.split('_')[0] != que_bins[0].split('_')[0]:
                            dup_dict[ref_bin].append(que_bins[0])
                    else:
                        if que_bins[0] in cis_LR_dict[ref_bin]:
                            dup_dict[ref_bin] = [que_bins[0]]
                        elif ref_bin.split('_')[0] != que_bins[0].split('_')[0]:
                            dup_dict[ref_bin] = [que_bins[0]]

            elif type(ref_bins[0]) == type('str') and type(que_bins[0]) == type(['list']):
                for que_bin in que_bins[0]:
                    if ref_bins[0] in dup_dict.keys():
                        if que_bin in cis_LR_dict[ref_bins[0]]:
                            dup_dict[ref_bins[0]].append(que_bin)
                        elif que_bin.split('_')[0] != ref_bins[0].split('_')[0]:
                            dup_dict[ref_bins[0]].append(que_bin)
                    else:
                        if que_bin in cis_LR_dict[ref_bins[0]]:
                            dup_dict[ref_bins[0]] = [que_bin]
                        elif que_bin.split('_')[0] != ref_bins[0].split('_')[0]:
                            dup_dict[ref_bins[0]] = [que_bin]

            elif type(ref_bins[0]) == type('str') and type(que_bins[0]) == type('str'):
                if ref_bins[0] in dup_dict.keys():
                    if que_bins[0] in cis_LR_dict[ref_bins[0]]:
                        dup_dict[ref_bins[0]].append(que_bins[0])
                    elif que_bins[0].split('_')[0] != ref_bins[0].split('_')[0]:
                        dup_dict[ref_bins[0]].append(que_bins[0])
                else:
                    if que_bins[0] in cis_LR_dict[ref_bins[0]]:
                        dup_dict[ref_bins[0]] = [que_bins[0]]
                    elif que_bins[0].split('_')[0] != ref_bins[0].split('_')[0]:
                        dup_dict[ref_bins[0]] = [que_bins[0]]

        missing = False
        if not que_bins:
            missing = check_if_missing(que_chr, que_start, que_stop, missing)
        if not ref_bins:
            missing = check_if_missing(ref_chr, ref_start, ref_stop, missing)
        if missing:
            missing_duplication +=1
        else:
            unexplained_missing_duplication += 1
    if commandLineArgs['verbose']:
        print ('recap')
        print ('out of {} duplications {} were found and {} were not'.format(len(duplications), found_duplication, missing_duplication))
        print (len(duplications), found_duplication, missing_duplication)
        print (np.sum([len(x) for x in dup_dict.values()]))
    return dup_dict

def check_if_missing(chr, start, stop, missing=False):
    #checks if the bin is missing
    B = missing_bins[missing_bins.loc[:, 'chr'] == chr]
    for missing_interval in B.index:
        if start > B.loc[missing_interval, 'start'] and start < B.loc[missing_interval, 'stop']:
            if stop > B.loc[missing_interval, 'start'] and stop < B.loc[missing_interval, 'stop']:
                missing = True
            else:
                missing = True
        elif stop > B.loc[missing_interval, 'start'] and stop < B.loc[missing_interval, 'stop']:
            missing = True
        if start < B.loc[missing_interval, 'start'] and stop > B.loc[missing_interval, 'start']:
            missing = True
    return missing

def annotate_AGRl(bin_annot, synteny_file1, synteny_file2, bins):
    #annotates the bins with the synteny file
    binsize = pd.DataFrame(0, index=bins.index, columns=['size', 'coverage'])
    for bin in bins.index:
        binsize.loc[bin, ['size', 'coverage']] = [bins.loc[bin, 'stop'] - bins.loc[bin, 'start'], 0]

    try:
        agr_df = pd.read_csv(synteny_file1, sep='\t', header=None)
        for i in agr_df.index:
            if agr_df.loc[i, 0] != 'UnplacedScaffolds':
                _b = _get_bin(agr_df.loc[i, 0], agr_df.loc[i, 1], agr_df.loc[i, 2], bins)
                if _b:
                    if type(_b) == type('str'):
                        binsize.loc[_b[0], 'coverage'] = binsize.loc[_b[0], 'coverage'] + _b[1]
                    if type(_b[0]) == type(['list']):
                        BINS = _b[0]
                        COV = _b[1]
                        for x in range(len(_b[0])):
                            binsize.loc[BINS[x], 'coverage'] = binsize.loc[BINS[x], 'coverage'] + COV[x]
    except TypeError:
        agr_df = pd.read_csv(synteny_file1, sep='\t')
        for i in agr_df.index:
            if agr_df.loc[i, 'Chr'] != 'UnplacedScaffolds':
                _b = _get_bin(agr_df.loc[i, 'Chr'], agr_df.loc[i, 'start'], agr_df.loc[i, 'stop'], bins)
                if _b:
                    if type(_b[0]) == type('str'):
                        binsize.loc[_b[0], 'coverage'] = binsize.loc[_b[0], 'coverage'] + _b[1]
                    if type(_b[0]) == type(['list']):
                        BINS = _b[0]
                        COV = _b[1]
                        for x in range(len(_b[0])):
                            binsize.loc[BINS[x], 'coverage'] = binsize.loc[BINS[x], 'coverage'] + COV[x]

    try:
        agr_df = pd.read_csv(synteny_file2, sep='\t', header=None)
        for i in agr_df.index:
            if agr_df.loc[i, 0] != 'UnplacedScaffolds':
                _b = _get_bin(agr_df.loc[i, 0], agr_df.loc[i, 1], agr_df.loc[i, 2], bins)
                if _b:
                    if type(_b) == type('str'):
                        binsize.loc[_b[0], 'coverage'] = binsize.loc[_b[0], 'coverage'] + _b[1]
                    if type(_b[0]) == type(['list']):
                        BINS = _b[0]
                        COV = _b[1]
                        for x in range(len(_b[0])):
                            binsize.loc[BINS[x], 'coverage'] = binsize.loc[BINS[x], 'coverage'] + COV[x]
    except TypeError:
        agr_df = pd.read_csv(synteny_file2, sep='\t')
        for i in agr_df.index:
            if agr_df.loc[i, 'Chr'] != 'UnplacedScaffolds':
                _b = _get_bin(agr_df.loc[i, 'Chr'], agr_df.loc[i, 'start'], agr_df.loc[i, 'stop'], bins)
                if _b:
                    if type(_b[0]) == type('str'):
                        binsize.loc[_b[0], 'coverage'] = binsize.loc[_b[0], 'coverage'] + _b[1]
                    if type(_b[0]) == type(['list']):
                        BINS = _b[0]
                        COV = _b[1]
                        for x in range(len(_b[0])):
                            binsize.loc[BINS[x], 'coverage'] = binsize.loc[BINS[x], 'coverage'] + COV[x]
    for bin in binsize.index:
        if binsize.loc[bin, 'coverage']/binsize.loc[bin, 'size'] > 0.5:
            bin_annot.loc[bin, ['chr', 'start', 'stop', 'bintype']] = [bins.loc[bin, 'Chr'], bins.loc[bin, 'start'], bins.loc[bin, 'stop'], 'AGR_l']

    return bin_annot

def _annotate_AGRl(bin_annot, synteny_file, bins):
    #annotates the bins with the synteny file [for AGRl]
    binsize = pd.DataFrame(0, index=bins.index, columns=['size', 'coverage'])
    for bin in bins.index:
        binsize.loc[bin, ['size', 'coverage']] = [bins.loc[bin, 'stop'] - bins.loc[bin, 'start'], 0]

    agr_df = pd.read_csv(synteny_file, sep='\t', header=None)

    for i in agr_df.index:
        if agr_df.loc[i, 0] != 'UnplacedScaffolds':
            _b = _get_bin(agr_df.loc[i, 0], agr_df.loc[i, 1]-1000, agr_df.loc[i, 2]+1000, bins)
            if _b:
                if type(_b) == type('str'):
                    binsize.loc[_b[0], 'coverage'] = binsize.loc[_b[0], 'coverage'] + _b[1]
                if type(_b[0]) == type(['list']):
                    BINS = _b[0]
                    COV = _b[1]
                    for x in range(len(_b[0])):
                        binsize.loc[BINS[x], 'coverage'] = binsize.loc[BINS[x], 'coverage'] + int(COV[x])

    for bin in binsize.index:
        if binsize.loc[bin, 'coverage']/binsize.loc[bin, 'size'] > 0.2:
            bin_annot.loc[bin, ['chr', 'start', 'stop', 'bintype']] = [bins.loc[bin, 'Chr'], bins.loc[bin, 'start'], bins.loc[bin, 'stop'], 'AGR_l']

    return bin_annot

def make_histogram(filt_LR_df, bin_annot, plot_dir, sample_name, counts_txt):
    #makes a histogram of the number of edges between each bin type
    core_count = len(bin_annot[bin_annot.loc[:, 'bintype'] == 'core'])
    centromere_count = len(bin_annot[bin_annot.loc[:, 'bintype'] == 'centromere'])
    agr_count = len(bin_annot[bin_annot.loc[:, 'bintype'] == 'AGR_l'])

    hue_order = ['core_core', 'centromere_core','AGR_l_core','centromere_centromere','AGR_l_centromere','AGR_l_AGR_l']

    for I in filt_LR_df.index:
        i1 = filt_LR_df.loc[I, 'level_0']
        i2 = filt_LR_df.loc[I, 'level_1']
        annot1 = bin_annot.loc[i1, 'bintype']
        annot2 = bin_annot.loc[i2, 'bintype']
        filt_LR_df.loc[I, 'edge_type'] = '_'.join(sorted([annot1, annot2]))

    filt_LR_df.to_csv('{}{}annotated_filtered_df.csv'.format(tsv_dir, sample_name), sep='\t')
    counts_txt += '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(sample_name, core_count, centromere_count, agr_count, len(filt_LR_df[filt_LR_df.loc[:, 'edge_type'] == 'core_core']), len(filt_LR_df[filt_LR_df.loc[:, 'edge_type'] == 'centromere_core']),len(filt_LR_df[filt_LR_df.loc[:, 'edge_type'] == 'AGR_l_core']),len(filt_LR_df[filt_LR_df.loc[:, 'edge_type'] == 'centromere_centromere']),len(filt_LR_df[filt_LR_df.loc[:, 'edge_type'] == 'AGR_l_centromere']),len(filt_LR_df[filt_LR_df.loc[:, 'edge_type'] == 'AGR_l_AGR_l']))

    perc99 = filt_LR_df.iloc[[int(len(filt_LR_df)*0.01)], 2]

    _filt_LR_df = filt_LR_df.iloc[int(len(filt_LR_df)*0.01):]

    ax = sns.histplot(data=filt_LR_df, x='0', hue='edge_type', common_norm=False, element="poly", hue_order=hue_order)
    ax.set_xlim(left=min(filt_LR_df.loc[:, '0'].values)-1, right=perc99.values)
    plt.savefig('{}histogram_{}.png'.format(plot_dir, sample_name))
    plt.close()

    ax = sns.violinplot(x="edge_type", y="0", data=filt_LR_df, order=hue_order)
    ax.tick_params(axis='x', labelrotation=12)
    plt.savefig('{}violin_{}.png'.format(plot_dir, sample_name))
    plt.close()

    return counts_txt, filt_LR_df

def get_bin_contacts(filt_LR_df, bin_annot):
    #gets the number of contacts between each bin type
    bins = list(set(filt_LR_df.loc[:, 'level_0'].values).union(set(filt_LR_df.loc[:, 'level_1'].values)))
    bindf = pd.DataFrame(0, index=bins, columns=['bintype', 'to_AGR_l', 'to_centromere', 'to_core', 'all_contacts'])

    level1bins = []
    for bin in list(set(filt_LR_df.loc[:, 'level_0'].values)):
        _df = filt_LR_df[filt_LR_df.loc[:, 'level_0'] == bin]
        for i in _df.index:
            bintype = bin_annot.loc[_df.loc[i, 'level_0'], 'bintype']
            bincount = _df.loc[i,'0']
            targettype = 'to_{}'.format(bin_annot.loc[_df.loc[i, 'level_1'], 'bintype'])
            #print (bintype, bincount, targettype)
            bindf.loc[bin, 'all_contacts'] += bincount
            #+1 or bincount, not sure
            bindf.loc[bin, targettype] += 1
            bindf.loc[bin, 'bintype'] = bintype
        level1bins.append(bin)

    for bin in list(set(filt_LR_df.loc[:, 'level_1'].values)):
        if bin not in level1bins:
            _df = filt_LR_df[filt_LR_df.loc[:, 'level_1'] == bin]
            for i in _df.index:
                bintype = bin_annot.loc[_df.loc[i, 'level_1'], 'bintype']
                bincount = _df.loc[i, '0']
                targettype = 'to_{}'.format(bin_annot.loc[_df.loc[i, 'level_0'], 'bintype'])
                # print (bintype, bincount, targettype)
                bindf.loc[bin, 'all_contacts'] += bincount
                # +1 or bincount, not sure
                bindf.loc[bin, targettype] += 1
                bindf.loc[bin, 'bintype'] = bintype

    #print ('total')
    #print (bindf.loc[:, ['to_AGR_l', 'to_centromere', 'to_core']].sum())
    #print (bindf.loc[:, ['to_AGR_l', 'to_centromere', 'to_core']].sum().sum())

    for bintype in ['AGR_l', 'centromere', 'core']:
        _bindf = bindf[bindf.loc[:,'bintype'] == bintype]
        #print (bintype)
        #print(_bindf.loc[:, ['to_AGR_l', 'to_centromere', 'to_core']].sum())
        #print(_bindf.loc[:, ['to_AGR_l', 'to_centromere', 'to_core']].sum().sum())

    return bindf

def calculate_AGR_enrichment(filt_LR_df, bin_annot):
    #calculates the enrichment of AGR bins in the AGR_l bin type
    bin_contacts = get_bin_contacts(filt_LR_df, bin_annot)
    #fig, axs = plt.subplots(3, sharex=True, sharey=True)
    #print (bin_annot)
    #print (filt_LR_df)
    agr_bins = bin_annot[bin_annot.loc[:, 'bintype'] == 'AGR_l']
    core_bins = bin_annot[bin_annot.loc[:, 'bintype'] == 'core']
    centromere_bins = bin_annot[bin_annot.loc[:, 'bintype'] == 'centromere']

    for edge_type in set(filt_LR_df.loc[:, 'edge_type']):
        print (edge_type, len(filt_LR_df[filt_LR_df.loc[:, 'edge_type'] == edge_type]))

    #print(bin_contacts)
    #print('{} AGR bins, {} core bins, {} centromere bins'.format(len(agr_bins), len(core_bins), len(centromere_bins)))
    #print('{} total contacts above the treshold'.format(len(filt_LR_df)))
    noncentromere_filt_LR_df = filt_LR_df[filt_LR_df.loc[:, 'edge_type'] != 'centromere_centromere']
    noncentromere_filt_LR_df = noncentromere_filt_LR_df[noncentromere_filt_LR_df.loc[:, 'edge_type'] != 'centromere_AGR_l']
    noncentromere_filt_LR_df = noncentromere_filt_LR_df[noncentromere_filt_LR_df.loc[:, 'edge_type'] != 'AGR_l_centromere']
    noncentromere_filt_LR_df = noncentromere_filt_LR_df[noncentromere_filt_LR_df.loc[:, 'edge_type'] != 'core_centromere']
    noncentromere_filt_LR_df = noncentromere_filt_LR_df[noncentromere_filt_LR_df.loc[:, 'edge_type'] != 'centromere_core']
    agr_filt_LR_df = noncentromere_filt_LR_df[noncentromere_filt_LR_df.loc[:, 'edge_type'] != 'core_core']
    core_filt_LR_df = noncentromere_filt_LR_df[noncentromere_filt_LR_df.loc[:, 'edge_type'] != 'AGR_l_AGR_l']
    #print (agr_filt_LR_df)

    print ('{} total contacts above the threshold'.format(len(filt_LR_df)/2))
    print ('{} noncentromere contacts above the threshold'.format(len(noncentromere_filt_LR_df)/2))
    print ('{} of which involve AGR bins {}%'.format(len(agr_filt_LR_df)/2, round((len(agr_filt_LR_df)/len(noncentromere_filt_LR_df))*100, 3)))
    print ('There are {} AGR and {} core bins. {}% AGR and {}% core'.format(len(agr_bins), len(core_bins), round(len(agr_bins)/(len(core_bins) + len(agr_bins)), 3)*100, round(len(core_bins)/(len(core_bins) + len(agr_bins)), 3)*100))
    expected_core = int(len(noncentromere_filt_LR_df)*(len(core_bins)/(len(core_bins) + len(agr_bins))))
    expected_agr = int(len(noncentromere_filt_LR_df)*(len(agr_bins)/(len(core_bins) + len(agr_bins))))
    agr_bin_contacts = bin_contacts[bin_contacts.loc[:, 'bintype']=='AGR_l']
    core_bin_contacts = bin_contacts[bin_contacts.loc[:, 'bintype'] == 'core']
    agr_interactions = agr_bin_contacts.loc[:, ['to_AGR_l', 'to_core']].sum().sum()
    core_interactions = core_bin_contacts.loc[:, ['to_AGR_l', 'to_core']].sum().sum()
    agr_interactions = len(agr_filt_LR_df)/2
    core_interactions = len(core_filt_LR_df)/2
    print ('Expected core interactions: {}\tExpected AGR interactions: {}'.format(expected_core, expected_agr))
    print ('Observed core interactions: {}\tObserved AGR interactions: {}'.format(core_interactions, agr_interactions))
    print ([[agr_interactions, core_interactions], [expected_agr, expected_core]])
    oddsratio, pval=fisher_exact([[agr_interactions, core_interactions], [expected_agr, expected_core]], alternative='greater')
    print (oddsratio, pval)
    #res = barnard_exact([[agr_interactions, core_interactions], [expected_agr, expected_core]], alternative='greater')
    #print (res.pvalue)
    print ('\n\n')
    chi_table=[]
    for x in range(3):
        bintype = ['AGR_l', 'core', 'centromere'][x]
        _bin_contacts = bin_contacts[bin_contacts.loc[:, 'bintype'] == bintype]
        #axs[x].violinplot(_bin_contacts.loc[:, ['to_AGR_l','to_centromere','to_core']])
        #axs[x].set_title(bintype)
        #axs[x].set_xticks([0, 1, 2])
        #axs[x].set_xticklabels(['to_AGR_l','to_centromere','to_core'])
        #print ("####\t{}\t####".format(bintype))
        #print ('{} LR interacting bins are {}.\nThese bins form {} interactions with with AGR, {} interactions with core and {} interactions with centromeres bins respectively.'.format(len(_bin_contacts), bintype, _bin_contacts.loc[:, 'to_AGR_l'].sum(), _bin_contacts.loc[:, 'to_core'].sum(), _bin_contacts.loc[:, 'to_centromere'].sum()))
        #print (list(_bin_contacts.loc[:, 'to_core'].values))
        #print (_bin_contacts.loc[:, ['to_AGR_l','to_core','to_centromere']].sum().sum())
        #print (_bin_contacts.loc[:, ['to_AGR_l','to_core','to_centromere']].sum())
        chi_table.append([_bin_contacts.loc[:, 'to_AGR_l'].sum(), _bin_contacts.loc[:, 'to_core'].sum(), _bin_contacts.loc[:, 'to_centromere'].sum()])
        bintype_bins = len(bin_annot[bin_annot.loc[:, 'bintype'] == bintype])
        bintype_noninteracting = bintype_bins - len(_bin_contacts)
        nonbintype_bins = len(bin_annot) - bintype_bins
        nonbintype_interacting = len(bin_contacts) - len(_bin_contacts)
        nonbintype_noninteracting = nonbintype_bins - nonbintype_interacting
        oddsratio, pvalue = fisher_exact([[len(_bin_contacts), bintype_noninteracting], [nonbintype_interacting , nonbintype_noninteracting]])
        #print ('[{} interacting bins, {} non interacting bins], [non{} interacing bins, non{} non interacting bins]'.format(bintype, bintype, bintype, bintype))
        #print ([[len(_bin_contacts), bintype_noninteracting], [nonbintype_interacting , nonbintype_noninteracting]])
        #print (oddsratio, pvalue)


    #print (fisher_exact([[695, 370], [2196, 2525]]))
    #stat, p, dof, expected = chi2_contingency(chi_table)
    #print (p, stat, dof, expected)
    #print (chi_table)
    #fisher_table = [[ , ], [, ]]
    #plt.show()
    #plt.close()

def overlap_duplications(filt_LR_df, dup_dict):
    filt_LR_df.reset_index(inplace=True)
    filt_LR_df.set_index(['level_0', 'level_1'], inplace=True)
    LR_dup = []
    LR_nodup = []
    LR_dup_type = []

    for ref_dup in dup_dict.keys():
        que_dups = dup_dict[ref_dup]
        for que_dup in que_dups:
            DUP = False
            if (ref_dup, que_dup) in filt_LR_df.index:
                LR_dup.append(filt_LR_df.loc[(ref_dup, que_dup), '0'])
                LR_dup_type.append(filt_LR_df.loc[(ref_dup, que_dup), 'edge_type'])
                DUP = True
        if DUP == False:
            if (ref_dup, que_dup) in filt_LR_df.index:
                LR_nodup.append(filt_LR_df.loc[(ref_dup, que_dup), '0'])
    print (LR_dup_type)
    return LR_dup

def check_bins(bins):
    #check for missing bins
    missing_bins_df = pd.DataFrame(0, index=[],columns=['chr', 'start', 'stop'])
    for x in range(len(bins)-1):
        endpos = bins.iloc[x, 2]
        nextpos = bins.iloc[x+1, 1]
        if endpos != nextpos and nextpos-endpos >0:
            missing_bins_df.loc[x , ['chr', 'start', 'stop']] = [bins.iloc[x, 0], endpos, nextpos]
    return missing_bins_df

def add_epigenetics():
    PDB_3K27AC1 = pd.read_csv('{}JR2_H3K27AC1.bed'.format(commandLineArgs['epigeneticDir']), sep='\t', index_col=3, header=None)
    PDB_3K27AC1.columns = ['Chrom', 'start', 'end', 'H3K27AC1']

    PDB_H3K27M3 = pd.read_csv('{}JR2_H3K27M3.bed'.format(commandLineArgs['epigeneticDir']), sep='\t', index_col=3, header=None)
    PDB_H3K27M3.columns = ['Chrom', 'start', 'end', 'H3K27M3']

    PDB_H3K4M2 = pd.read_csv('{}JR2_H3K4M2.bed'.format(commandLineArgs['epigeneticDir']), sep='\t', index_col=3, header=None)
    PDB_H3K4M2.columns = ['Chrom', 'start', 'end', 'H3K4M2']

    PDB_H3K9M3 = pd.read_csv('{}JR2_H3K9M3.bed'.format(commandLineArgs['epigeneticDir']), sep='\t', index_col=3, header=None)
    PDB_H3K9M3.columns = ['Chrom', 'start', 'end', 'H3K9M3']

    PDB_ATAC = pd.read_csv('{}JR2_PDB_ATAC.bed'.format(commandLineArgs['epigeneticDir']), sep='\t', index_col=3, header=None)
    PDB_ATAC.columns = ['Chrom', 'start', 'end', 'ATAC']

    meta = pd.concat([PDB_3K27AC1.loc[:, 'H3K27AC1'], PDB_H3K27M3.loc[:, 'H3K27M3'], PDB_H3K4M2.loc[:, 'H3K4M2'], PDB_H3K9M3.loc[:, 'H3K9M3'], PDB_ATAC.loc[:, 'ATAC']], axis=1)
    meta = meta.iloc[:-5]
    meta.to_csv('{}JR2_metadata.csv'.format(outputDir), sep='\t')
    return meta

def add_TE(bins, teFile):
    if not os.path.isfile('{}{}_TE.csv'.format(outputDir, referenceSampleName)):
        TE = pd.read_csv(teFile, sep='\t', header=None)
        TE.columns = ['Chrom', 'start', 'end', 'strand', 'full_id', 'type', 'class']
        te_class = list(set(TE.loc[:, 'class']))
        te_df = pd.DataFrame(0, index=bins.index, columns=['TE'])
        te_df_full = pd.DataFrame(0, index=bins.index, columns=te_class)

        for i in TE.index:
            bin = _get_bin(TE.loc[i, 'Chrom'], TE.loc[i, 'start'], TE.loc[i, 'end'], bins)
            if bin:
                _class = TE.loc[i, 'class']
                if type(bin[0]) == type('str'):
                    te_df.loc[bin[0], 'TE'] += 1
                    te_df_full.loc[bin[0], _class] += 1
                elif type(bin[0]) == type(['list']):
                    for b in bin[0]:
                        te_df.loc[b, 'TE'] += 1
                        te_df_full.loc[b, _class] += 1
        te_df.to_csv('{}{}_TE.csv'.format(outputDir, referenceSampleName), sep='\t')
        te_df_full.to_csv('{}{}_full_TE.csv'.format(outputDir, referenceSampleName), sep='\t')

    te_df = pd.read_csv('{}{}_TE.csv'.format(outputDir, referenceSampleName), sep='\t', index_col=0)
    te_df_full = pd.read_csv('{}{}_full_TE.csv'.format(outputDir, referenceSampleName), sep='\t', index_col=0)
    return te_df, te_df_full

def add_genes(bins, geneAnnotation):
    if not os.path.isfile('{}{}_genes.csv'.format(outputDir, referenceSampleName)):
        genes_df = pd.DataFrame(0, index=bins.index, columns=['gene_density'])
        genes = pd.read_csv(geneAnnotation, sep=',', header=0)
        categories = list(set(genes.loc[:, 'COG']))
        genes_complete_df = pd.DataFrame(0, index=bins.index, columns=categories)
        for i in genes.index:
            bin = _get_bin(genes.loc[i, 'Chr'], genes.loc[i, 'start'], genes.loc[i, 'end'], bins)
            if bin:
                cat = genes.loc[i, 'COG']
                if type(bin[0]) == type('str'):
                    genes_df.loc[bin[0], 'gene_density'] += 1
                    genes_complete_df.loc[bin[0], cat] += 1
                elif type(bin[0]) == type(['list']):
                    for b in bin[0]:
                        genes_complete_df.loc[b, cat] += 1
                        genes_df.loc[b, 'gene_density'] += 1
        genes_df.to_csv('{}{}_genes.csv'.format(outputDir, referenceSampleName), sep='\t')
        genes_complete_df.to_csv('{}{}_full_genes.csv'.format(outputDir, referenceSampleName), sep='\t')

    genes_df = pd.read_csv('{}{}_genes.csv'.format(outputDir, referenceSampleName), sep='\t', index_col=0)
    return genes_df

def _add_epigenetics(bins, epigeneticDir):
    if not os.path.isfile('{}{}_metadata.csv'.format(outputDir, referenceSampleName)):
        metadf = pd.DataFrame(0, index=bins.index, columns=['H3K27AC1', 'H3K27M3', 'H3K4M2', 'H3K9M3'])
        PDB_3K27AC1 = pd.read_csv('{}H3K27AC.PEAKS.bed'.format(epigeneticDir), sep='\t', header=None)
        PDB_3K27AC1.columns = ['Chrom', 'start', 'end', 'id', 'H3K27AC1', 'drop']
        for i in PDB_3K27AC1.index:
            bin = _get_bin(PDB_3K27AC1.loc[i, 'Chrom'], PDB_3K27AC1.loc[i, 'start'], PDB_3K27AC1.loc[i, 'end'], bins)
            if bin:
                if type(bin[0]) == type('str'):
                    metadf.loc[bin[0], 'H3K27AC1'] += 1
                elif type(bin[0]) == type(['list']):
                    for b in bin[0]:
                        metadf.loc[b, 'H3K27AC1'] += 1

        PDB_H3K27M3 = pd.read_csv('{}H3K27M3.PEAKS.bed'.format(epigeneticDir), sep='\t', header=None)
        PDB_H3K27M3.columns = ['Chrom', 'start', 'end', 'id', 'H3K27M3', 'drop']
        for i in PDB_H3K27M3.index:
            bin = _get_bin(PDB_H3K27M3.loc[i, 'Chrom'], PDB_H3K27M3.loc[i, 'start'], PDB_H3K27M3.loc[i, 'end'], bins)
            if bin:
                if type(bin[0]) == type('str'):
                    metadf.loc[bin[0], 'H3K27M3'] += 1
                elif type(bin[0]) == type(['list']):
                    for b in bin[0]:
                        metadf.loc[b, 'H3K27M3'] += 1

        PDB_H3K4M2 = pd.read_csv('{}H3K4M2.PEAKS.bed'.format(epigeneticDir), sep='\t', header=None)
        PDB_H3K4M2.columns = ['Chrom', 'start', 'end', 'id', 'H3K4M2', 'drop']
        for i in PDB_H3K4M2.index:
            bin = _get_bin(PDB_H3K4M2.loc[i, 'Chrom'], PDB_H3K4M2.loc[i, 'start'], PDB_H3K4M2.loc[i, 'end'], bins)
            if bin:
                if type(bin[0]) == type('str'):
                    metadf.loc[bin[0], 'H3K4M2'] += 1
                elif type(bin[0]) == type(['list']):
                    for b in bin[0]:
                        metadf.loc[b, 'H3K4M2'] += 1

        PDB_H3K9M3 = pd.read_csv('{}H3K9M3.PEAKS.bed'.format(epigeneticDir), sep='\t',  header=None)
        PDB_H3K9M3.columns = ['Chrom', 'start', 'end', 'id', 'H3K9M3', 'drop']
        for i in PDB_H3K9M3.index:
            bin = _get_bin(PDB_H3K9M3.loc[i, 'Chrom'], PDB_H3K9M3.loc[i, 'start'], PDB_H3K9M3.loc[i, 'end'], bins)
            if bin:
                if type(bin[0]) == type('str'):
                    metadf.loc[bin[0], 'H3K9M3'] += 1
                elif type(bin[0]) == type(['list']):
                    for b in bin[0]:
                        metadf.loc[b, 'H3K9M3'] += 1
        metadf.to_csv('{}{}_metadata.csv'.format(outputDir, referenceSampleName), sep='\t')

    metadf = pd.read_csv('{}{}_metadata.csv'.format(outputDir, referenceSampleName), sep='\t', index_col=0)
    return metadf

def get_figure_stats(filt_LR_df):
    filt_LR_df = filt_LR_df.iloc[range(0, len(filt_LR_df), 2)]
    hue_order = ['core', 'AGR', 'centromere']
    sns.violinplot(data=filt_LR_df.loc[:, ['0', 'edge_type']], x='0', y='edge_type')
    plt.show()
    plt.close()

    df = pd.DataFrame(0, index=[], columns=['0','edge_type'])

    for i in filt_LR_df.index:
        type1 = [x for x in filt_LR_df.loc[i, 'edge_type'].split('_') if x!= 'l'][0]
        type2 = [x for x in filt_LR_df.loc[i, 'edge_type'].split('_') if x!= 'l'][1]
        df = df.append({'0': filt_LR_df.loc[i, '0'], 'edge_type': type1}, ignore_index=True)
        df = df.append({'0': filt_LR_df.loc[i, '0'], 'edge_type': type2}, ignore_index=True)

    sns.violinplot(data=df, x='0', y='edge_type', order=hue_order)
    plt.show()

    sns.histplot(data=df, x='0', hue='edge_type', element="poly", common_norm=False, hue_order=hue_order, stat='probability', log_scale=True)
    plt.show()

    for type1 in ['centromere_centromere', 'AGR_l_centromere', 'AGR_l_AGR_l', 'core_core', 'AGR_l_core', 'centromere_core']:
        for type2 in ['centromere_centromere', 'AGR_l_centromere', 'AGR_l_AGR_l', 'core_core', 'AGR_l_core', 'centromere_core']:
            if type1 != type2:
                df1 = filt_LR_df[filt_LR_df.loc[:, 'edge_type'] == type1]
                df2 = filt_LR_df[filt_LR_df.loc[:, 'edge_type'] == type2]
                df1 = df1.loc[:, '0']
                df2 = df2.loc[:, '0']
                df1 = df1.values
                df2 = df2.values
                if commandLineArgs['verbose']:
                    print(type1, type2)
                stat, p = ranksums(x=df1, y=df2)
                if commandLineArgs['verbose']:
                    print('ranksums {}'.format(p))
                stat, p = ttest_ind(x=df1,y=df2)
                if commandLineArgs['verbose']:
                    print('ttest {}'.format(p))
                    print('mannwithneyu')
                    print (mannwhitneyu(x=df1, y=df2))

    for type1 in ['AGR', 'core', 'centromere']:
        for type2 in ['AGR', 'core', 'centromere']:
            if type1 != type2:
                df1 = df[df.loc[:, 'edge_type'] == type1]
                df2 = df[df.loc[:, 'edge_type'] == type2]
                df1 = df1.loc[:, '0']
                df2 = df2.loc[:, '0']
                df1 = df1.values
                df2 = df2.values
                if commandLineArgs['verbose']:
                    print (type1, type2)
                stat, p = ranksums(x=df1, y=df2)
                if commandLineArgs['verbose']:
                    print('ranksums {}'.format(p))
                stat, p = ttest_ind(x=df1, y=df2)
                if commandLineArgs['verbose']:
                    print('ttest {}'.format(p))
                    print ('mannwithneyu')
                    print(mannwhitneyu(x=df1, y=df2))
        if commandLineArgs['verbose']:
            print (type1)
            print (np.mean(df1))

def plot_epigenetics(meta, filt_LR_df, bin_annot):
    filt_LR_df_0 = filt_LR_df.set_index('level_0')
    filt_LR_df_1 = filt_LR_df.set_index('level_1')


    for bintype in set(list(bin_annot.loc[:, 'bintype'])):
        type_bins = bin_annot[bin_annot.loc[:, 'bintype'] == bintype]
        if commandLineArgs['verbose']:
            print(bintype)
            print (type_bins)
        I0 = [x for x in type_bins.index if x in filt_LR_df_0.index]
        I1 = [x for x in type_bins.index if x in filt_LR_df_1.index]
        I = list(set(I0).union(set(I1)))
        not_I = [x for x in type_bins.index if x not in I]
        I = [x for x in I if x in meta.index]
        not_I = [x for x in not_I if x in meta.index]
        RANGE = len(I) + len(not_I)
        plotting_df = pd.DataFrame(0, index=range(RANGE), columns=['binname', 'bintype', 'metadata_category', 'activity', 'value'])
        count = 0

        pvals=[]
        for metadata in ['H3K27AC1','H3K27M3','H3K4M2','H3K9M3','ATAC']:
            ttest = ttest_ind(meta.loc[I, metadata], meta.loc[not_I, metadata])
            pvals.append(ttest.pvalue)
            if commandLineArgs['verbose']:
                print (metadata)
                print(ranksums(meta.loc[I, metadata], meta.loc[not_I, metadata]))
                print (ttest_ind(meta.loc[I, metadata], meta.loc[not_I, metadata]))
                print(ttest)

        adj_pvals = statsmodels.stats.multitest.fdrcorrection(pvals)
        for x in range(5):
            print(['H3K27AC1', 'H3K27M3', 'H3K4M2', 'H3K9M3', 'ATAC'][x], adj_pvals[1][x])

        for i in I:
            for metadata in ['H3K27AC1','H3K27M3','H3K4M2','H3K9M3','ATAC']:
                plotting_df.loc[count, ['binname', 'bintype', 'metadata_category', 'activity', 'value']] = [i, bintype, metadata, 'active', meta.loc[i, metadata]]
                count += 1
        for i in not_I:
            for metadata in ['H3K27AC1','H3K27M3','H3K4M2','H3K9M3','ATAC']:
                plotting_df.loc[count, ['binname', 'bintype', 'metadata_category', 'activity', 'value']] = [i, bintype, metadata, 'inactive', meta.loc[i, metadata]]
                count += 1

        if commandLineArgs['verbose']:
            print (plotting_df)
            print (meta)

        sns.boxplot(data=plotting_df, x='metadata_category', y='value', hue='activity', palette='muted')
        plt.title(bintype)
        plt.show()

def plot_epigenetics_AGR(meta, filt_LR_df, bin_annot):

    agr = bin_annot[bin_annot.loc[:, 'bintype'] == 'AGR_l']
    core = bin_annot[bin_annot.loc[:, 'bintype'] == 'core']

    AGR_bins = [x for x in agr.index if x in meta.index]
    core_bins = [x for x in core.index if x in meta.index]
    plotting_df = pd.DataFrame(0, index=range(len(meta)), columns=['binname', 'bintype', 'metadata_category', 'activity', 'value'])
    count = 0
    if commandLineArgs['verbose']:
        print(ranksums(meta.loc[AGR_bins], meta.loc[core_bins]))
        print(ttest_ind(meta.loc[AGR_bins], meta.loc[core_bins]))

    pvals = []
    for metadata in ['H3K27AC1', 'H3K27M3', 'H3K4M2', 'H3K9M3', 'ATAC']:
        ttest = ttest_ind(meta.loc[AGR_bins, metadata], meta.loc[core_bins, metadata])
        pvals.append(ttest.pvalue)
        if commandLineArgs['verbose']:
            print(metadata)
            print(ranksums(meta.loc[AGR_bins, metadata], meta.loc[core_bins, metadata]))
            print(ttest_ind(meta.loc[AGR_bins, metadata], meta.loc[core_bins, metadata]))
            print(ttest)


    adj_pvals = statsmodels.stats.multitest.fdrcorrection(pvals)

    for x in range(5):
        if commandLineArgs['verbose']:
            print (['H3K27AC1', 'H3K27M3', 'H3K4M2', 'H3K9M3', 'ATAC'][x], adj_pvals[1][x])

    for i in AGR_bins:
        for metadata in ['H3K27AC1', 'H3K27M3', 'H3K4M2', 'H3K9M3', 'ATAC']:
            if meta.loc[i, metadata] > 0:
                plotting_df.loc[count, ['binname', 'bintype', 'metadata_category', 'activity', 'value']] = [i, 'AGR_l', metadata, 'active', meta.loc[i, metadata]]
                count += 1

    for i in core_bins:
        for metadata in ['H3K27AC1', 'H3K27M3', 'H3K4M2', 'H3K9M3', 'ATAC']:
            if meta.loc[i, metadata] > 0:
                plotting_df.loc[count, ['binname', 'bintype', 'metadata_category', 'activity', 'value']] = [i, 'core', metadata, 'inactive', meta.loc[i, metadata]]
                count += 1

    if commandLineArgs['verbose']:
        print(plotting_df)

    plotting_df = plotting_df.loc[~(plotting_df == 0).all(axis=1)]
    sns.boxplot(data=plotting_df, x='metadata_category', y='value', hue='bintype', palette='muted')
    plt.title('AGR_vs_core')
    plt.show()

def plot_TE(TE_full, TE, filt_LR_df, bin_annot):
    filt_LR_df_0 = filt_LR_df.set_index('level_0')
    filt_LR_df_1 = filt_LR_df.set_index('level_1')


    for bintype in set(list(bin_annot.loc[:, 'bintype'])):
        if commandLineArgs['verbose']:
            print (bintype)
        type_bins = bin_annot[bin_annot.loc[:, 'bintype'] == bintype]
        if commandLineArgs['verbose']:
            print (type_bins)
        I0 = [x for x in type_bins.index if x in filt_LR_df_0.index]
        I1 = [x for x in type_bins.index if x in filt_LR_df_1.index]
        I = list(set(I0).union(set(I1)))
        not_I = [x for x in type_bins.index if x not in I]
        I = [x for x in I if x in TE_full.index]
        not_I = [x for x in not_I if x in TE_full.index]
        RANGE = len(I) + len(not_I)
        plotting_df_full = pd.DataFrame(0, index=range(RANGE), columns=['binname', 'bintype', 'metadata_category', 'activity', 'value'])
        plotting_df = pd.DataFrame(0, index=range(RANGE), columns=['binname', 'bintype', 'metadata_category', 'activity', 'value'])
        count = 0
        if commandLineArgs['verbose']:
            print (ranksums(TE_full.loc[I], TE_full.loc[not_I]))
            print(ttest_ind(TE_full.loc[I], TE_full.loc[not_I]))
            print (np.mean(TE_full.loc[I]), np.mean(TE_full.loc[not_I]))
        for metadata in ['LTR/Copia', 'DNA/Mutator', 'LTR/Gypsy', 'LINE/I', 'Unspecified']:
            if commandLineArgs['verbose']:
                print (metadata)
                print(ranksums(TE_full.loc[I, metadata], TE_full.loc[not_I, metadata]))
                print (ttest_ind(TE_full.loc[I, metadata], TE_full.loc[not_I, metadata]))
        for i in I:
            for metadata in ['LTR/Copia', 'DNA/Mutator', 'LTR/Gypsy', 'LINE/I', 'Unspecified']:
                if TE_full.loc[i, metadata] >0:
                    plotting_df_full.loc[count, ['binname', 'bintype', 'metadata_category', 'activity', 'value']] = [i, bintype, metadata, 'active', TE_full.loc[i, metadata]]
                    count += 1
        for i in not_I:
            for metadata in ['LTR/Copia', 'DNA/Mutator', 'LTR/Gypsy', 'LINE/I', 'Unspecified']:
                if TE_full.loc[i, metadata] >0:
                    plotting_df_full.loc[count, ['binname', 'bintype', 'metadata_category', 'activity', 'value']] = [i, bintype, metadata, 'inactive', TE_full.loc[i, metadata]]
                    count += 1
        for i in I:
            for metadata in ['TE']:
                if TE.loc[i, metadata] >0:
                    plotting_df.loc[count, ['binname', 'bintype', 'metadata_category', 'activity', 'value']] = [i, bintype, metadata, 'active', TE.loc[i, metadata]]
                    count += 1
        for i in not_I:
            for metadata in ['TE']:
                if TE.loc[i, metadata] >0:
                    plotting_df.loc[count, ['binname', 'bintype', 'metadata_category', 'activity', 'value']] = [i, bintype, metadata, 'inactive', TE.loc[i, metadata]]
                    count += 1

        if commandLineArgs['verbose']:
            print (plotting_df)
            print(plotting_df_full)

        plotting_df = plotting_df.loc[~(plotting_df == 0).all(axis=1)]
        sns.boxplot(data=plotting_df, x='metadata_category', y='value', hue='activity', palette='muted')
        plt.title(bintype)
        plt.show()

        plotting_df = plotting_df_full.loc[~(plotting_df_full == 0).all(axis=1)]
        sns.boxplot(data=plotting_df_full, x='metadata_category', y='value', hue='activity', palette='muted')
        plt.title(bintype)
        plt.show()

def plot_genes(genes_full, filt_LR_df, bin_annot):
    filt_LR_df_0 = filt_LR_df.set_index('level_0')
    filt_LR_df_1 = filt_LR_df.set_index('level_1')

    for bintype in set(list(bin_annot.loc[:, 'bintype'])):
        print (bintype)
        type_bins = bin_annot[bin_annot.loc[:, 'bintype'] == bintype]
        I0 = [x for x in type_bins.index if x in filt_LR_df_0.index]
        I1 = [x for x in type_bins.index if x in filt_LR_df_1.index]
        I = list(set(I0).union(set(I1)))
        not_I = [x for x in type_bins.index if x not in I]
        I = [x for x in I if x in genes_full.index]
        not_I = [x for x in not_I if x in genes_full.index]
        RANGE = len(I) + len(not_I)
        plotting_df = pd.DataFrame(0, index=range(RANGE), columns=['binname', 'bintype', 'metadata_category', 'activity', 'value'])
        count = 0
        #print (ranksums(genes_full.loc[I], genes_full.loc[not_I]))
        print(ttest_ind(genes_full.loc[I], genes_full.loc[not_I]))
        print (np.mean(genes_full.loc[I]), np.mean(genes_full.loc[not_I]))
        pvals = []

        for metadata in genes_full.columns:
            #print(ranksums(genes_full.loc[I, metadata], genes_full.loc[not_I, metadata]))
            print (ttest_ind(genes_full.loc[I, metadata], genes_full.loc[not_I, metadata]))
            pval = ttest_ind(genes_full.loc[I, metadata], genes_full.loc[not_I, metadata])
            print (type(pval.pvalue))

            if type(pval.pvalue) != np.nan:
                print(metadata)
                pvals.append(ttest_ind(genes_full.loc[I, metadata], genes_full.loc[not_I, metadata]).pvalue)

        adj_pvals = statsmodels.stats.multitest.fdrcorrection(pvals)
        print (adj_pvals)

        for x in range(len(genes_full.columns)):
            print(genes_full.columns[x], adj_pvals[1][x])

        for i in I:
            for metadata in genes_full.columns :
                if genes_full.loc[i, metadata] >0:
                    plotting_df.loc[count, ['binname', 'bintype', 'metadata_category', 'activity', 'value']] = [i, bintype, metadata, 'active', genes_full.loc[i, metadata]]
                    count += 1
        for i in not_I:
            for metadata in genes_full.columns:
                if genes_full.loc[i, metadata] >0:
                    plotting_df.loc[count, ['binname', 'bintype', 'metadata_category', 'activity', 'value']] = [i, bintype, metadata, 'inactive', genes_full.loc[i, metadata]]
                    count += 1

        print (plotting_df)
        plotting_df = plotting_df.loc[~(plotting_df == 0).all(axis=1)]
        sns.boxplot(data=plotting_df, x='metadata_category', y='value', hue='activity', palette='muted')
        plt.title(bintype)
        plt.show()

def get_distance_from_dup(stA, enA, chrA, stB, enB, chrB, dups):
    #print (dups)

    stA = int(stA); enA = int(enA);
    stB = int(stB); enB = int(enB)
    #print (stA, enA, chrA, stB, enB, chrB)

    dist1 = []; dist2 = []
    match = []; step = []
    chr_dups = dups[dups.loc[:, 'ref_chr'] == chrA]
    #print (chr_dups)
    #chr_dups.to_csv('debug.csv', sep='\t')
    for dup_i in chr_dups.index:
        #check if it's inside the dup
        if stA > chr_dups.loc[dup_i, 'ref_start'] and enA < chr_dups.loc[dup_i, 'ref_stop']:
            #LR on one end is in a dup, now find if the other match is also in the other dup
            if np.all(enB > chr_dups.loc[dup_i, 'que_start'] and enB < chr_dups.loc[dup_i, 'que_stop'] and chr_dups.loc[dup_i, 'que_chr'] == chrB):
                #other bin also in the other dup
                dist1.append(0)
                dist2.append(0)
                match.append(dup_i)
                step.append(1)

            if abs(stB - chr_dups.loc[dup_i, 'que_start']) < abs(50000) and chr_dups.loc[dup_i, 'que_chr'] == chrB:
                dist1.append(0)
                dist2.append(stB - chr_dups.loc[dup_i, 'que_start'])
                match.append(dup_i)
                step.append(2)

            if abs(chr_dups.loc[dup_i, 'que_stop'] - enB) < abs(50000) and chr_dups.loc[dup_i, 'que_chr'] == chrB:
                dist1.append(0)
                dist2.append(enB - chr_dups.loc[dup_i, 'que_stop'])
                match.append(dup_i)
                step.append(3)

        #check if it's close to the start
        if abs(stA - chr_dups.loc[dup_i, 'ref_start']) < abs(50000) and chr_dups.loc[dup_i, 'que_chr'] == chrB:
            if enB > chr_dups.loc[dup_i, 'que_start'] and enB < chr_dups.loc[dup_i, 'que_stop']:
                #other bin also in the other dup
                dist1.append(stA - chr_dups.loc[dup_i, 'ref_start'])
                dist2.append(0)
                match.append(dup_i)
                step.append(4)

            if abs(stB - chr_dups.loc[dup_i, 'que_start']) < abs(50000) and chr_dups.loc[dup_i, 'que_chr'] == chrB:
                dist1.append(stA - chr_dups.loc[dup_i, 'ref_start'])
                dist2.append(stB - chr_dups.loc[dup_i, 'que_start'])
                match.append(dup_i)
                step.append(5)

            if abs(chr_dups.loc[dup_i, 'que_stop'] - enB) < abs(50000) and chr_dups.loc[dup_i, 'que_chr'] == chrB:
                dist1.append(stA - chr_dups.loc[dup_i, 'ref_start'])
                dist2.append(enB - chr_dups.loc[dup_i, 'que_stop'])
                match.append(dup_i)
                step.append(6)

        #check if it's close to the end
        if abs(chr_dups.loc[dup_i, 'ref_stop'] - enA) < abs(50000) and chr_dups.loc[dup_i, 'que_chr'] == chrB:
            if enB > chr_dups.loc[dup_i, 'que_start'] and enB < chr_dups.loc[dup_i, 'que_stop']:
                #other bin also in the other dup
                dist1.append(chr_dups.loc[dup_i, 'ref_stop'] - enA)
                dist2.append(0)
                match.append(dup_i)
                step.append(7)

            if abs(stB - chr_dups.loc[dup_i, 'que_start']) < abs(50000) and chr_dups.loc[dup_i, 'que_chr'] == chrB:
                dist1.append(chr_dups.loc[dup_i, 'ref_stop'] - enA)
                dist2.append(stB - chr_dups.loc[dup_i, 'que_start'])
                match.append(dup_i)
                step.append(8)

            if abs(chr_dups.loc[dup_i, 'que_stop'] - enB) < abs(50000) and chr_dups.loc[dup_i, 'que_chr'] == chrB:
                dist1.append(chr_dups.loc[dup_i, 'ref_stop'] - enA)
                dist2.append(enB - chr_dups.loc[dup_i, 'que_stop'])
                match.append(dup_i)
                step.append(9)

    chr_dups = dups[dups.loc[:, 'que_chr'] == chrB]
    for dup_i in chr_dups.index:
        #check if it's inside the dup
        if np.all(stA > chr_dups.loc[dup_i, 'ref_start'] and enA < chr_dups.loc[dup_i, 'ref_stop'] and dups.loc[dup_i, 'ref_chr'] == chrA):
            #LR on one end is in a dup, now find if the other match is also in the other dup
            if enB > chr_dups.loc[dup_i, 'que_start'] and enB < chr_dups.loc[dup_i, 'que_stop']:
                #other bin also in the other dup
                dist1.append(0)
                dist2.append(0)
                match.append(dup_i)
                step.append(10)

            if abs(stB - chr_dups.loc[dup_i, 'que_start']) < abs(50000):
                dist1.append(0)
                dist2.append(stB - chr_dups.loc[dup_i, 'que_start'])
                match.append(dup_i)
                step.append(11)

            if abs(chr_dups.loc[dup_i, 'que_stop'] - enB) < abs(50000):
                dist1.append(0)
                dist2.append(enB - chr_dups.loc[dup_i, 'que_stop'])
                match.append(dup_i)
                step.append(12)

        #check if it's close to the start
        if abs(stA - chr_dups.loc[dup_i, 'ref_start']) < abs(50000) and dups.loc[dup_i, 'ref_chr'] == chrA:
            if enB > chr_dups.loc[dup_i, 'que_start'] and enB < chr_dups.loc[dup_i, 'que_stop']:
                #other bin also in the other dup
                dist1.append(stA - chr_dups.loc[dup_i, 'ref_start'])
                dist2.append(0)
                match.append(dup_i)
                step.append(13)

            if abs(stB - chr_dups.loc[dup_i, 'que_start']) < abs(50000):
                dist1.append(stA - chr_dups.loc[dup_i, 'ref_start'])
                dist2.append(stB - chr_dups.loc[dup_i, 'que_start'])
                match.append(dup_i)
                step.append(14)

            if abs(chr_dups.loc[dup_i, 'que_stop'] - enB) < abs(50000):
                dist1.append(stA - chr_dups.loc[dup_i, 'ref_start'])
                dist2.append(enB - chr_dups.loc[dup_i, 'que_stop'])
                match.append(dup_i)
                step.append(15)

        #check if it's close to the end
        if abs(chr_dups.loc[dup_i, 'ref_stop'] - enA) < abs(50000) and dups.loc[dup_i, 'ref_chr'] == chrA:
            if enB > chr_dups.loc[dup_i, 'que_start'] and enB < chr_dups.loc[dup_i, 'que_stop']:
                #other bin also in the other dup
                dist1.append(chr_dups.loc[dup_i, 'ref_stop'] - enA)
                dist2.append(0)
                match.append(dup_i)
                step.append(16)

            if abs(stB - chr_dups.loc[dup_i, 'que_start']) < abs(50000):
                dist1.append(chr_dups.loc[dup_i, 'ref_stop'] - enA)
                dist2.append(stB - chr_dups.loc[dup_i, 'que_start'])
                match.append(dup_i)
                step.append(17)

            if abs(chr_dups.loc[dup_i, 'que_stop'] - enB) < abs(50000):
                dist1.append(chr_dups.loc[dup_i, 'ref_stop'] - enA)
                dist2.append(enB - chr_dups.loc[dup_i, 'que_stop'])
                match.append(dup_i)
                step.append(18)

    if match:
        best = 1000000

        for m in range(len(match)):
            if (abs(dist1[m])+abs(dist2[m]))/2 < best:
                best_id = dups.iloc[match[m]]
                best = (abs(dist1[m])+abs(dist2[m]))/2
        return best_id, best

    return None, None

def association_with_dist(filt_LR_df, dups):
    withdup_same = 0;    withdup_other = 0
    withoutdup_same = 0;    withoutdup_other = 0
    matches = [];   _matches = []
    filt_LR_df.reset_index(inplace=True)

    for _I in range(0, len(filt_LR_df), 2):
        I = filt_LR_df.index[_I]
        if 'centromere' not in filt_LR_df.loc[I, 'edge_type'].split('_'):
            bin1 = filt_LR_df.loc[I, 'level_0']
            bin2 = filt_LR_df.loc[I, 'level_1']

            chr1 = bin1.split("_")[0]
            start1 = bin1.split("_")[1]
            end1 = bin1.split("_")[2]

            chr2 = bin2.split("_")[0]
            start2 = bin2.split("_")[1]
            end2 = bin2.split("_")[2]
            best_match, distance = get_distance_from_dup(start1, end1, chr1, start2, end2, chr2, dups)

            if np.any(best_match):
                dup1 = '{}_{}_{}'.format(best_match.loc['ref_chr'], best_match.loc['ref_start'], best_match.loc['ref_stop'])
                dup2 = '{}_{}_{}'.format(best_match.loc['que_chr'], best_match.loc['que_start'], best_match.loc['que_stop'])
                matches.append((bin1, bin2, dup1, dup2, distance))
                #print (bin1, bin2, distance, filt_LR_df.loc[I, 'edge_type'])
                if bin1.split("_")[0] == bin2.split("_")[0]:
                    withdup_same+=1
                else:
                    withdup_other+=1
            else:
                _matches.append((start1, end1, chr1, start2, end2, chr2))
                if bin1.split("_")[0] == bin2.split("_")[0]:
                    withoutdup_same+=1
                else:
                    withoutdup_other+=1

    multiindex_df = filt_LR_df.set_index(['level_0', 'level_1'])
    distance_df = pd.DataFrame(index=[], columns=['level_0', 'level_1', 'distance', 'edge_type'])
    for match in matches:
        distance_df = distance_df.append({'level_0': match[0], 'level_1' : match[1], 'distance': match[4], 'edge_type' : multiindex_df.loc[(match[0], match[1]), 'edge_type']}, ignore_index=True)
    #sns.histplot(data=distance_df, x='distance', hue='edge_type', multiple='stack', hue_order=['AGR_l_core', 'AGR_l_AGR_l', 'core_core'])
    #plt.show()
    print ('out ouf {} interactions, {} co-occur with duplications and {} do not'.format(withdup_same+withdup_other+withoutdup_same+withoutdup_other, withdup_same+withdup_other, withoutdup_same+withoutdup_other))
    #print (distance_df)
    return withdup_same+withdup_other+withoutdup_same+withoutdup_other, withdup_same+withdup_other, withoutdup_same+withoutdup_other, distance_df

def association_with_dist_core(filt_LR_df, dups):
    withdup_same = 0;    withdup_other = 0
    withoutdup_same = 0;    withoutdup_other = 0
    matches = [];   _matches = []
    #filt_LR_df.reset_index(inplace=True)

    for _I in range(0, len(filt_LR_df), 2):
        I = filt_LR_df.index[_I]
        if 'centromere' not in filt_LR_df.loc[I, 'edge_type'].split('_'):
            if 'core' in filt_LR_df.loc[I, 'edge_type'].split('_'):
                bin1 = filt_LR_df.loc[I, 'level_0']
                bin2 = filt_LR_df.loc[I, 'level_1']

                chr1 = bin1.split("_")[0]
                start1 = bin1.split("_")[1]
                end1 = bin1.split("_")[2]

                chr2 = bin2.split("_")[0]
                start2 = bin2.split("_")[1]
                end2 = bin2.split("_")[2]
                best_match, distance = get_distance_from_dup(start1, end1, chr1, start2, end2, chr2, dups)

                if np.any(best_match):
                    dup1 = '{}_{}_{}'.format(best_match.loc['ref_chr'], best_match.loc['ref_start'], best_match.loc['ref_stop'])
                    dup2 = '{}_{}_{}'.format(best_match.loc['que_chr'], best_match.loc['que_start'], best_match.loc['que_stop'])
                    matches.append((bin1, bin2, dup1, dup2, distance))
                    #print (bin1, bin2, distance, filt_LR_df.loc[I, 'edge_type'])
                    if bin1.split("_")[0] == bin2.split("_")[0]:
                        withdup_same+=1
                    else:
                        withdup_other+=1
                else:
                    _matches.append((start1, end1, chr1, start2, end2, chr2))
                    if bin1.split("_")[0] == bin2.split("_")[0]:
                        withoutdup_same+=1
                    else:
                        withoutdup_other+=1

    multiindex_df = filt_LR_df.set_index(['level_0', 'level_1'])
    distance_df = pd.DataFrame(index=[], columns=['level_0', 'level_1', 'distance', 'edge_type'])
    for match in matches:
        distance_df = distance_df.append({'level_0': match[0], 'level_1' : match[1], 'distance': match[4], 'edge_type' : multiindex_df.loc[(match[0], match[1]), 'edge_type']}, ignore_index=True)
    #sns.histplot(data=distance_df, x='distance', hue='edge_type', multiple='stack', hue_order=['AGR_l_core', 'AGR_l_AGR_l', 'core_core'])
    #plt.show()
    print ('out ouf {} core interactions, {} co-occur with duplications and {} do not'.format(withdup_same+withdup_other+withoutdup_same+withoutdup_other, withdup_same+withdup_other, withoutdup_same+withoutdup_other))
    #print (distance_df)
    return withdup_same+withdup_other+withoutdup_same+withoutdup_other, withdup_same+withdup_other, withoutdup_same+withoutdup_other


def association_with_dist_AGR(filt_LR_df, dups):
    withdup_same = 0;    withdup_other = 0
    withoutdup_same = 0;    withoutdup_other = 0
    matches = [];   _matches = []
    #filt_LR_df.reset_index(inplace=True)

    for _I in range(0, len(filt_LR_df), 2):
        I = filt_LR_df.index[_I]
        if 'centromere' not in filt_LR_df.loc[I, 'edge_type'].split('_'):
            if 'AGR' in filt_LR_df.loc[I, 'edge_type'].split('_'):
                bin1 = filt_LR_df.loc[I, 'level_0']
                bin2 = filt_LR_df.loc[I, 'level_1']

                chr1 = bin1.split("_")[0]
                start1 = bin1.split("_")[1]
                end1 = bin1.split("_")[2]

                chr2 = bin2.split("_")[0]
                start2 = bin2.split("_")[1]
                end2 = bin2.split("_")[2]
                best_match, distance = get_distance_from_dup(start1, end1, chr1, start2, end2, chr2, dups)

                if np.any(best_match):
                    dup1 = '{}_{}_{}'.format(best_match.loc['ref_chr'], best_match.loc['ref_start'], best_match.loc['ref_stop'])
                    dup2 = '{}_{}_{}'.format(best_match.loc['que_chr'], best_match.loc['que_start'], best_match.loc['que_stop'])
                    matches.append((bin1, bin2, dup1, dup2, distance))
                    #print (bin1, bin2, distance, filt_LR_df.loc[I, 'edge_type'])
                    if bin1.split("_")[0] == bin2.split("_")[0]:
                        withdup_same+=1
                    else:
                        withdup_other+=1
                else:
                    _matches.append((start1, end1, chr1, start2, end2, chr2))
                    if bin1.split("_")[0] == bin2.split("_")[0]:
                        withoutdup_same+=1
                    else:
                        withoutdup_other+=1

    multiindex_df = filt_LR_df.set_index(['level_0', 'level_1'])
    distance_df = pd.DataFrame(index=[], columns=['level_0', 'level_1', 'distance', 'edge_type'])
    for match in matches:
        distance_df = distance_df.append({'level_0': match[0], 'level_1' : match[1], 'distance': match[4], 'edge_type' : multiindex_df.loc[(match[0], match[1]), 'edge_type']}, ignore_index=True)
    #sns.histplot(data=distance_df, x='distance', hue='edge_type', multiple='stack', hue_order=['AGR_l_core', 'AGR_l_AGR_l', 'core_core'])
    #plt.show()
    print ('out ouf {} AGR interactions, {} co-occur with duplications and {} do not'.format(withdup_same+withdup_other+withoutdup_same+withoutdup_other, withdup_same+withdup_other, withoutdup_same+withoutdup_other))
    #print (distance_df)
    return withdup_same+withdup_other+withoutdup_same+withoutdup_other, withdup_same+withdup_other, withoutdup_same+withoutdup_other

def make_background(bin_annot, size, df_size):
    #size == full > all genome [sans centromere]
    #size == AGR/core > only those regions
    if size == 'full':
        bins_selection = bin_annot[bin_annot.loc[:, 'bintype'] != 'centromere']
    else:
        bins_selection = bin_annot[bin_annot.loc[:, 'bintype'] == size]
    #print (df_size)
    new_df = pd.DataFrame(index=[], columns=['level_0', 'level_1', 'edge_type', '0'])
    for x in range(df_size):
        bins = random.sample(list(bins_selection.index), 2)
        new_df = new_df.append({'level_0' : bins[0], 'level_1' : bins[1], 'edge_type' : '{}_{}'.format(bin_annot.loc[bins[0], 'bintype'],bin_annot.loc[bins[1], 'bintype']), '0' : 100}, ignore_index=True)
    return new_df

def get_vdls17_genes(binID):
    gene_count_df = pd.DataFrame(0, index=binID.index, columns=['gene_density'])
    genes = pd.read_csv('/home/vittorio/work/metadata/vdls17_genes.bed',sep='\t', header=None)
    for I in genes.index:
        B= _get_bin(genes.loc[I, 0], genes.loc[I, 1], genes.loc[I, 2], binID)
        if B:
            if type(B[0]) == type(['list']):
                for b in B[0]:
                    gene_count_df.loc[b, 'gene_density']+=1
            else:
                gene_count_df.loc[B[0], 'gene_density'] += 1
    print (gene_count_df)
    gene_count_df.to_csv('gene_count_vdls17.csv',sep='\t')



def commandLineParser():
    '''
    Parses input and holds default values.
    Returns a dictionary with all the parameters for the wrapper
    '''
    parser = argparse.ArgumentParser(description='Pipeline for Hic analysis related to "Three-dimensional chromatin organization promotes genome evolution in a fungal plant pathogen"')
    local = os.getcwd()
    #General argnuments to specify where to put/find files
    parser.add_argument('-cb', '--centromereBedDir', type=str, required=True, help='path to the folder with the bed files containing the centromere coordinates. Bed files baseName should match the genome id')
    parser.add_argument('-sd', '--syntenyDir', type=str, required=False, default=False, help='path to the folder with the bed files containing the synteny coordinates. Bed files name should follow the notation /path/to/directory/{genomeID}_merged_dist{mergingDistanceBasePairs}_size{minimalSizeAfterMerge}.bed')
    parser.add_argument('-sf', '--syntenyFiles', type=str, required=False, default=False, help='comma separated list of the paths to the the bed files containing the synteny coordinates.')
    parser.add_argument('-asd', '--AGRSyntenyDir', type=str, required=False, default=False, help='path to the folder with the bed files containing the AGR coordinates in the different genomes. Bed files name should follow the notation /path/to/directory/{genomeID}_merged_dist{mergingDistanceBasePairs}_size{minimalSizeAfterMerge}.bed')
    parser.add_argument('-asf', '--AGRSyntenyFiles', type=str, required=False, default=False, help='comma separated list of the paths to the the bed files containing the AGR coordinates in the different genomes.')
    parser.add_argument('-agr', '--JR2AGR', type=str, required=True, help='path to the bed file containing JR2 agr coordinates')
    parser.add_argument('-ug', '--unmaskedGenomeDir', type=str, required=True, help='path to the unmasked genome directory.')
    parser.add_argument('-mg', '--maskedGenomeDir', type=str, required=True, help='path to the masked genome directory.')
    parser.add_argument('-sm', '--selfMapping', type=str, required=True, help='path to the directory containing the genome self mapping for duplication detection.')
    parser.add_argument('-op', '--outputPlots', type=str, required=True, help='path to the directory to place the figures. It will be created if it does not exist.')
    parser.add_argument('-oe', '--observedExpectedTsv', type=str, required=True, help='path to the observed/expected tsv file folder')
    parser.add_argument('-rg', '--referenceObsExp', type=str, required=True, help='path to the expected observed tsv file for the reference genome.')
    parser.add_argument('-rb', '--referenceBins', type=str, required=True, help='path to the bed file containing the bins start-end position for the reference genome.')
    parser.add_argument('-rba', '--referenceBinsAnnotation', type=str, required=True, help='path to the tsv file containing the bins annotation for the reference genome.')
    parser.add_argument('-rc', '--referenceCentromeres', type=str, required=True, help='path to the bed file containing the bins start-end position for the reference centromeres.')
    parser.add_argument('-ga', '--geneAnnotation', type=str, required=True, help='path to the csv file containing the gene annotation for the reference genome.')
    parser.add_argument('-te', '--teFile', type=str, required=True, help='path to the csv file containing the TE annotation for the reference genome.')
    parser.add_argument('-ed', '--epigeneticDir', type=str, required=True, help='path to the folder containing the epigenetics PEAK files for the reference genome.')
    parser.add_argument('-o', '--outputDir', type=str, required=True, default=local, help='Main output path, generates different subfolders to organize output files.')
    parser.add_argument('-v', '--verbose', required=False, default=False, action='store_true', help='Set verbosity while running')
    return vars(parser.parse_args())

def parse_command_line_args(commandLineArgs):
    #check if all the variables are provided and parse the output
    if commandLineArgs['syntenyFiles']:
        commandLineArgs['syntenyFiles'] = commandLineArgs['syntenyFiles'].split(',')
    if commandLineArgs['AGRSyntenyFiles']:
        commandLineArgs['AGRSyntenyFiles'] = commandLineArgs['AGRSyntenyFiles'].split(',')
    if not commandLineArgs['syntenyDir'] and not commandLineArgs['syntenyFiles']:
        raise ValueError('No synteny files provided')
    if not commandLineArgs['AGRSyntenyDir'] and not commandLineArgs['AGRSyntenyFiles']:
        raise ValueError('No AGR synteny files provided')
    if not os.path.isdir(commandLineArgs['outputPlots']):
        os.mkdir(commandLineArgs['outputPlots'])
    if not os.path.isdir(commandLineArgs['output']):
        os.mkdir(commandLineArgs['output'])
    return commandLineArgs

if __name__ == "__main__":
    commandLineArgs = commandLineParser()
    commandLineArgs = parse_command_line_args(commandLineArgs)

    if commandLineArgs['syntenyDir']:
        synteny_dir = commandLineArgs['syntenyDir']

    if commandLineArgs['AGRSyntenyDir']:
        agr_dir = commandLineArgs['AGRSyntenyDir']

    agr = pd.read_csv(commandLineArgs['JR2AGR'], sep='\t')
    unmasked_genome_dir = commandLineArgs['unmaskedGenomeDir']
    masked_genome_dir = commandLineArgs['maskedGenomeDir']
    self_mapping_dir = commandLineArgs['selfMapping']
    plot_dir = commandLineArgs['outputPlots']
    tsv_dir = commandLineArgs['observedExpectedTsv']
    referenceObsExp = commandLineArgs['referenceObsExp']
    referenceBins = commandLineArgs['referenceBins']
    referenceBinsAnnotation = commandLineArgs['referenceBinsAnnotation']
    referenceCentromeres = commandLineArgs['referenceCentromeres']
    outputDir = commandLineArgs['outputDir']
    geneAnnotation = commandLineArgs['geneAnnotation']
    teFile = commandLineArgs['teFile']
    epigeneticDir = commandLineArgs['epigeneticDir']
    referenceSampleName = referenceObsExp.split('/')[-1].lstrip('binID_').split('.')[0]

    if commandLineArgs['verbose']:
        print('Reference sample name: {}'.format(referenceSampleName))

    obs_exp = pd.read_csv(referenceObsExp, sep='\t')
    bins = pd.read_csv(referenceBins, sep='\t', index_col=0)
    bins = remove_UnplacedScaffolds(bins)
    missing_bins = check_bins(bins)
    centromeres = pd.read_csv(referenceCentromeres, sep='\t', index_col=0, header=None)
    bin_annot = pd.read_csv(referenceBinsAnnotation, sep='\t', index_col=0)
    bin_annot = bin_annot.loc[:, ['chr', 'start', 'stop', 'bintype']]

    counts_txt = 'genome\tcore\tcentromere\tAGR_l\tcore_core\tcentromere_core\tAGR_l_core\tcentromere_centromere\tAGR_l_centromere\tAGR_l_AGR_l\n'
    dup_table = 'genome\tAGR_AGR\tAGR_core\tcore_core\n'

    for bin in bin_annot.index:
        if bin_annot.loc[bin, 'bintype'] == 'AGR':
            bin_annot.loc[bin, 'bintype'] = 'AGR_l'
    obs_exp_df = pd.read_csv(referenceObsExp, index_col=0, sep='\t')
    obs_exp_df = obs_exp_df.loc[bins.index, bins.index]
    c2c_interactions = get_centromere_interactions(obs_exp_df, bin_annot)
    treshold = c2c_interactions > 0.001
    treshold = c2c_interactions[treshold]
    tr = np.mean(treshold)
    cis_LR_dict, trans_LR_dict = get_LR_bins(obs_exp_df.index)

    if not os.path.isdir('{}df/{}/'.format(outputDir, referenceSampleName)):
        os.mkdir('{}df/{}/'.format(outputDir, referenceSampleName))

    if not os.path.isfile('{}df/{}_filtered_df.csv'.format(outputDir, referenceSampleName)):
        filt_df = obs_exp_df >= tr
        filt_df = obs_exp_df[filt_df].fillna(0)
        filt_df.dropna(inplace=True)
        if commandLineArgs['verbose']:
            print('current threshold = {}\n{} LR interactions left'.format(tr, np.sum(filt_df.values.astype(bool) * 1)))
            print('with {} local interactions'.format(np.sum(filt_df.values.astype(bool) * 1)))
        filt_LR_df = filter_df(filt_df, cis_LR_dict, tsv_dir)
    else:
        filt_LR_df = pd.read_csv('{}df/{}_filtered_df.csv'.format(outputDir, referenceSampleName), sep='\t')

    filt_LR_df.sort_values(by='0', axis=0, inplace=True, ascending=False)
    counts_txt, filt_LR_df = make_histogram(filt_LR_df, bin_annot, plot_dir, referenceSampleName, counts_txt)
    duplications = _genome2genome(masked_genome_dir, referenceSampleName, self_mapping_dir)

    if commandLineArgs['verbose']:
        print ('there are {} duplications'.format(len(duplications)))

    df_size, withdup, withoutdup = association_with_dist_AGR(filt_LR_df, duplications)
    df_size, withdup, withoutdup = association_with_dist_core(filt_LR_df, duplications)
    df_size, withdup, withoutdup, distance_df = association_with_dist(filt_LR_df, duplications)
    edge_dup_type_count = {edge_type: len(distance_df[distance_df.loc[:, 'edge_type'] == edge_type]) for edge_type in ['AGR_l_AGR_l', 'AGR_l_core', 'core_core']}
    dup_table += '{}\t{}\t{}\t{}\n'.format(referenceSampleName, edge_dup_type_count['AGR_l_AGR_l'], edge_dup_type_count['AGR_l_core'], edge_dup_type_count['core_core'])

    if commandLineArgs['verbose']:
        print (dup_table)

    calculate_AGR_enrichment(filt_LR_df, bin_annot)
    dup_dict = find_dup_bins(duplications, bins, cis_LR_dict)
    overlap_duplications(filt_LR_df, dup_dict)

    meta = _add_epigenetics(bins)

    plot_epigenetics(meta, filt_LR_df, bin_annot)
    plot_epigenetics_AGR(meta, filt_LR_df, bin_annot)

    genes = add_genes(bins, geneAnnotation)
    plot_genes(genes, filt_LR_df, bin_annot)
    TE, full_TE = add_TE(bins, teFile)
    plot_TE(full_TE, TE, filt_LR_df, bin_annot)

    # positive set
    df_size, withdup, withoutdup = association_with_dist_AGR(filt_LR_df, duplications)

    background_dup_with = []
    for iteration in range(100):
        mock_filt_LR_df = make_background(bin_annot, 'full', df_size)
        df_size_mock, mockwithdup, mockwithoutdup = association_with_dist(mock_filt_LR_df, duplications)
        background_dup_with.append(mockwithdup)
        if commandLineArgs['verbose']:
            print('iteration number {}'.format(iteration))
            print(mockwithdup, mockwithoutdup, df_size_mock)

    if commandLineArgs['verbose']:
        print('z-test (and 1 sample t-test) for observing {} LR interactions overlap with duplications versus the genome-wide distribution in {} [average {} overlaps]:'.format(withdup, referenceSampleName, np.mean(background_dup_with)))
        print(ztest(background_dup_with, value=withdup))
        print(ttest_1samp(background_dup_with, withdup))


    background_dup_with = []
    for iteration in range(100):
        mock_filt_LR_df = make_background(bin_annot, 'AGR_l', df_size)
        df_size_mock, mockwithdup, mockwithoutdup = association_with_dist(mock_filt_LR_df, duplications)
        background_dup_with.append(mockwithdup)
        if commandLineArgs['verbose']:
            print('iteration number {}'.format(iteration))
            print(mockwithdup, mockwithoutdup, df_size_mock)

    if commandLineArgs['verbose']:
        print('z-test for observing {} LR interactions overlap with duplications versus the AGR distribution:'.format(withdup))
    
    exp_with_dup = round(np.mean(background_dup_with))
    exp_without_dup = 100 - exp_with_dup
    obs_with_dup = withdup
    obs_without_dup = withoutdup
    if commandLineArgs['verbose']:
        print (background_dup_with)
        print(ztest(background_dup_with, value=withdup))
        print(ttest_1samp(background_dup_with, withdup))
        print(chisquare())

    filt_LR_df_0 = filt_LR_df.set_index(['level_0'])
    get_figure_stats(filt_LR_df)

    for bintype in ['core', 'AGR_l', 'centromere']:
        I = bin_annot[bin_annot.loc[:, 'bintype'] == bintype]
        I = I.index
        I0 = [x for x in I if x in filt_LR_df_0.index]
        vals = obs_exp_df.loc[I0].values
        counts0 = filt_LR_df_0.loc[I0, '0']

    #other vsp
    tsv_dir = commandLineArgs['observedExpectedTsv']
    genomes = [x for x in os.listdir(tsv_dir) if x.startswith('binID_Verticillium')]

    for genome in genomes:
        sample_name = genome.lstrip('binID_').split('.')[0]
        if commandLineArgs['verbose']:
            print (sample_name)
        obs_exp = pd.read_csv('{}{}_1_obsexp_nonzero.tsv'.format(tsv_dir, sample_name), sep='\t')
        bins = pd.read_csv('{}binID_{}.tsv'.format(tsv_dir, sample_name), sep='\t', index_col=0)
        bins = remove_UnplacedScaffolds(bins)
        missing_bins = check_bins(bins)
        centromeres = pd.read_csv('{}{}.bed'.format(commandLineArgs['centromereBedDir'], sample_name), sep='\t', index_col=0, header=None)
        binning_distance = 1000;    min_block_size = 2000
        #print ('{}_{}_bin_annot.tsv'.format(tsv_dir, sample_name))
        if not os.path.isfile('{}_{}_bin_annot.tsv'.format(tsv_dir, sample_name)):
            bin_annot = pd.DataFrame(index=bins.index, columns=['chr', 'start', 'stop', 'bintype'])
            synteny_file1 = '{}{}_merged_dist{}_size{}.bed'.format(synteny_dir, sample_name, binning_distance, min_block_size)
            synteny_file2 = '{}{}_merged_dist{}_size{}.bed'.format(agr_dir, sample_name, binning_distance, min_block_size)
            if os.path.isfile(synteny_file1) and os.path.isfile(synteny_file2):
                bin_annot = _annotate_AGRl(bin_annot, synteny_file2, bins)
                bin_annot = annotate_centromere(bins, centromeres, bin_annot)
                bin_annot.to_csv('{}_{}_bin_annot.tsv'.format(tsv_dir, sample_name), sep='\t')
                bin_annot = pd.read_csv('{}_{}_bin_annot.tsv'.format(tsv_dir, sample_name), sep='\t', index_col=0)
        else:
            bin_annot = pd.read_csv('{}_{}_bin_annot.tsv'.format(tsv_dir, sample_name), sep='\t', index_col=0)

        obs_exp_df = pd.read_csv('{}{}_1_df_obsexp_nonzero.tsv'.format(tsv_dir, sample_name), index_col=0, sep='\t')
        obs_exp_df = obs_exp_df.loc[bins.index, bins.index]
        c2c_interactions = get_centromere_interactions(obs_exp_df, bin_annot)
        nonzeroc2c = c2c_interactions>0.001
        nonzeroc2c = c2c_interactions[nonzeroc2c]
        tr = np.mean(nonzeroc2c)
        cis_LR_dict, trans_LR_dict = get_LR_bins(obs_exp_df.index)
        if not os.path.isdir('{}df/{}/'.format(outputDir, sample_name)):
            os.mkdir('{}df/{}/'.format(outputDir, sample_name))
        if not os.path.isfile('{}{}_filtered_df.csv'.format(tsv_dir, sample_name)):
            filt_df = obs_exp_df>=tr
            filt_df = obs_exp_df[filt_df].fillna(0)
            filt_df.dropna(inplace=True)
            filt_LR_df = filter_df(filt_df, cis_LR_dict, tsv_dir)
        else:
            filt_LR_df = pd.read_csv('{}{}_filtered_df.csv'.format(tsv_dir, sample_name), sep='\t')

        filt_LR_df.sort_values(by='0', axis=0, inplace=True, ascending=False)
        counts_txt, filt_LR_df = make_histogram(filt_LR_df, bin_annot, plot_dir, sample_name, counts_txt)
        duplications = genome2genome(masked_genome_dir, sample_name, self_mapping_dir)
        if commandLineArgs['verbose']:
            print ('there are {} duplications'.format(len(duplications)))
        df_size, withdup, withoutdup = association_with_dist_AGR(filt_LR_df, duplications)
        df_size, withdup, withoutdup = association_with_dist_core(filt_LR_df, duplications)
        df_size, withdup, withoutdup, distance_df = association_with_dist(filt_LR_df, duplications)
        edge_dup_type_count = {edge_type: len(distance_df[distance_df.loc[:, 'edge_type'] == edge_type]) for edge_type in ['AGR_l_AGR_l', 'AGR_l_core', 'core_core']}
        dup_table += '{}\t{}\t{}\t{}\n'.format(sample_name, edge_dup_type_count['AGR_l_AGR_l'], edge_dup_type_count['AGR_l_core'], edge_dup_type_count['core_core'])
        if commandLineArgs['verbose']:
            print(dup_table)
        calculate_AGR_enrichment(filt_LR_df, bin_annot)

        background_dup_with = []
        for iteration in range(100):
            mock_filt_LR_df = make_background(bin_annot, 'full', df_size)
            df_size_mock, mockwithdup, mockwithoutdup, _ = association_with_dist(mock_filt_LR_df, duplications)

            if commandLineArgs['verbose']:
                print(mockwithdup, mockwithoutdup, df_size_mock)
                print('iteration number {}'.format(iteration))
            background_dup_with.append(mockwithdup)

        if commandLineArgs['verbose']:
            print('z-test (and 1 sample t-test) for observing {} LR interactions overlap with duplications versus the genome-wide distribution in {} [average {} overlaps]:'.format(withdup, sample_name, np.mean(background_dup_with)))
            print(ztest(background_dup_with, value=withdup))
            print(ttest_1samp(background_dup_with, withdup))

        df_size, withdup, withoutdup = association_with_dist_AGR(filt_LR_df, duplications)

        background_dup_with = []
        for iteration in range(100):
            #print('iteration number {}'.format(iteration))
            mock_filt_LR_df = make_background(bin_annot, 'AGR_l', df_size)
            df_size_mock, mockwithdup, mockwithoutdup, _ = association_with_dist(mock_filt_LR_df, duplications)
            #print(mockwithdup, mockwithoutdup, df_size_mock)
            background_dup_with.append(mockwithdup)

        if commandLineArgs['verbose']:
            print('z-test (and 1 sample t-test) for observing {} LR interactions overlap with duplications versus the AGR distribution in {} [average {} overlaps]:'.format(withdup, sample_name, round(np.mean(background_dup_with))))
            print(ztest(background_dup_with, value=withdup))
            print(ttest_1samp(background_dup_with, withdup))


    filt_LR_df_0 = filt_LR_df.set_index(['level_0'])

    for bintype in ['core', 'AGR_l', 'centromere']:
        I = bin_annot[bin_annot.loc[:, 'bintype'] == bintype]
        I = I.index
        I0 = [x for x in I if x in filt_LR_df_0.index]
        vals = obs_exp_df.loc[I0].values
        counts0 = filt_LR_df_0.loc[I0, '0']

        if commandLineArgs['verbose']:
            print (bintype)
            print (np.mean(vals))
            print (np.mean(counts0))

    get_figure_stats(filt_LR_df)

