---
toc: true
url: tulane_bioinf_1
covercopy: <a href="https://www.hood.edu/graduateacademicsprograms/bioinformatics-ms">© HOOD College</a>
priority: 10000
date: 2021-08-24 13:24:57
title: "Introduction of Bioinformatics"
ytitle: "生物信息学导论"
description: "Classes notes of the Bioinformatics"
excerpt: "Classes 1, Fundamental Concept"
tags: [Classes, Bioinformatics, Tulane Classes]
category: [Notes, Class, Tulane, Bioinformatics]
cover: "https://www.hood.edu/sites/default/files/styles/width_720/public/content/program/hero/istock_56013860_molecule_computer_2500.jpg?itok=L8YHtcy2"
thumbnail: "https://cdn.iconscout.com/icon/premium/png-256-thumb/bioinformatics-2355481-1985942.png"
---

## Introduction of Bioinformatics
Data units:
|![Unit of byte](https://qph.fs.quoracdn.net/main-qimg-8ff2446aa03c6d8486b3bd8171f9c26b.webp)|
|:-:|
|[&copy; Quora](https://www.quora.com/What-is-the-biggest-byte)|

### 5V of Big Data

*[Veracity]: conformity to facts; accuracy.
- Volume
- Velocity
- Variety
- Veracity
- Value


## Difinition of Bioinformatic

>Bioinformatics is the use of computer databases and computer algorithms to analyze proteins, genes, and the complete collection of deoxyribonucleic acid (DNA) that comprises an organism (the genome).
> -- Bioinformatics and Functional Genomics (BFG) Book, 3rd Edition (2015):

>Bioinformatics refers to “research, development, or application of computational tools and approaches for expand- ing the use of biological, medical, behavioral, or health data, including those to acquire, store, organize, analyze, or visualize such data.”
> -- National Institutes of Health (NIH)


GeneCards.org (Human)
Genome.ucsc.edu (Genome browser)

## For tolls for this classes

- [HUGO gene names](https://www.genenames.org/)
- [Uniport](https://www.uniprot.org/)
- [RCSB protein structure databases](https://www.rcsb.org/) (PDB)
- [Human protein Atlas](https://www.proteinatlas.org/)
- [cBioPortal for cancer genomics](https://www.cbioportal.org/)
  - Frequency of the gene mutated in cancers
- [Immune epotope database](https://www.iedb.org/)
  - Antibody Antigen


- Homolog
- ***Analog***: Convergent evelution

## Structure Alignment

## Sequence Alignment

ClsutalW
Muscle
[TCoffee](http://tcoffee.crg.cat/)


## Markov Models for Multisequence Alignment

Wiki: [Hiddne Markov Model](https://en.wikipedia.org/wiki/Hidden_Markov_model)


Profile Hiden Markove Models are simoly a way to represent the features of these aligned sequences from the same family using a statistic model.

Parts of the protein is important to this specific family.
HMMs can make a profile to each protein family.
We can take this profile for familiy specific and align the same protein family.

> PS: The HMMs could build profiles for families. We can using the profile of the pritein family and align our sequence. So we can have a better weighted alignment result by the families feature implied in HMMs profils (pre-bild model)


Markove Chains: Sequences of random varibales in which the futrue variable is detemined by the present variable stochasitic/probabilistic process.

- First Order Markov model: the current event only depends on the immediatetly preceding event.
Second order models uss the two preceding evetns,
THird order models use three, etc.


###  s
Learn all the insert and the deleted probabilistic from the aligned sequence database and build the model. Then we can evaluate the this family of the




Aligned Sequence for build the Markove Model
- Who do we get a proprate aligned sequence?
- What is the golden standard of the apropriate aligned squence?

## How to buidl

- Remvove low occupancy columns (> 50%)
- Assign match states, delete states, insert states from MSA
- Get the path of each sequence
- Count the amino acid freqencies emitted from match or insert states, which are converted into probabilities for each state
- Count...

- Tran model using a given multiple sequence alignment
  - Baum-Welch Algorithm odes the training/parameter estmation for HMM
  - Baum-Welch is an ecpetation-Maximization (EM) algorithm - iterative
  - Converges on a set of probabilities for the model that best fir the multiple alignment you gaie it
  - Only finds a local maxima - good to try different intial conditions
  - Baum-Welch can also perfom an MSA to from a set of unaligned sequences



- faster and more fitable than PIS-blast
- Classification: wihch fimaily of protein belongs.



## Meloular Phyligenies

Describ, Meaning, ==Access Confidence==.


## NGS

### Sanger Sequencing




### reciprocal translocation: long reads align

| ![](https://ars.els-cdn.com/content/image/3-s2.0-B0122270800013100-gr2.jpg) |
| :-------: |
| ![](https://ars.els-cdn.com/content/image/3-s2.0-B0122270800013100-gr1.jpg)     |
| [© C.V.Beechey A.G.Searle](https://www.sciencedirect.com/science/article/pii/B0122270800013100)       |

## Genome Sequencing




## Sequencing and Diseases
## Sample workflow

DNA → Library → capture based selection → Sequencing
DNA →  PCR based selection→ Library → Sequencing

### Exp
Severe combined immunodeficiency syndrome

Il-2/4/7/9/15/21 → X-SCID (γ domain)

==γ domain mutate== failed to active jak3 → Jak3-SCID

### Newborn sscreening
- Amino acid disorder
  - PKU; MSUD
- Fatty acid disorder
  - MCAD; VLCAD
- Organic acid disorders
  - pH disorder

Card screen

- Hemoglobinopathies
  - Sickle cell, SC, S-β-thalassemia
  - MSUD
- Galactosemia
- CF; [CF links](https://www.cff.org/What-is-CF/Testing/Newborn-Screening-for-CF/)
- SCID - recommended by HRSA, detects T cell receptro excision circles (TRECs)
  - T cell recombination.

### TREC screening

*[Thymus]: 胸腺

- False positive
- False negatives
  - Zap70 deficiency
  - MHC Class II deficiency
  - NF-κb essential modulator
  - late-onset ADA
- Positive: Genes tsted: ADA CD3D; CD3E...
  FOXN1: required for the development of Thymus
  - WES, WGS
    - Can detect novel variants
    - Often uses trios or unaffected sibs
    - Can be used in CMC-STAT3 gof, IL-17R mutations


### CF Screening -immunoreactive trypsinogen
- Typically measured by fluoroimmunoassay
- False postive
  - Perinatal asphyxia, infection
  - CF carrier (heterozygote)
- False negatives - rare
  - Lab or specimen error
  - pseudo-Bartter's syndrome
==Positive==:
- Sweat test
  - high salting in sweat; less salting concentration among patients
  - dehydrate
- Genetic testing

**Bordeline sweat test**
- Targeted sequence of CFTR


## Challenges of WGS, WES
- Each individual harbors 2.7-4.2m SNV

### NGS and cancer
- Can compare somatic to germline mutation
  - Exmaple AML
  - Micro-dissected tumor
  - Circulating tumor RNA

Cancer Early Casses
- Lukas Whartman
  - Dx All 2003
  - Treatede with sibling related ..

  - Sequenced the cancer genome for actionable mutations
  - No mutations detected
  - RNAseq found iverexpression in FLT3
  - Remitted with FLT3 inhibitor(Sutent)
- 2nd stem cell transplatnt after remission
- Suffers from GVHD

# Panels
  - Sequences regions of interest
  - Hybridization or PCR based
  - Often disease specific
  - Eg Breast , lung, colon CA
  - Sequence coverage is high (up to 80x)

- GeneDx Panel data -positive yield
- 9.7% for breast
- 13.4% for ovarian
- 14.8% for colon/stomacj
- pathogenic or likely pathogenic mutation in over 8%-15% ofr ...

- Between 70%-92% of the patients remains mutation-negative or undiagnosed
...


- Mutations in PALB2 and ATM in pancreatic CA
- XRCC2, FANCC and BLM in HBOC
- Germline RNA-splice mutations using RNA-seq
- Germline splice variants in BRCA1/BRCA2
- ...
- NF1

### NGS and cancer - Clinical utility
- Diagnosis
- Survival prediction
  - GNAS & KRAS

- Up to 20% of NGS tests were actionable
- Another 50% were actionable if you include mutations that could be targeted by the use of a FDA approved drug for off-label use
- Can identify candidates for anti-EGFR therapies
- Re-classification of tumors.
  - Troditional: look unde the microscopy

Pathology
- Diagnosis
  - Useful in small smaples -FNAs
  - Thyroseq panel for thyroid cancer
- Survival prediction

==Liquid Biopsy==
- Rationable
-Has been used for liquid and solid tumors
- May be useful in lieu of biospy...


### Pharmacogenomics
- Ratoionale
- Exampel
  - Dihydropyrimidine dehydrogenase:D{D
  - Mutations associated with greate toxicity of 5-fluorouracil, capecitiabine and..

### Futrue Directions for CA
- Epigenome
  - CHIP-seq (Bulk cell)
  - ATAC-seq (More advance; nano scale cells.)
-RANseq
  - Transcriptomes including non-coding RNA



### - -

TPM: normalized by the size of the library
- Different tissues has different expression profile and the size of the profile is different.
- Transcripts is different, too.








.