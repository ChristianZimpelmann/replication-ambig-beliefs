# Repository for the research project: The Distribution of Ambiguity Attitudes

This repository contains the replication package for the paper ["The Distribution of Ambiguity Attitudes"](https://www.wiwi.uni-bonn.de/gaudecker/_static/ambiguity-attitudes.pdf) by Hans-Martin von Gaudecker, Axel Wogrolly, and Christian Zimpelmann.

## Data

The data is based on the [LISS (Longitudinal Internet Studies for the Social sciences)](https://www.lissdata.nl/) -- an internet-based household panel administered by CentERdata (Tilburg University, The Netherlands).

In `ambig_beliefs/original_data/liss-data`, general data cleaning steps (renaming of variables and values, merging of yearly files, etc.) are conducted on the raw data files of the LISS. This step is based on a general LISS data cleaning repository and also run automatically by `pytask` (see below). See [this documentation](https://liss-data-management-documentation.readthedocs.io/en/latest/#) for more information.

Before running the project, download all LISS raw data files and put them in the directory `ambig_beliefs/original_data/liss-data/data`. Alternatively, you can contact us and we can give you access to the raw files once you have registered for LISS data access. As of 2022 the data sets that we collected ourselves are not yet publicly available on the website, but we are happy to share them.

## Run the replication

The replication of figures and tables proceeds as follows:

- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) on your computer and create a conda environment based on `environment.yml`.
- Activate the environment and follow the manual steps listed at the top of `environment.yml`.
- Run `pytask`, which will perform
  - the general data cleaning, project specific data cleaning, 
  - analyses, 
  - and creation of figures and tables
- All created tables and figures will be saved in the directory `out`
