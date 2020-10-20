# Tough Tables 
This repository contains Python code to generate the **Tough Tables (2T)** dataset, a dataset for
benchmarking table annotation algorithms on the *CEA* and *CTA* tasks. The target KG is DBpedia 2016-10.

The 2T dataset is available in Zenodo
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3840647.svg)](https://doi.org/10.5281/zenodo.3840647)

The 2T dataset is compliant with the SemTab 2019 format. It is possible to evaluate
all the annotation algorithms that produce a results file compatible with the SemTab
challenge submission file format.
For details, see [SemTab 2019](http://www.cs.ox.ac.uk/isg/challenges/sem-tab/)
([CEA](https://www.aicrowd.com/challenges/iswc-2019-cell-entity-annotation-cea-challenge),
[CTA](https://www.aicrowd.com/challenges/iswc-2019-column-type-annotation-cta-challenge)).

## Reference
This work is based on the following paper:
> Cutrona, V., Bianchi, F., Jimenez-Ruiz, E. and Palmonari, M. (2020). Tough Tables: Carefully Evaluating 
> Entity Linking for Tabular Data. ISWC 2020, LNCS 12507, pp. 1–16.

## How to use
### Installing
The code is developed for Python 3.8.
Install all the required packages listed in the `requirements.txt` file.
```shell script
virtualenv -p python3.8 venv # we suggest to create a virtual environment
source venv/bin/activate
pip install -r requirements.txt
```

### Create the Gold standard
The following command reads the tables under the `control` and `tough` directories, 
and generate the gold standard (GS).
```shell script
python tough_tables.py dbp make_gs --output_folder ./gs \
                                   --endpoint http://dbpedia.org/sparql
```
Note: the resultant GS may differ in different executions, due to the unsorted results of SPARQL queries.

### Create tables and ground truth for CEA
Starting from the GS tables, the following command generates a) the set of tables to annotate,
and b) the ground truth file.
```shell script
python tough_tables.py to_cea --input_folder ./gs \
                              --output_tables_folder ./2T_cea/tables \
                              --output_gs_folder ./2T_cea \
                              --endpoint http://dbpedia.org/sparql \
                              --sameas_file dbp_sameas.json
```
The `dbp_sameas.zip` file contains the collection of all the sameAs links used to build 2T.

### Derive CTA Ground Truth from CEA
It is possible to derive the CTA ground truth from the CEA ground truth using a majority voting strategy.
```shell script
python tough_tables.py cta_from_cea --cea_gs_file ./2T_cea/2T_gt.csv \
                                    --output_gs_folder ./2T_cta  \
                                    --instance_types_file ./instance_types_en.ttl  \
                                    --ontology_file ./dbpedia_2016-10.nt
```
The command requires two external sources:
- the `instance_types_file`, containing the list of all the DBpedia instances and their types
  ([.ttl](http://downloads.dbpedia.org/2016-10/core-i18n/en/instance_types_en.ttl.bz2))
- the DBpedia 2016-10 ontology ([.nt](http://downloads.dbpedia.org/2016-10/dbpedia_2016-10.nt))

### Score an algorithm
To score an algorithm, run:
```shell script
python tough_tables.py score --annotations_file <your_annotation_file.csv>
```
The annotations file format must be the same used in the SemTab 2019 challenge (tab_id, col_id, row_id, annotation).
Along with the overall result (*ALL*), all the performance metrics are computed for each category of tables.
A radar plot (`<your_annotation_file>.pdf`) is saved in the submission file directory.

### Utils
Other utility commands are available in the script. See the full list by executing:
```shell script
python em_gs.py --help
``` 
