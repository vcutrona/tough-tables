# Tough Tables 
This repository contains Python code to generate the **Tough Tables (2T)** dataset, a dataset for
benchmarking table annotation algorithms on the *CEA* and *CTA* tasks (as defined in the
[SemTab](http://www.cs.ox.ac.uk/isg/challenges/sem-tab/) challenge).
The target KG is DBpedia 2016-10.

The 2T dataset is available in Zenodo
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3840646.svg)](https://doi.org/10.5281/zenodo.3840646)

The 2T dataset is compliant with the SemTab 2019 format. It is possible to evaluate
all the annotation algorithms that produce a results file compatible with the SemTab
challenge submission file format.
For details, see [SemTab 2019](http://www.cs.ox.ac.uk/isg/challenges/sem-tab/2019/index.html)
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
and generates the gold standard (GS).
```shell script
python tough_tables.py make_gs --output_folder ./gs \
                               --endpoint http://dbpedia.org/sparql
```
Note: the resultant GS may differ in different executions, due to the unsorted results of SPARQL queries.

### Create tables and ground truth for CEA
Starting from the GS tables, the following command generates a) the set of tables to annotate,
and b) the ground truth file.
```shell script
python tough_tables.py to_cea --input_folder ./gs \
                              --output_tables_folder ./2T/tables \
                              --output_gs_folder ./2T/gt \
                              --output_target_folder ./2T/targets \
                              --endpoint http://dbpedia.org/sparql \
                              --sameas_file dbp_sameas.json
```
The `resources/dbp_sameas.json` file contains the collection of all the sameAs links used to build 2T.

### Derive CTA Ground Truth from CEA
It is possible to derive the CTA ground truth from the CEA ground truth using a majority voting strategy.
```shell script
python tough_tables.py cta_from_cea --cea_gs_file ./2T/gt/CEA_2T_gt.csv \
                                    --output_gs_folder ./2T/gt  \
                                    --output_target_folder  ./2T/targets \
                                    --instance_types_file ./instance_types_en.ttl  \
                                    --ontology_file ./dbpedia_2016-10.nt
```
The command requires two external sources:
- the `instance_types_en` file containing the list of all the DBpedia instances and their types
  ([.ttl](http://downloads.dbpedia.org/2016-10/core-i18n/en/instance_types_en.ttl.bz2))
- the DBpedia ontology ([.nt](http://downloads.dbpedia.org/2016-10/dbpedia_2016-10.nt))

### Score an algorithm (CEA)
To score an algorithm, run:
```shell script
python tough_tables.py score_cea --annotations_file <your_annotation_file.csv> \
                                 --gs_file ./2T_cea/2T_gt.csv
```
The annotations file format must be the same used in the SemTab 2019 challenge (tab_id, col_id, row_id, annotation).
Along with the overall result (*ALL*), all the performance metrics are computed for each category of tables.
A radar plot (`<your_annotation_file>.pdf`) is saved in the submission file directory.

### Utils
Other utility commands are available in the script. See the full list by executing:
```shell script
python tough_tables.py --help
``` 


## SemTab2020 Version
The 2T dataset has been converted into its corresponding Wikidata version and it has been adopted as part of the
SemTab2020 challenge - Round 4.

**NOTE: the new format for CEA is <tab_id, row_id, col_id, entity>.**
Check out the [SemTab 2020](http://www.cs.ox.ac.uk/isg/challenges/sem-tab/2020/index.html) website for more details.

The conversion script `to_wikidata.py` requires the following files to be downloaded and put in the `resources`
directory to generate a conversion map:

- wd-sameas-all-wikis ([.ttl](https://downloads.dbpedia.org/repo/dbpedia/wikidata/sameas-all-wikis/2020.08.01/sameas-all-wikis.ttl.bz2))
- interlanguage_links_en ([.ttl](http://downloads.dbpedia.org/2016-10/core-i18n/en/interlanguage_links_en.ttl.bz2))
- transitive_redirects_en ([.ttl](http://downloads.dbpedia.org/2016-10/core-i18n/en/transitive_redirects_en.ttl.bz2))

**NOTE: commented lines (e.g., "# started 2017-07-06T12:05:32Z") must be removed from the above files.**

A pre-computed conversion map is available under the `resources` directory (`db_wd_conversion_map.pickle`).


## Credits
Along with packages listed in the requirements, this repository uses the
[tabular-data-semantics-py](https://github.com/ernestojimenezruiz/tabular-data-semantics-py) package to query
SPARQL endpoints. We slightly adapted the package to meet our needs (the resultant version is available under the
`tabular_semantics` directory).

In previous versions, we exploited the [py-sparql-transformer](https://github.com/D2KLab/py-sparql-transformer)
package for querying the DBpedia SPARQL endpoint.
