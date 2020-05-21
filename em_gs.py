# coding: utf-8

import argparse
import csv
import json
import logging
import os
import random
import urllib.parse
from datetime import datetime
from statistics import mean, stdev

import numpy as np
import pandas as pd
from SPARQLTransformer import sparqlTransformer
from SPARQLWrapper import SPARQLWrapper, CSV

SPARQL_ENDPOINT = 'http://dbpedia.org/sparql'
TABLE_TYPES = ['DBP', 'WIKI', 'WEB', 'T2D']

random.seed(42)


def _write_df(df, filename, drop=True, strip=True, index=False, header=True, quoting=csv.QUOTE_ALL):
    if drop:
        df = df.drop_duplicates()
    if strip:
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df.to_csv(filename, index=index, header=header, quoting=quoting)


def to_dbp_uri(uri, endpoint):
    db_uri = uri
    try:
        db_uri = urllib.parse.unquote(uri)
        if "http://dbpedia.org/resource/" not in db_uri:
            db_uri = uri.replace(
                "https://en.wikipedia.org/wiki/",
                "http://dbpedia.org/resource/"
            ).replace(
                "http://dbpedia.org/page/",
                "http://dbpedia.org/resource/"
            )
    except:
        logger.critical(f'Impossible to create a valid URI from {db_uri}')
        return None
    query = {
        "proto": {
            "p": "?p$anchor",
            "o": "?o"
        },
        "$limit": 1,
        "$where": f'<{db_uri}> ?p ?o'
    }
    res = _json_exec(query, endpoint)
    if len(res) > 0:
        return db_uri
    logger.warning(f'No DB_URI found for {uri} -> {db_uri}')
    return None


def _json_exec(query, endpoint, debug=False):
    return sparqlTransformer(query, {'endpoint': endpoint, 'debug': debug})


def _sparql_exec(query, endpoint):
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(CSV)
    result = sparql.queryAndConvert()
    return result


def _check_uris(folder):
    with os.scandir(folder) as it:
        for entry in it:
            if entry.name.endswith(".csv") and entry.is_file():
                logger.info(f'Processing file: {entry.path}')
                df = pd.read_csv(entry.path, dtype=object)
                for col in df.columns:
                    if '__URI' in col:
                        for uri in df[col]:
                            if uri is not np.nan and "http://dbpedia.org/resource/" not in uri:
                                logger.error(f"Unvalid URI: {uri}")


def gt_stats(folder):
    stats = {}
    with os.scandir(folder) as it:
        for entry in it:
            if entry.name.endswith(".csv") and entry.is_file():
                logger.info(f'Processing file: {entry.path}')
                df = pd.read_csv(entry.path, dtype=object)
                stats[entry.name] = {
                    'rows': df.shape[0],
                    'columns': 0,
                    'cells': 0,
                    'annotated_columns': 0,
                    'entities': set(),
                    'annotated_cells': 0}
                for col in df.columns:
                    if '__URI' in col:
                        stats[entry.name]['entities'].update(df[col].unique())
                        stats[entry.name]['annotated_cells'] = stats[entry.name]['annotated_cells'] + df[col].dropna().shape[0]
                        stats[entry.name]['annotated_columns'] = stats[entry.name]['annotated_columns'] + 1
                    else:
                        stats[entry.name]['columns'] = stats[entry.name]['columns'] + 1
                        stats[entry.name]['cells'] = stats[entry.name]['cells'] + df[col].shape[0]
                stats[entry.name]['entities'] = list(stats[entry.name]['entities'])

    for t_type in ['ALL'] + TABLE_TYPES:
        if t_type == 'ALL':
            tmp_stats = stats
        else:
            tmp_stats = {k: v for k, v in stats.items() if k.startswith(t_type)}
        total_tables = len(tmp_stats)
        rows_lengths = list(map(lambda x: x['rows'], tmp_stats.values()))
        cols_lengths = list(map(lambda x: x['columns'], tmp_stats.values()))
        cells_lengths = list(map(lambda x: x['cells'], tmp_stats.values()))
        ann_cols_lengths = list(map(lambda x: x['annotated_columns'], tmp_stats.values()))
        ann_cells_lengths = list(map(lambda x: x['annotated_cells'], tmp_stats.values()))
        distinct_entities_lengths = (list(map(lambda x: len(x['entities']), tmp_stats.values())))
        distinct_entities = set()
        for v in tmp_stats.values():
            distinct_entities.update(v['entities'])

        print(t_type)
        print('total tables:', total_tables)

        funcs = [mean, stdev, sum, min, max]
        print('Avg. Rows # (± Std Dev) <tot, min, max>: %.2f ± %.2f <%d, %d, %d> '
              % tuple([f(rows_lengths) for f in funcs]))
        print('Avg. Cols # (± Std Dev) <tot, min, max>: %.2f ± %.2f <%d, %d, %d> '
              % tuple([f(cols_lengths) for f in funcs]))
        print('Avg. Cells # (± Std Dev) <tot, min, max>: %.2f ± %.2f <%d, %d, %d> '
              % tuple([f(cells_lengths) for f in funcs]))
        print('Avg. Columns with target cells # (± Std Dev) <tot, min, max>: %.2f ± %.2f <%d, %d, %d> '
              % tuple([f(ann_cols_lengths) for f in funcs]))
        print('Avg. Target Cells # (± Std Dev) <tot, min, max>: %.2f ± %.2f <%d, %d, %d> '
              % tuple([f(ann_cells_lengths) for f in funcs]))
        s = list([f(distinct_entities_lengths) for f in funcs])
        s[2] = len(distinct_entities)
        print('Avg. Entities # (± Std Dev) <tot, min, max>: %.2f ± %.2f <%d, %d, %d> '
              % tuple(s))
        print("---")
    return stats


def wiki_to_gs(input_dir, output_dir, endpoint, prefix='', suffix=''):
    with os.scandir(input_dir) as it:
        for entry in it:
            if os.path.isfile(f'{output_dir}/{prefix}WIKI_{suffix}{entry.name}'):
                logger.info(f'Skipping file: {entry.path} - {output_dir}/{prefix}WIKI_{suffix}{entry.name} already exists.')
            elif entry.name.endswith(".csv") and entry.is_file():
                logger.info(f'Processing file: {entry.path}')
                df = pd.read_csv(entry.path, dtype=object)
                for col in df.columns:
                    if '__URI' in col:
                        new_values = []
                        for uri in df[col]:
                            # dbp_uri = to_dbp_uri(uri, endpoint)
                            # if dbp_uri is None and uri is not np.nan:
                            # logger.warning(f'No DB_URI found for {uri} in {col}')
                            new_values.append(to_dbp_uri(uri, endpoint))
                        df[col] = new_values
                _write_df(df, f'{output_dir}/{prefix}WIKI_{suffix}{entry.name}')


def web_to_gs(input_dir, output_dir, prefix='', suffix=''):
    with os.scandir(input_dir) as it:
        for entry in it:
            if os.path.isfile(f'{output_dir}/{prefix}WEB{suffix}_{entry.name}'):
                logger.info(f'Skipping file: {entry.path} - {output_dir}/{prefix}WEB{suffix}_{entry.name} already exists.')
            elif entry.name.endswith(".csv") and entry.is_file():
                logger.info(f'Processing file: {entry.path}')
                df = pd.read_csv(entry.path, dtype=object)
                _write_df(df, f'{output_dir}/{prefix}WEB{suffix}_{entry.name}')


def sparql_to_gs(input_dir, output_dir, endpoint, prefix='', suffix=''):
    with os.scandir(input_dir) as it:
        for entry in it:
            if os.path.isfile(f'{output_dir}/{prefix}DBP{suffix}_{entry.name.replace(".rq", ".csv")}'):
                logger.info(f'Skipping file: {entry.path} - {output_dir}/{prefix}DBP{suffix}_{entry.name.replace(".rq", ".csv")} already exists.')
            elif entry.name.endswith(".rq") and entry.is_file():
                logger.info(f'Processing file: {entry.path}')
                with open(f'{output_dir}/{prefix}DBP{suffix}_{entry.name.replace(".rq", ".csv")}', 'wb') as out:
                    out.write(_sparql_exec(open(entry.path, 'r').read(), endpoint))


def t2d_to_gs(input_dir, output_dir, endpoint, prefix='', suffix=''):
    with os.scandir(input_dir) as it:
        for entry in it:
            if os.path.isfile(f'{output_dir}/{prefix}T2D{suffix}_{entry.name}'):
                logger.info(f'Skipping file: {entry.path} - {output_dir}/{prefix}T2D{suffix}_{entry.name} already exists.')
            elif entry.name.endswith(".csv") and entry.is_file():
                logger.info(f'Processing file: {entry.path}')
                df = pd.read_csv(entry.path, dtype=object)
                for col in df.columns:
                    if '__URI' in col:
                        new_values = []
                        for uri in df[col]:
                            new_values.append(to_dbp_uri(str(uri).split(" ")[0], endpoint))
                        df[col] = new_values
                _write_df(df, f'{output_dir}/{prefix}T2D{suffix}_{entry.name}')


"""
Precision = (# correctly annotated cells) / (# annotated cells)
Recall = (# correctly annotated cells) / (# target cells)
F1 Score = (2 * Precision * Recall) / (Precision + Recall)
"""


def precision_score(correct_cells, annotated_cells):
    return float(len(correct_cells)) / len(annotated_cells) if len(annotated_cells) > 0 else 0.0


def recall_score(correct_cells, gt_cell_ent):
    return float(len(correct_cells)) / len(gt_cell_ent.keys())


def f1_score(precision, recall):
    return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


"""
Notes:

6) Annotations for cells out of the target cells are ignored.

1) # denotes the number.

2) F1 Score is used as the primary score; Precision is used as the secondary score.

3) An empty annotation of a cell will lead to an annotated cell; we suggest to exclude the cell with empty annotation in the submission file.
"""


def compute_score(gs_file, submission_file, tables_folder, wrong_cells_file, remove_unseen):
    logger.info(f'GS file: {gs_file}')
    logger.info(f'Annotations file: {submission_file}')
    scores = {}

    gt = pd.read_csv(gs_file, delimiter=',', names=['tab_id', 'col_id', 'row_id', 'entity'],
                     dtype={'tab_id': str, 'col_id': str, 'row_id': str, 'entity': str}, keep_default_na=False)
    sub = pd.read_csv(submission_file, delimiter=',', names=['tab_id', 'col_id', 'row_id', 'entity'],
                      dtype={'tab_id': str, 'col_id': str, 'row_id': str, 'entity': str}, keep_default_na=False)

    if remove_unseen:
        logger.info('Removing unseen tables...')
        gt = gt[gt['tab_id'].isin(sub['tab_id'].unique())]

    gt_cell_ent = dict()
    gt_cell_ent_orig = dict()
    for index, row in gt.iterrows():
        cell = '%s %s %s' % (row['tab_id'], row['col_id'], row['row_id'])
        gt_cell_ent[cell] = urllib.parse.unquote(row['entity']).lower().split(' ')
        gt_cell_ent_orig[cell] = row['entity'].split(' ')

    correct_cells, wrong_cells, annotated_cells = set(), list(), set()
    for index, row in sub.iterrows():
        cell = '%s %s %s' % (row['tab_id'], row['col_id'], row['row_id'])
        if cell in gt_cell_ent:
            if cell in annotated_cells:
                raise Exception("Duplicate cells in the submission file")
            else:
                annotated_cells.add(cell)

            annotation = urllib.parse.unquote(row['entity']).lower()
            if annotation in gt_cell_ent[cell]:
                correct_cells.add(cell)
            else:
                wrong_cells.append({
                    'table': row['tab_id'],
                    'col': int(row['col_id']),
                    'row': int(row['row_id']),
                    'actual': row['entity'],
                    'target': " ".join(gt_cell_ent_orig[cell])
                })
    precision = precision_score(correct_cells, annotated_cells)
    recall = recall_score(correct_cells, gt_cell_ent)
    f1 = f1_score(precision, recall)

    scores['ALL'] = {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    for t in TABLE_TYPES:
        c_cells = {x for x in correct_cells if x.startswith(t)}
        a_cells = {x for x in annotated_cells if x.startswith(t)}
        g_cells = dict(filter(lambda elem: elem[0].startswith(t), gt_cell_ent.items()))

        precision = precision_score(c_cells, a_cells)
        recall = recall_score(c_cells, g_cells)
        f1 = f1_score(precision, recall)
        scores[t] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    wcells = pd.DataFrame(data=wrong_cells)

    if tables_folder:
        pds = {}
        for file in set([x['table'] for x in wrong_cells]):
            pds[file] = pd.read_csv(f'{tables_folder}/{file}.csv', header=None, dtype=object)
        for wc in wrong_cells:
            wc['col_name'] = pds[wc['table']].at[0, wc['col']]  # 0 is always the header row
            wc['value'] = pds[wc['table']].at[wc['row'], wc['col']]
        wcells = pd.DataFrame(data=wrong_cells)[['table', 'col', 'row', 'col_name', 'value', 'actual', 'target']]

    if wrong_cells_file:
        _write_df(wcells, wrong_cells_file, strip=False)

    return scores


def _get_sameas(uri, endpoint):
    sameas = set()
    try:
        sameas_query = {
            "proto": {
                "db": "?db$anchor",
                "sameas": "?sameas"
            },
            "$prefixes": {
                "dbo": "http://dbpedia.org/ontology/"
            },
            "$from": "http://dbpedia.org",
            "$where": '?sameas dbo:wikiPageRedirects|owl:sameAs ?db .',
            "$values": {
                "db": uri
            }
        }
        inverse_sameas_query = {
            "proto": {
                "db": "?main$anchor",
                "sameas": "?sameas",
                "redirected": "?redirected"
            },
            "$prefixes": {
                "dbo": "http://dbpedia.org/ontology/"
            },
            "$from": "http://dbpedia.org",
            "$where": """
                {
                    ?db dbo:wikiPageRedirects ?main .
                    ?main owl:sameAs ?sameas .
                    ?redirected dbo:wikiPageRedirects ?main .
                }
            """,
            "$values": {
                "db": uri
            }
        }

        res1 = _json_exec(sameas_query, endpoint)
        res2 = _json_exec(inverse_sameas_query, endpoint)
        if res1:
            sameas.add(res1[0]["db"])
            if isinstance(res1[0]["sameas"], list):
                sameas = sameas | set(res1[0]["sameas"])
            else:
                sameas.add(res1[0]["sameas"])
        if res2:
            sameas.add(res2[0]["db"])
            if isinstance(res2[0]["sameas"], list):
                sameas = sameas | set(res2[0]["sameas"])
            else:
                sameas.add(res2[0]["sameas"])
            if isinstance(res2[0]["redirected"], list):
                sameas = sameas | set(res2[0]["redirected"])
            else:
                sameas.add(res2[0]["redirected"])
        if not res1 and not res2:
            sameas.add(uri)
    except:
        print("error for URI", uri)
        pass
    return list(filter(lambda x: 'http://dbpedia.org' in x, list(sameas)))


def remove_duplicates_from_gs(input_dir):
    with os.scandir(input_dir) as it:
        for entry in it:
            if entry.name.endswith(".csv") and entry.is_file():
                df = pd.read_csv(entry.path, dtype=object)

                # remove duplicates without considering __URI cols
                df = df[~df[[col for col in df.columns if '__URI' not in col]].duplicated()]
                _write_df(df, entry.path)


def noise_1(input_dir, output_dir):
    with os.scandir(input_dir) as it:
        for entry in it:
            if entry.name.endswith(".csv") and 'MISSP' in entry.name and entry.is_file():
                df = pd.read_csv(entry.path, dtype=object)
                msp_col = [x for x in df.columns if x.endswith('_misspelled')][0]
                pure_col = msp_col.replace("_misspelled", "")
                for i in np.arange(0.0, 1.0, 0.1):
                    i = round(float(i), 2)

                    msk = np.random.rand(len(df)) < i

                    corrupted = df[msk]
                    pure = df[~msk]

                    corrupted = corrupted.drop(columns=[pure_col, f'{pure_col}__URI'])
                    pure = pure.drop(columns=[msp_col, f'{msp_col}__URI'])

                    corrupted.columns = [x.replace("_misspelled", "") for x in corrupted.columns]

                    noisy_df = pd.concat([corrupted, pure])
                    _write_df(noisy_df, f'{output_dir}/{entry.name[:-4]}_NOISE1_{str(i)}.csv')


def random_noise(x):
    if x is not np.nan and not x.isnumeric():
        rnd = random.random()
        if rnd > 0.8 and len(x) > 1:  # duplicate a random char
            rnd_posix = random.randint(1, len(x) - 1)
            x = x[:rnd_posix] + x[:rnd_posix][-1] + x[rnd_posix:]
        elif rnd > 0.3:  # duplicate last letter:
            x = x + x[-1]
        # else -> return the string as is with no intervention
    return x


def noise_2(input_dir, output_dir):
    with os.scandir(input_dir) as it:
        for entry in it:
            if entry.name.endswith(".csv") and entry.is_file():
                df = pd.read_csv(entry.path, dtype=object)
                target_cols = [x for x in df.columns if f'{x}__URI' in df.columns]
                target_cols = random.choices(target_cols, k=random.randrange(len(target_cols)))
                for col in target_cols:
                    df[col] = df[col].apply(random_noise)
                _write_df(df, f'{output_dir}/{entry.name[:-4]}_NOISE2.csv')


def to_cea_format(input_dir, output_tables_dir, output_gs_dir, endpoint, sameas_file):
    if sameas_file is None:
        logger.warning(f'SameAs file not provided. Queries to DBpedia could take some time.')
        sameas_dict = {}
    else:
        sameas_dict = json.load(open(sameas_file, 'r'))
        json.dump(sameas_dict, open(f'dbp_sameas_bkp_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w'), indent=4)

    annotations = []
    ext_annotations = []

    with os.scandir(input_dir) as it:
        for entry in it:
            if entry.name.endswith(".csv") and entry.is_file():
                tab_id = entry.name[:-4]

                logger.info(f'Processing file: {entry.path}')
                df = pd.read_csv(entry.path, dtype=object)

                count = 0
                for col_id, (columnName, columnData) in enumerate(df.iteritems()):
                    if '__URI' in columnName:
                        count = count + 1
                        for row_id, value in columnData.iteritems():
                            if value is not np.nan:
                                # row_id + 1 due to the header row
                                ann = {'tab_id': tab_id,
                                       'col_id': str(col_id - count),
                                       'row_id': str(row_id + 1),
                                       'entity': value}
                                annotations.append(ann)

                                if value not in sameas_dict:
                                    logger.info(f'Getting sameas for entity: {value}')
                                    sameas_entities = _get_sameas(value, endpoint)
                                    for entity in sameas_entities:
                                        sameas_dict[entity] = sameas_entities

                                ann['entity'] = " ".join(sameas_dict[value])
                                ext_annotations.append(ann)

                df = df[[col for col in df.columns if '__URI' not in col]]
                _write_df(df, f'{output_tables_dir}/{entry.name}')

            json.dump(sameas_dict, open('dbp_sameas.json', 'w'), indent=4)
    _write_df(pd.DataFrame(ext_annotations)[['tab_id', 'col_id', 'row_id', 'entity']],
              f'{output_gs_dir}/2T_gt.csv',
              header=False)


def to_mantis_format(gs_dir, tables_dir, tables_list_file):
    tables = []
    with os.scandir(gs_dir) as it:
        for entry in it:
            if entry.name.endswith(".csv") and entry.is_file():
                logger.info(f'Processing file: {entry.path}')
                df = pd.read_csv(entry.path, dtype=object)
                df.to_json(f'{tables_dir}/{entry.name.replace(".csv", ".json")}', orient='records')
                tables.append(entry.name.replace(".csv", ""))
    if tables_list_file:
        logger.info(f'Dumping the list of tables into a JSON file...')
        json.dump(tables, open(tables_list_file, "w"), indent=4)


def to_idlab_format(cea_gs_file, output_dir):
    cea_gs = pd.read_csv(cea_gs_file, names=['filename', 'col_id', 'row_id', 'annotation'], dtype=object)[
        ['filename', 'col_id', 'row_id']]
    for f, df in list(cea_gs.groupby('filename')):
        df['target'] = "cell"
        _write_df(df[['target', 'col_id', 'row_id']], f'{output_dir}/{f}.csv', header=False)


def make_gs(output_folder, endpoint):
    sparql_to_gs('control/query', output_folder, endpoint, prefix='CTRL_')
    wiki_to_gs('control/wiki', output_folder, endpoint, prefix='CTRL_')
    sparql_to_gs('tough/homonyms/queries', output_folder, endpoint, prefix='TOUGH_', suffix='_HOMO')
    web_to_gs('tough/homonyms', output_folder, prefix='TOUGH_', suffix='_HOMO')
    t2d_to_gs('tough/t2d', output_folder, endpoint, prefix='TOUGH_')
    web_to_gs('tough/misspelled', output_folder, prefix='TOUGH_', suffix='_MISSP')
    web_to_gs('tough/open_data', output_folder, prefix='TOUGH_', suffix='_OD')
    _check_uris(output_folder)
    noise_1(output_folder, output_folder)
    noise_2(output_folder, output_folder)
    remove_duplicates_from_gs(output_folder)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    subparsers = argparser.add_subparsers(help='commands')

    wiki_argparser = subparsers.add_parser("wiki", help='Transform Wikipedia tables.')
    wiki_argparser.set_defaults(action='wiki')
    wiki_argparser.add_argument('--input_folder', type=str, default='./wiki',
                                help='Path to the folder containing Wikipedia tables. DEFAULT: ./wiki')
    wiki_argparser.add_argument('--output_folder', type=str, default='./gs',
                                help='Path to output folder. DEFAULT: ./gs')
    wiki_argparser.add_argument('--endpoint', type=str, default=SPARQL_ENDPOINT,
                                help=f'SPARQL endpoint. DEFAULT: {SPARQL_ENDPOINT}')

    web_argparser = subparsers.add_parser("web", help='Transform Web tables.')
    web_argparser.set_defaults(action='web')
    web_argparser.add_argument('--input_folder', type=str, default='./web',
                               help='Path to the folder containing Web tables. DEFAULT: ./web')
    web_argparser.add_argument('--output_folder', type=str, default='./gs',
                               help='Path to output folder. DEFAULT: ./gs')

    dbp_argparser = subparsers.add_parser("dbp", help='Create tables from DBpedia SPARQL queries.')
    dbp_argparser.set_defaults(action='dbp')
    dbp_argparser.add_argument('--input_folder', type=str, default='./dbp',
                               help='Path to the folder containing queries (.rq files). DEFAULT: ./dbp')
    dbp_argparser.add_argument('--output_folder', type=str, default='./gs', help='Path to output folder. DEFAULT: ./gs')
    dbp_argparser.add_argument('--endpoint', type=str, default=SPARQL_ENDPOINT,
                               help=f'SPARQL endpoint. DEFAULT: {SPARQL_ENDPOINT}')

    t2d_argparser = subparsers.add_parser("t2d", help='Transform T2D tables.')
    t2d_argparser.set_defaults(action='t2d')
    t2d_argparser.add_argument('--input_folder', type=str, default='./t2d',
                               help='Path to the folder containing T2D tables. DEFAULT: ./t2d')
    t2d_argparser.add_argument('--output_folder', type=str, default='./gs', help='Path to output folder. DEFAULT: ./gs')
    t2d_argparser.add_argument('--endpoint', type=str, default=SPARQL_ENDPOINT,
                               help=f'SPARQL endpoint. DEFAULT: {SPARQL_ENDPOINT}')

    check_argparser = subparsers.add_parser("check", help='Check if all DBpedia URIs in a table are valid.')
    check_argparser.set_defaults(action='check')
    check_argparser.add_argument('--input_folder', type=str, default='./gs',
                                 help='Path to the folder containing tables. DEFAULT: ./gs')

    stats_argparser = subparsers.add_parser("stats", help='Print stats about the GS.')
    stats_argparser.set_defaults(action='stats')
    stats_argparser.add_argument('--input_folder', type=str, default='./gs',
                                 help='Path to the folder containing tables. DEFAULT: ./gs')

    to_cea_argparser = subparsers.add_parser("to_cea", help='Convert GS tables to CEA format.')
    to_cea_argparser.set_defaults(action='to_cea')
    to_cea_argparser.add_argument('--input_folder', type=str, default='./gs',
                                  help='Path to the folder containing GS tables DEFAULT: ./gs')
    to_cea_argparser.add_argument('--output_tables_folder', type=str, default='./2T_cea/tables',
                                  help='Path to output folder for tables. DEFAULT: ./2T_cea/tables')
    to_cea_argparser.add_argument('--output_gs_folder', type=str, default='./2T_cea',
                                  help='Path to output folder for gold standard files. DEFAULT: ./2T_cea')
    to_cea_argparser.add_argument('--endpoint', type=str, default=SPARQL_ENDPOINT,
                                  help=f'SPARQL endpoint. DEFAULT: {SPARQL_ENDPOINT}')
    to_cea_argparser.add_argument('--sameas_file', type=str, default=None,
                                  help='Provide a JSON file containing sameAs links for DBP entities. DEFAULT: None')

    to_mantis_argparser = subparsers.add_parser("to_mantis", help='Convert GS tables to Mantistable format.')
    to_mantis_argparser.set_defaults(action='to_mantis')
    to_mantis_argparser.add_argument('--input_folder', type=str, default='./2T_cea/tables',
                                     help='Path to the folder containing GS tables. DEFAULT: ./2T_cea/tables')
    to_mantis_argparser.add_argument('--tables_folder', type=str, help='Path to Mantistable folder.')
    to_mantis_argparser.add_argument('--tables_list_file', type=str, default=None,
                                     help='File to store the list of tables to be imported. DEFAULT: None')

    to_idlab_argparser = subparsers.add_parser("to_idlab", help='Convert GS tables to IDLab format.')
    to_idlab_argparser.set_defaults(action='to_idlab')
    to_idlab_argparser.add_argument('--gs_file', type=str, default='./cea_gs/cea_gs_EXT.csv',
                                    help='Path to the ground truth file.')
    to_idlab_argparser.add_argument('--output_folder', type=str, help='Path to IDLab target files folder.')

    scorer_argparser = subparsers.add_parser("score", help='Evaluate your annotation system.')
    scorer_argparser.set_defaults(action='score')
    scorer_argparser.add_argument('--annotations_file', type=str, help='Path to the annotations file (CEA format).')
    scorer_argparser.add_argument('--gs_file', type=str, default='./cea_gs/cea_gs_EXT.csv',
                                  help='Path to the ground truth file. DEFAULT: ./cea_gs/cea_gs_EXT.csv')
    scorer_argparser.add_argument('--tables_folder', type=str, default=None,
                                  help='Path to folder with original tables. Provide it only if you want cells content along with wrong annotations. DEFAULT: None')
    scorer_argparser.add_argument('--wrong_cells_file', type=str, default=None,
                                  help='File to store the wrong cells as CSV.  DEFAULT: None')
    scorer_argparser.add_argument('--remove_unseen', action='store_true',
                                  help='Remove unseen tables from the evaluation.')

    make_gs_argparser = subparsers.add_parser("make_gs", help='Automatic script for getting a new GS from all the tables.')
    make_gs_argparser.set_defaults(action='make_gs')
    make_gs_argparser.add_argument('--output_folder', type=str, default='./gs',
                                help='Path to output folder for gold standard tables. DEFAULT: ./gs')
    make_gs_argparser.add_argument('--endpoint', type=str, default=SPARQL_ENDPOINT,
                                help=f'SPARQL endpoint. DEFAULT: {SPARQL_ENDPOINT}')

    args = argparser.parse_args()

    logger = logging.getLogger('em_gs')
    logger.setLevel(logging.INFO)

    if "action" in args:
        if args.action == 'wiki':
            wiki_to_gs(args.input_folder, args.output_folder, args.endpoint)
        elif args.action == 'web':
            web_to_gs(args.input_folder, args.output_folder)
        elif args.action == 'dbp':
            sparql_to_gs(args.input_folder, args.output_folder, args.endpoint)
        elif args.action == 't2d':
            t2d_to_gs(args.input_folder, args.output_folder, args.endpoint)
        elif args.action == 'check':
            _check_uris(args.input_folder)
        elif args.action == 'stats':
            gt_stats(args.input_folder)
        elif args.action == 'to_cea':
            to_cea_format(args.input_folder, args.output_tables_folder, args.output_gs_folder, args.endpoint,
                          args.sameas_file)
        elif args.action == 'to_mantis':
            to_mantis_format(args.input_folder, args.tables_folder, args.tables_list_file)
        elif args.action == 'to_idlab':
            to_idlab_format(args.gs_file, args.output_folder)
        elif args.action == 'score':
            print(json.dumps(
                compute_score(args.gs_file, args.annotations_file, args.tables_folder, args.wrong_cells_file,
                              args.remove_unseen), indent=4))
        elif args.action == 'make_gs':
            make_gs(args.output_folder, args.endpoint)

    else:
        argparser.print_help()
