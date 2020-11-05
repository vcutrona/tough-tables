import json
import os
import pickle
import random
import string

import numpy as np
import pandas as pd
from pronounceable import PronounceableWord
from pronounceable.components import INITIAL_CONSONANTS, FINAL_CONSONANTS, double_vowels
from pronounceable.digraph import DIGRAPHS_FREQUENCY

from tabular_semantics.kg.endpoints import WikidataEndpoint
from tabular_semantics.kg.entity import URI_KG
from tabular_semantics.sem_tab.CTA_Wikidata_Extend import extend_cta
from tough_tables import _write_df


class FakeKnowledgeGapGenerator(PronounceableWord):
    """
    Override some methods to avoid using the "secret" module (reproducibility issues)
    """

    _INITIAL_CONSONANTS = sorted(INITIAL_CONSONANTS)  # IMPORTANT! Otherwise the order changes every time!
    _FINAL_CONSONANTS = sorted(FINAL_CONSONANTS)

    def length(self, min_length, max_length):
        min_length = min_length or self.min_length
        max_length = max_length or self.max_length

        if not min_length and not max_length:
            return

        # When munging characters, we need to know where to start counting
        # letters from
        length = min_length + random.randrange(max_length - min_length)
        char = self._generate_nextchar(self.total_sum, self.start_freq)
        a = ord('a')
        word = chr(char + a)

        for i in range(1, length):
            char = self._generate_nextchar(self.row_sums[char],
                                           DIGRAPHS_FREQUENCY[char])
            word += chr(char + a)
        return word

    def _generate_nextchar(self, all, freq):
        i, pos = 0, random.randrange(all)

        while i < (len(freq) - 1) and pos >= freq[i]:
            pos -= freq[i]
            i += 1

        return i

    def _generate_word(self):
        return random.choice(self._INITIAL_CONSONANTS) \
               + random.choice(random.choice(['aeiouy', list(double_vowels())])) \
               + random.choice(['', random.choice(self._FINAL_CONSONANTS)])

    def get_fake_knowledge_gap(self):
        words = [self._generate_word()]
        for _ in range(random.randint(1, 3)):
            word = random.choice([self.length(random.randint(1, 3),
                                              random.randint(7, 10)),
                                  self._generate_word()])
            rnd = random.random()
            if rnd < 0.85 or '-' in words[0] or ',' in words[0]:
                words.insert(0, word)
            elif rnd < 0.95:
                words[0] = "-".join([word, words[0]])
            else:
                words[0] = ", ".join([word, words[0]])
        words[0] = words[0].capitalize()
        return " ".join(words)


def get_conversion_map():
    ep = WikidataEndpoint()
    conv_map_file = './resources/db_wd_conversion_map.pickle'

    # merge contains the list of all the entities in 2T_dbpedia
    merge = pd.DataFrame(json.load(open('./resources/dbp_sameas.json', 'rb')).keys(), columns=["redirect"]).merge(
        pd.read_csv('./resources/transitive_redirects_en.ttl',
                    sep=" ",
                    names=['redirect', 'prop', 'target', 'point'],
                    usecols=['redirect', 'target'])[1:-1].applymap(lambda x: x[1:-1]),
        how='left')
    # replace the target with the redirect, if empty (i.e., the entity in redirect is the target)
    merge['target'] = merge['target'].mask(pd.isnull, merge['redirect'])

    if not os.path.exists('./resources/interlanguage_links_en.pkl'):
        sameas_file = './resources/interlanguage_links_en.ttl'
        wd_sameas = pd.read_csv(sameas_file,
                                sep=' ',
                                names=['dbp', 'prop', 'wd', 'point'],
                                usecols=['dbp', 'wd']) \
            .applymap(lambda x: x[1:-1]) \
            .set_index('dbp')['wd']
        pickle.dump(wd_sameas.to_dict(), open('./resources/interlanguage_links_en.pkl', 'wb'))

    if not os.path.exists('./resources/wd-sameas-all-wikis.pkl'):
        sameas_file = './resources/wd-sameas-all-wikis.ttl'
        wd_sameas = pd.read_csv(sameas_file,
                                sep=' ',
                                names=['wd', 'prop', 'dbp', 'point'],
                                usecols=['wd', 'dbp']) \
            .applymap(lambda x: x[1:-1]) \
            .set_index('dbp')['wd'] \
            .apply(lambda x: x.replace("http://wikidata.dbpedia.org/resource/",
                                       "http://www.wikidata.org/entity/"))
        pickle.dump(wd_sameas.to_dict(), open('./resources/wd-sameas-all-wikis.pkl', 'wb'))

    wd_sameas = pickle.load(open('./resources/wd-sameas-all-wikis.pkl', 'rb'))  # wd to dbpedia
    wd_sameas.update(pickle.load(open('./resources/interlanguage_links_en.pkl', 'rb')))  # dbpedia to wd
    wd_sameas.update({  # missing
        'http://dbpedia.org/resource/Centennial_Bank': 'http://www.wikidata.org/entity/Q92384230',
        'http://dbpedia.org/resource/Lake_Dot_(Orlando_lake)': 'http://www.wikidata.org/entity/Q28452068',
        'http://dbpedia.org/resource/Kyle_York_(American_football)': 'http://www.wikidata.org/entity/Q27920029',
        'http://dbpedia.org/resource/Marie_Adams_(singer)': 'http://www.wikidata.org/entity/Q1897235',
        'http://dbpedia.org/resource/The_Three_Tons_of_Joy': 'http://www.wikidata.org/entity/Q1897235',
        'http://dbpedia.org/resource/Ollie_Marie_Adams': 'http://www.wikidata.org/entity/Q1897235'
    })

    conv_map = {}
    if os.path.exists(conv_map_file):
        conv_map = pickle.load(open(conv_map_file, 'rb'))
    d = [(db, db_t) for db, db_t in dict(merge.to_dict('sp')['data']).items() if db not in conv_map]

    for i in range(0, len(d), 100):
        wd_sameas_chunk = {db: wd_sameas[db_t] for (db, db_t) in d[i:i + 100] if db_t in wd_sameas}
        redirects_chunk = ep.getRedirectEntitiesMulti(list(wd_sameas_chunk.values()))
        sameas_chunk = ep.getSameEntitiesMulti(list(wd_sameas_chunk.values()))
        for db, wd in wd_sameas_chunk.items():
            entry = [wd_sameas_chunk[db], set(), set()]
            if wd in redirects_chunk:
                entry[1] = redirects_chunk[wd]
            if wd in sameas_chunk:
                entry[2] = sameas_chunk[wd]
            conv_map[db] = entry
        print(f'Chunk {i / 100}/{len(d) / 100})')
        pickle.dump(conv_map, open(conv_map_file, 'wb'))

    return conv_map


def db_to_wd(input_dir, output_gs_dir):
    conv_map = get_conversion_map()
    # concatenate the values in list
    # keep first the db2wd sameas links, then owl:sameAs (redirect), then P340 (like sameas)
    conv_map = {db: " ".join([wd_list[0]] +
                             list(sorted(wd_list[1] | wd_list[2] - {wd_list[0]})))
                for db, wd_list in conv_map.items()}

    with os.scandir(input_dir) as it:
        for entry in it:
            if os.path.isfile(f'{output_gs_dir}/{entry.name}'):
                print(f'Skipping file: {entry.path} - {output_gs_dir}/{entry.name} already exists.')
            elif entry.name.endswith(".csv") and entry.is_file():
                print(f'Processing file: {entry.path}')
                df = pd.read_csv(entry.path, dtype=object)
                target_cols = [col for col in df.columns if '__URI' in col]
                df = df.replace(dict(zip(target_cols, [conv_map] * len(target_cols))))
                for col in target_cols:
                    print(f'Checking col: {col}')
                    drop_na = df[col].dropna()
                    assert len(drop_na[drop_na.str.contains('http://dbpedia')]) == 0
                _write_df(df, f"{output_gs_dir}/{entry.name}")


def get_fake_tab_id(length=8, dictionary=string.ascii_uppercase + string.digits):
    return ''.join(random.choices(dictionary, k=length))


def to_cea_format(input_dir, output_tables_dir, output_gs_dir, output_target_dir):
    annotations = []
    anonymous_dict = {}
    random.seed(99)
    kgg = FakeKnowledgeGapGenerator()

    with os.scandir(input_dir) as it:
        for entry in it:
            if entry.name.endswith(".csv") and entry.is_file():
                original_tab_id = entry.name[:-4]
                tab_id = get_fake_tab_id()
                while tab_id in anonymous_dict:
                    tab_id = get_fake_tab_id()

                anonymous_dict[tab_id] = original_tab_id

                print(f'Processing file: {entry.path}')
                df = pd.read_csv(entry.path, dtype=object)

                count = 0
                for col_id, (columnName, columnData) in enumerate(df.iteritems()):
                    new_cols = [col for col in df.columns if '__URI' not in col]
                    if '__URI' in columnName:
                        count = count + 1
                        for row_id, value in columnData.iteritems():
                            if value is not np.nan:
                                # row_id + 1 due to the header row
                                ann = {'tab_id': tab_id,
                                       'col_id': str(new_cols.index(columnName.replace("__URI", ""))),
                                       'row_id': str(row_id + 1),
                                       'entity': value}
                                annotations.append(ann)
                            else:  # fake knowledge gap
                                label = df[columnName.replace("__URI", "")][row_id]
                                if label and label is not np.nan:
                                    df.at[row_id, columnName.replace("__URI", "")] = kgg.get_fake_knowledge_gap()
                                    ann = {'tab_id': tab_id,
                                           'col_id': str(new_cols.index(columnName.replace("__URI", ""))),
                                           'row_id': str(row_id + 1),
                                           'entity': 'NIL'}
                                    annotations.append(ann)
                df = df[new_cols]
                rename_columns_map = {col: f"col{idx}" for idx, col in enumerate(df.columns)}
                df = df.rename(columns=rename_columns_map)
                _write_df(df, f'{output_tables_dir}/{tab_id}.csv')

    _write_df(pd.DataFrame(annotations)[['tab_id', 'row_id', 'col_id', 'entity']],
              f'{output_gs_dir}/CEA_2T_WD_gt.csv',
              header=False)
    _write_df(pd.DataFrame(annotations)[['tab_id', 'row_id', 'col_id']],
              f'{output_target_dir}/CEA_2T_WD_Targets.csv',
              header=False)
    json.dump(anonymous_dict, open(f'{output_gs_dir}/filename_map.json', 'w'), indent=2)


def _prefetch_data(entities):
    ep = WikidataEndpoint()

    entities_types_file = './resources/wd_entities_types.pickle'
    types_supertypes_file = './resources/wd_types_supertypes.pickle'
    types_sameas_file = './resources/wd_types_sameas.pickle'

    # STEP 1: get all types for all the entities
    entities_types = {}
    if os.path.exists(entities_types_file):
        entities_types = pickle.load(open(entities_types_file, 'rb'))
        entities = [e for e in entities if e not in entities_types]

    chunk_size = 100
    for i in range(0, len(entities), chunk_size):
        types = ep.getAllTypesForEntityMulti(entities[i:i + chunk_size])

        classes = [entity for entity in entities[i:i + chunk_size] if entity not in types]  # entities with no types
        if classes:
            superclasses = ep.getAllSuperClassesMulti(classes)
            types.update(superclasses)  # replace classes with types, when available

        no_type_no_classes = {entity: {} for entity in entities[i:i + chunk_size] if entity not in types}  # redirects
        if no_type_no_classes:
            types.update(no_type_no_classes)

        entities_types.update(types)

        print(f'ENT-TYP: Chunk {i / chunk_size}/{len(entities) / chunk_size}')
        pickle.dump(entities_types, open(entities_types_file, 'wb'))

    # STEP 2: get all supertypes for all the types
    types = list(set().union(*pickle.load(open(entities_types_file, 'rb')).values()))
    types_supertypes = {}
    if os.path.exists(types_supertypes_file):
        types_supertypes = pickle.load(open(types_supertypes_file, 'rb'))
        types = [t for t in types if t not in types_supertypes]

    for i in range(0, len(types), chunk_size):
        types_supertypes.update(ep.getAllSuperClassesMulti(types[i:i + chunk_size]))
        no_supertypes = {type_: set() for type_ in types[i:i + chunk_size] if type_ not in types_supertypes}
        if no_supertypes:
            types_supertypes.update(no_supertypes)
        print(f'TYP_STYP: Chunk {i / chunk_size}/{len(types) / chunk_size}')
        pickle.dump(types_supertypes, open(types_supertypes_file, 'wb'))

    # STEP 3: get sameas/redirected types
    types = list(set().union(*pickle.load(open(entities_types_file, 'rb')).values())
                 | set().union(*pickle.load(open(types_supertypes_file, 'rb')).values()))
    types_sameas = {}
    if os.path.exists(types_sameas_file):
        types_sameas = pickle.load(open(types_sameas_file, 'rb'))
        types = [t for t in types if t not in types_sameas]
    for i in range(0, len(types), chunk_size):
        for t in types[i:i + chunk_size]:
            types_sameas[t] = {'sameas': set(), 'redirect': set()}
        sameas_types = ep.getSameEntitiesMulti(types[i:i + chunk_size])
        for entry in sameas_types:
            types_sameas[entry]['sameas'] = sameas_types[entry]
        redirect_types = ep.getRedirectEntitiesMulti(types[i:i + chunk_size])
        for entry in redirect_types:
            types_sameas[entry]['redirect'] = redirect_types[entry]
    pickle.dump(types_sameas, open(types_sameas_file, 'wb'))


def cta_from_cea(cea_gs_file, output_gs_dir, output_target_dir):
    entities_types_file = './resources/wd_entities_types.pickle'
    types_supertypes_file = './resources/wd_types_supertypes.pickle'
    types_sameas_file = './resources/wd_types_sameas.pickle'

    gt = pd.read_csv(cea_gs_file, dtype=object, names=['tab_id', 'row_id', 'col_id', 'entities'])
    gt['entities'] = gt['entities'].apply(lambda x: x.split())
    gt = gt.explode('entities')

    _prefetch_data(gt['entities'].unique())

    # append types to each entity
    entity_types = pickle.load(open(entities_types_file, 'rb'))
    gt = gt.merge(pd.DataFrame({'entities': list(entity_types.keys()),
                                'types': [list(x) for x in entity_types.values()]}),
                  how='left')

    gt = gt.explode('types')  # explode -> one type per entity per row
    gt = gt[~gt.types.isin(URI_KG.avoid_top_concepts)]  # filter top concepts

    # cell voting: count how many entities vote the same type and get the max
    gt = gt.groupby(['tab_id', 'row_id', 'col_id', 'types']).count().rename(columns={'entities': 'count'})
    gt = gt.join(gt.groupby(['tab_id', 'row_id', 'col_id']).max().rename(columns={'count': 'max'}))
    gt = gt[gt['count'] == gt['max']].drop(columns=['count', 'max']).reset_index()  # filter types with #votes < max

    # column voting: count types frequency and max freq in column
    gt = gt.groupby(['tab_id', 'col_id', 'types']).count().rename(columns={'row_id': 'count'})
    gt = gt.join(gt.groupby(['tab_id', 'col_id']).max().rename(columns={'count': 'max'}))
    gt = gt[gt['count'] == gt['max']].drop(columns=['count', 'max']).reset_index()  # filter types with #votes < max

    # group by tab, col and get the list of types
    gt = gt.groupby(['tab_id', 'col_id']).agg(list)

    cta = []
    types_supertypes = pickle.load(open(types_supertypes_file, 'rb'))
    types_sameas = pickle.load(open(types_sameas_file, 'rb'))

    manual_fixes = {
        ('0IR0XIUW', '2'): {'http://www.wikidata.org/entity/Q65742449'},  # Formula One race
        ('WWBIR8H6', '2'): {'http://www.wikidata.org/entity/Q65742449'},  # Formula One race
        ('3DOM5NIW', '4'): {'http://www.wikidata.org/entity/Q13417114'},  # noble family
        ('60G94POT', '4'): {'http://www.wikidata.org/entity/Q13417114'},  # noble family
        ('9Y8MPU2Q', '4'): {'http://www.wikidata.org/entity/Q13417114'},  # noble family
        ('SNUO09BH', '4'): {'http://www.wikidata.org/entity/Q13417114'},  # noble family
        ('KUGCH9I3', '2'): {'http://www.wikidata.org/entity/Q1093829'},  # city of US
        ('UEJEB27H', '2'): {'http://www.wikidata.org/entity/Q1093829'},  # city of US
        ('1MQL5T7F', '5'): {'http://www.wikidata.org/entity/Q5119'},  # capital
        ('AUU9A6KL', '5'): {'http://www.wikidata.org/entity/Q5119'},  # capital
        ('5DKX42VB', '1'): {'http://www.wikidata.org/entity/Q847017'},  # sport club
        ('BID0NRU0', '1'): {'http://www.wikidata.org/entity/Q847017'},  # sport club
        ('EV6LDIB8', '1'): {'http://www.wikidata.org/entity/Q847017'},  # sport club
        ('GINQPZQC', '1'): {'http://www.wikidata.org/entity/Q847017'},  # sport club
        ('HB00DX4L', '1'): {'http://www.wikidata.org/entity/Q847017'},  # sport club
        ('J9EJV2S3', '1'): {'http://www.wikidata.org/entity/Q847017'},  # sport club
        ('FF00TEZG', '0'): {'http://www.wikidata.org/entity/Q35657'},  # u.s. state
        ('IZF82AX9', '0'): {'http://www.wikidata.org/entity/Q35657'},  # u.s. state
        ('JZ22O0DD', '2'): {'http://www.wikidata.org/entity/Q13027888'},  # baseball team
        ('LV5N8XDB', '2'): {'http://www.wikidata.org/entity/Q13027888'},  # baseball team
        ('FVKKTA8O', '4'): {'http://www.wikidata.org/entity/Q210167'},  # videogame developer
        ('JZLRN9PL', '4'): {'http://www.wikidata.org/entity/Q210167'},  # videogame developer
        ('8R6ZM8HE', '1'): {'http://www.wikidata.org/entity/Q5119'},  # capital
        ('MSCT8MJD', '1'): {'http://www.wikidata.org/entity/Q5119'},  # capital
        ('9BGE5Y4L', '2'): {'http://www.wikidata.org/entity/Q1093829'},  # U.S. city
        ('8D8CVBT0', '2'): {'http://www.wikidata.org/entity/Q1093829'},  # U.S. city
        ('5HD27KI3', '0'): {'http://www.wikidata.org/entity/Q34918903'},  # U.S. national park
        ('U1FDHL7N', '0'): {'http://www.wikidata.org/entity/Q34918903'},  # U.S. national park
        ('QW492LGU', '2'): {'http://www.wikidata.org/entity/Q34379'},  # musical instrument
        ('7QNYGYI7', '2'): {'http://www.wikidata.org/entity/Q34379'},  # musical instrument
        ('CNQ5Z0BG', '2'): {'http://www.wikidata.org/entity/Q515'},  # city
        ('HGIUTSCG', '2'): {'http://www.wikidata.org/entity/Q515'},  # city
        ('I6BBMPNU', '2'): {'http://www.wikidata.org/entity/Q35657'},  # u.s. state
        ('S8UOQYBG', '2'): {'http://www.wikidata.org/entity/Q35657'},  # u.s. state
        ('MANO2PKR', '1'): {'http://www.wikidata.org/entity/Q5119'},  # capital
        ('E22XXKVQ', '1'): {'http://www.wikidata.org/entity/Q5119'},  # capital
        ('QNP7O8L5', '2'): {'http://www.wikidata.org/entity/Q34379'},  # musical instruments
        ('X23TMJ3R', '2'): {'http://www.wikidata.org/entity/Q34379'},  # musical instruments
        ('VKWTT7F7', '2'): {'http://www.wikidata.org/entity/Q1154710'},  # association football stadium
        ('51MYHYDF', '2'): {'http://www.wikidata.org/entity/Q1154710'},  # association football stadium
        ('BDH3WFGJ', '1'): {'http://www.wikidata.org/entity/Q7930989'},  # city/town
        ('MZ0BI8NN', '1'): {'http://www.wikidata.org/entity/Q7930989'},  # city/town
    }

    for tuple_ in gt.itertuples():
        types = set(tuple_.types)
        if tuple_.Index in manual_fixes:
            types = manual_fixes[tuple_.Index]

        to_remove = set()
        for type_ in types:  # discard types that are supertypes of other types in list
            for supertype in types_supertypes[type_]:
                if supertype in types:  # type_ subclassof supertype
                    to_remove.add(supertype)
        types -= to_remove  # remove all the supertypes from the set -> keep only minimal types

        if not types:
            raise Exception("Whoops! No valid type here (" + tuple_ + ")")

        # If |candidates| > 1,  find which is the lowest common ancestor (if any)
        # e.g., Q852446  (administrative territorial entity of the United States )
        #       Q1799794 (administrative territorial entity of a specific level)
        #   ->  Q56061 (administrative territorial entity)
        if len(types) > 1:
            supertypes = [types_supertypes[type_] for type_ in types]
            common_supertypes = set.intersection(*supertypes).difference(URI_KG.avoid_top_concepts)
            if common_supertypes:
                types = common_supertypes

        sameas = set().union(*[types_sameas[t]['sameas'] for t in types]).difference(URI_KG.avoid_top_concepts)
        redirect = set().union(*[types_sameas[t]['redirect'] for t in types]).difference(URI_KG.avoid_top_concepts)

        cta.append(tuple_.Index + (" ".join(sorted(types | sameas | redirect)),))

    cta_gt = pd.DataFrame(cta, columns=['tab_id', 'col_id', 'types'])
    _write_df(cta_gt, f'{output_gs_dir}/CTA_2T_WD_gt.csv', header=False)
    _write_df(cta_gt[['tab_id', 'col_id']], f'{output_target_dir}/CTA_2T_WD_targets.csv', header=False)

    ancestor_file = f'{output_gs_dir}/CTA_2T_WD_gt_ancestor.json'
    descendent_file = f'{output_gs_dir}/CTA_2T_WD_gt_descendent.json'
    extend_cta(f'{output_gs_dir}/CTA_2T_WD_gt.csv', ancestor_file, descendent_file, 'both')


if __name__ == '__main__':
    db_to_wd(input_dir='./gs', output_gs_dir='./gs_wd')
    to_cea_format(input_dir='./gs_wd/', output_tables_dir='./2T_WD/tables',
                  output_gs_dir='./2T_WD/gt', output_target_dir='./2T_WD/targets')
    cta_from_cea(cea_gs_file='./2T_WD/gt/CEA_2T_WD_gt.csv', output_gs_dir='./2T_WD/gt',
                 output_target_dir='./2T_WD/targets')
