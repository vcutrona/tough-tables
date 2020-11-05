import json
import os
from tabular_semantics.kg.endpoints import WikidataEndpoint
import pandas as pd


def extend_cta(input_gt_file, gt_ancestor_file, gt_descendent_file, extension_type):
    gt = pd.read_csv(input_gt_file, names=['table', 'col', 'classes'])
    gt['classes'] = gt['classes'].apply(str.split)
    gt = gt.explode('classes')
    gts = gt['classes'].unique().tolist()

    ep = WikidataEndpoint()

    if extension_type == 'ancestor' or extension_type == 'both':
        gt_sup2d = json.load(open(gt_ancestor_file)) if os.path.exists(gt_ancestor_file) else dict()
        for i, gt in enumerate(gts):
            if gt not in gt_sup2d:
                sup2dist = ep.getDistanceToAllSuperClasses(gt)
                sup_d = dict()
                for sup in sup2dist:
                    d = list(sup2dist[sup])[0]
                    sup_d[sup] = d
                gt_sup2d[gt] = sup_d
                if i > 0 and i % 20 == 0:
                    print('%d done' % i)
                    json.dump(gt_sup2d, open(gt_ancestor_file, 'w'), indent=2)

        json.dump(gt_sup2d, open(gt_ancestor_file, 'w'), indent=2)

    if extension_type == 'descendent' or extension_type == 'both':
        gt_sub2d = json.load(open(gt_descendent_file)) if os.path.exists(gt_descendent_file) else dict()
        for i, gt in enumerate(gts):
            if gt not in gt_sub2d:
                sub2dist = ep.getDistanceToAllSubClasses(uri_class=gt, max_level=3)
                sub_d = dict()
                for sub in sub2dist:
                    d = list(sub2dist[sub])[0]
                    sub_d[sub] = d
                gt_sub2d[gt] = sub_d
                if i > 0 and i % 20 == 0:
                    print('%d done' % i)
                    json.dump(gt_sub2d, open(gt_descendent_file, 'w'), indent=2)

        json.dump(gt_sub2d, open(gt_descendent_file, 'w'), indent=2)
