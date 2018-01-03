# -*- coding: utf-8 -*-
# Author: Xiaoming Qin

""" Get word-level features from meta data. """

import json
import re
import string

pattern = '[-_+=.,;/:#&$@*?!\n]'


def load_stop_words(stops_file):
    """ Load custom stop words. """

    with open(stops_file, 'rb') as f:
        stop_words = f.readlines()
    stop_words = set([s.strip() for s in stop_words])

    return stop_words


def create_query_label_map(data_root="/home/simon/webvision/data"):
    from os.path import join as pjoin
    with open(pjoin(data_root, 'label_map.txt'), 'rb') as f:
        wv_40_lbl = f.readlines()

    id_name_map = {}
    for item in wv_40_lbl:
        idx = item.strip().split(" ")[1]
        id_name_map[int(idx)] = " ".join(item.strip().split(" ")[2:])

    wv_40_lbl = set([int(l.strip().split()[1]) + 1 for l in wv_40_lbl])

    with open(pjoin(data_root, 'queries_synsets_map.txt'), 'rb') as f:
        ql_map = f.readlines()

    tiny_40_map = []
    for line in ql_map:
        query, label = line.strip().split()
        query = int(query)
        label = int(label)
        if label in wv_40_lbl:
            j_query = "q%04d" % query
            tiny_40_map.append("{} {} {}\n".format(
                j_query, label - 1, id_name_map[label - 1]))

    with open(pjoin(data_root, 'tiny_qln_map.txt'), 'wb') as f:
        f.writelines(tiny_40_map)


def get_feat_flickr_meta(json_file, pattern, stop_words,
                         class_name=set(['tench'])):
    """ get word level features from google metadata."""

    with open(json_file, 'rb') as f:
        j_str = json.load(f)

    id_list = []
    tags_list = []
    infos = []

    for i in xrange(len(j_str)):
        if j_str[i].has_key('tags'):
            tags = j_str[i]['tags']
            if len(tags):
                wd_collector = []
                for tg in tags:
                    # replace `.,;` with ` `
                    tg = re.sub(pattern, ' ', tg.encode('utf-8'))
                    # remove other punctuations
                    tg = tg.translate(None, string.punctuation)
                    # split into word
                    wd_collector.extend(tg.lower().split())

                # remove `class_name` and digits
                wd_collector = [w for w in wd_collector if not (
                    w in class_name or w.isdigit() or w in stop_words)]
                # get unique words
                wd_collector = list(set(wd_collector))
                tags_list.append(wd_collector)
            else:
                tags_list.append([])
        else:
            tags_list.append([])

    # Supplement for empty tags
    for i in xrange(len(j_str)):
        if len(tags_list[i]) == 0:
            if j_str[i].has_key('description'):
                dspt = j_str[i]['description']

                # replace `.,;` with ` `
                dspt = re.sub(pattern, ' ', dspt.encode('utf-8'))
                # remove other punctuations
                dspt = dspt.translate(None, string.punctuation)
                # split into word
                dspt = dspt.lower().split()
                # remove `class_name` and digits
                dspt = [w for w in dspt if not (
                    w in class_name or w.isdigit() or w in stop_words)]
                # get unique words
                tags_list[i] = list(set(dspt))

        infos.append({'id': j_str[i]['id'].encode('utf-8'),
                      'tags': tags_list[i]})

    return infos


def get_feat_google_meta(json_file, pattern, stop_words,
                         class_name=set(['tench'])):
    """ get word level features from google metadata."""

    with open(json_file, 'rb') as f:
        j_str = json.load(f)

    infos = []

    for i in xrange(len(j_str)):

        if j_str[i].has_key('description'):
            dspt = j_str[i]['description'].encode('utf-8')
        else:
            dspt = ""
        if j_str[i].has_key('site'):
            site = j_str[i]['site'].encode('utf-8')
        else:
            site = ""
        if j_str[i].has_key('title'):
            title = j_str[i]['title'].encode('utf-8')
        else:
            title = ""

        if j_str[i].has_key('page_url'):
            page_url = j_str[i]['page_url'].encode('utf-8')
        else:
            page_url = ""

        if len(dspt):
            # replace `.,;` with ` `
            dspt = re.sub(pattern, ' ', dspt)
            # remove other punctuations
            dspt = dspt.translate(None, string.punctuation)
            # split into word
            dspt = dspt.lower().split()
            # remove `class_name` and digits
            dspt = [w for w in dspt if not (
                w in class_name or w.isdigit() or w in stop_words)]
        else:
            dspt = []

        if len(site):
            # replace `.,;` with ` `
            site = re.sub(pattern, ' ', site)
            # remove other punctuations
            site = site.translate(None, string.punctuation)
            # split into word
            site = site.lower().split()
            # remove `class_name` and digits
            site = [w for w in site if not (
                w in class_name or w.isdigit() or w in stop_words)]
        else:
            site = []

        if len(title):
            # replace `.,;` with ` `
            title = re.sub(pattern, ' ', title)
            # remove other punctuations
            title = title.translate(None, string.punctuation)
            # split into word
            title = title.lower().split()
            # remove `class_name` and digits
            title = [w for w in title if not (
                w in class_name or w.isdigit() or w in stop_words)]
        else:
            title = []

        if len(page_url):
            # replace `.,;` with ` `
            page_url = re.sub(pattern, ' ', page_url)
            # remove other punctuations
            page_url = page_url.translate(None, string.punctuation)
            # split into word
            page_url = page_url.lower().split()
            # remove `class_name` and digits
            page_url = [w for w in page_url if not (
                w in class_name or w.isdigit() or w in stop_words)]
        else:
            page_url = []

        infos.append({'id': j_str[i]['id'].encode('utf-8'),
                      'tags': list(set(dspt + site + title + page_url))})

    return infos


def extract_word_feature(qln_map_file, stops_file,
                         src_data_root, dst_data_root):
    """ Extract word level feature. 

    Args: 
        qln_map_file: query-label-name map file

    """
    import os
    from os.path import join as pjoin

    with open(qln_map_file, 'rb') as f:
        map_list = f.readlines()

    pattern = '[-_+=.,;/:#&$@*?!\n]'
    stop_words = load_stop_words(stops_file)

    for item in map_list:
        item = re.sub('[,\n]', '', item)
        fname = item.split(' ')[0] + '.json'
        cls_names = item.strip().split()[2:]
        cls_names = set([c.lower() for c in cls_names])
        gg_file = pjoin(src_data_root, 'google', fname)
        if os.path.isfile(gg_file):
            infos = get_feat_google_meta(
                gg_file, pattern, stop_words, cls_names)
            with open(pjoin(dst_data_root, 'google',
                            fname), 'w') as outfile:
                json.dump(infos, outfile, indent=4)

        fl_file = pjoin(src_data_root, 'flickr', fname)
        if os.path.isfile(fl_file):
            infos = get_feat_flickr_meta(
                fl_file, pattern, stop_words, cls_names)
            with open(pjoin(dst_data_root, 'flickr',
                            fname), 'w') as outfile:
                json.dump(infos, outfile, indent=4)


def main():
    src_data_root = "/data/webvision/metadata"
    dst_data_root = "/home/simon/webvision/data"
    stops_file = "/home/simon/webvision/data/stopwords.txt"
    qln_map_file = "/home/simon/webvision/data/tiny_qln_map.txt"

    extract_word_feature(qln_map_file, stops_file,
                         src_data_root, dst_data_root)


if __name__ == "__main__":
    main()
