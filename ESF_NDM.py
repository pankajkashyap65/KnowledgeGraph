#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install spacy
import wikipedia
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from numbers import Number
from pandas import DataFrame
import numpy as np
import statsmodels.stats.api as sms
import os, sys, codecs, argparse, pprint, time
import glob   
import os
import math
from os import walk
import numpy as np
from scipy.special import softmax
from scipy import spatial
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
import re
import subprocess
from nltk.corpus import stopwords
from collections import defaultdict
from ast import literal_eval
import pickle
import string
import time
import tagme
from collections import OrderedDict
from wikimapper import WikiMapper
import networkx as nx
from networkx.algorithms import community
import numpy as np
from wikidata.client import Client
from wikidata.entity import EntityState
import spacy
from urllib.request import urlopen
from bs4 import BeautifulSoup
tagme.GCUBE_TOKEN = "cee7e095-9914-4f70-82eb-8eb60156d81b-843339462" #place your tagme token here
# Loading WikiMapper mappings ****** Change Path Accordingly *****************
mapper = WikiMapper('Models/Wikimapper_index/index_enwiki-20190420.db')

#Loading Spacy Model
nlp = spacy.load('en_core_web_lg')

#Constants
INSTANCE_OF_ID = 'P31'
ITEM = "item"


# In[2]:
indrirunquery_command =  '/home/pankaj/indri_5.12/runquery/IndriRunQuery'
dumpindex_command = '/home/pankaj/indri-5.12/dumpindex/dumpindex'


# In[3]:

#loads transe vectors pickle file to iterate with
def load_transe_pickle(): 
    # 
    vec = np.memmap('./Transe/entity2vec.bin' , dtype='float32', mode='r')
    vec = vec.reshape((20982733,100)) #no. of entity, each embedding size is 100
    with open('entity2id.pickle', 'rb') as handle:
        entity2index = pickle.load(handle)
    return vec, entity2index


# In[4]:


vec, entity2index =  load_transe_pickle()
complete_time = 0.0

# In[5]:

#loads transe vectors pickle file to iterate with
def load_transe(): 
    vec = np.memmap('./Transe/entity2vec.bin' , dtype='float32', mode='r')
    vec = vec.reshape((20982733,100))
    entity2id = {}
    with open("./Transe/entity2id.txt") as f1:
        next(f1) #skip first line because total entity number
        for line in f1:
            row = line.replace("\n","").split("\t")
            #print(row)
            entity2id[row[0]]=int(row[1]) # row[0]= id e.g. Q76, row[1] = index_no. e.g. 5
    return vec, entity2id


# In[6]:

#retrieving a document's content by obtaining its ID using dumpindex based on its docno
def get_file(docno, index_filepath): #
    index = ""+index_filepath+""
    result = subprocess.Popen([dumpindex_command, index, 'di', 'docno', docno], 
                   stdout=subprocess.PIPE, 
                   stderr=subprocess.STDOUT)
    stdout,stderr = result.communicate()
    id = stdout.decode('utf-8').strip()
    #print("id: ", id)
    result = subprocess.Popen([dumpindex_command, index, 'dt', id], 
                   stdout=subprocess.PIPE, 
                   stderr=subprocess.STDOUT)
    stdout2,stderr2 = result.communicate()
    text = str(stdout2.replace(b'\r',b' ').decode('utf-8','ignore'))
    start_text = text.find('<TEXT>')+ 6
    end_text = text.find('</TEXT>')
    txt = text[start_text:end_text]
#    print(txt)
    return txt


# In[7]:

#calculating the Gaussian (normal) distribution for a given input x, with mean mu and standard deviation sig
def gaussian(x, mu=0, sig=1): 
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


# In[8]:


# Creates a graph in networkx, creating a similaritymatrix => sparsification => graph_construction
def create_Graph(feedback_entities_id): #entity_id:entity_vec
    G = nx.Graph()
    wgt_matrix = dict()
 #   print(len(feedback_entities_id))
    # iterating through the enumerate objects
    for i in feedback_entities_id.items(): #i,j are the index over ordereddict
        if i != len(feedback_entities_id):
            for j in feedback_entities_id.items():
                if i[0] != j[0]:
    #                entity_score = 1 - spatial.distance.cosine(qcluster_center, vector)
                    lst = sorted([i[0], j[0]])
                    wgt_matrix[lst[0] + lst[1]] = 1 - spatial.distance.cosine(i[1], j[1])
#                    print("i1 and j1: ", i[1], j[1])
    #normalizing the values
    max_value = max(wgt_matrix.values())
    min_value = min(wgt_matrix.values())
    #  print(len(wgt_matrix.values()))
    for key in wgt_matrix.keys():
        #print("max_value:{}".format(max_value))
        #print("min value:{}".format(min_value))
        wgt_matrix[key] = (wgt_matrix[key] - min_value) / (max_value - min_value)

    for i in feedback_entities_id.items():
        if i != len(feedback_entities_id):
            for j in feedback_entities_id.items():
                if i[0] != j[0]:
                    lst = sorted([i[0], j[0]])
                    wgt = wgt_matrix[lst[0] + lst[1]]
                    #sparsification
                    if wgt > 0.5: 
                        G.add_edge(i[0], j[0], weight=wgt)

    return G


# In[9]:
#calculating the Euclidean distance between a query centroid and entity vectors,
#filters entities based on a 95% confidence interval of similarity scores, and returns the filtered entity IDs while printing their titles
def gaussian_distribution(query_centroid, ent_ids, id2title):
#    print("code here")
    filter_entity = []
    similarity_score = OrderedDict()
    for ids in ent_ids:
        vec = transe_vec_for_entity(ids)
        sim = distance.euclidean(query_centroid, vec)
        similarity_score[ids] = round(sim,2)
    print(len(similarity_score))
    #95% confidence interval
    interval = sms.DescrStatsW(list(similarity_score.values())).tconfint_mean(alpha=0.05, alternative='two-sided') #95% confidence interval        
    print(interval)
    for i in similarity_score:
        if similarity_score[i] >= interval[0] and similarity_score[i] <= interval[1]:
            filter_entity.append(i)
        else:
            continue
    print(filter_entity, " and len ", len(filter_entity))
    for i in filter_entity:
        print(i, " : ", id2title.get(i))
        
    return filter_entity

# performing community identification in the entity Graph
def cluster(feedback_annotations, qcluster_centre, id2title, sig=1.0):
    selected_ent2weight = OrderedDict()
    cluster_score_list = []
    G = create_Graph(feedback_annotations)
    print("graph type: ", type(G))
    comp = community.girvan_newman(G)
    print("comp type: ", type(comp))
    #iterating over all communities
    m_idx = 0
    tmp_comp = tuple(c for c in next(comp))
    for cls in tmp_comp:
        #print(cls)
        comp_graph = G.subgraph(list(cls))
        m_idx = m_idx + 1 # index of the cluster
        print("length of cluster: ", len(cls))
        print("m_idx: ", m_idx)
#        cls = next_community[m_idx] #cls =cluster
        print("cls type: ", type(cls))
        cls_coefficient = nx.average_clustering(comp_graph)
        print("Cls__coefficient: ", cls_coefficient)

        cls_vec = None
        for word in cls:
            if cls_vec is not None:
                cls_vec = np.add(cls_vec, transe_vec_for_entity(word))
            else:
                cls_vec = transe_vec_for_entity(word)
        cls_vec = cls_vec / len(cls) #centroid of cluster
        query_relatedness = gaussian(distance.euclidean(cls_vec, qcluster_centre), sig=sig)
        #calculating cluster_score
        cls_score = cls_coefficient * query_relatedness
        print("Cluster score for cluster: ", m_idx, " is: ", cls_score)
        cluster_score_list.append(cls_score)
    softmax_prob = softmax(cluster_score_list)
    print("softmax_prob: ", softmax_prob)
    print("sum softmax: ", softmax_prob.sum())
    mean = softmax_prob.sum()/m_idx #taking the clusters having softmax higher than mean
    print("mean value: ", mean)
    softmax_list = softmax_prob.tolist()
    
    ent_ids = []
    for idx, val in enumerate(softmax_list):
        if val > mean:
            print("cluster number: ", idx, " Selected")
            cls = tmp_comp[idx]
            for word in cls:
                ent_ids.append(word)
                print("ent_id: ", word, " : ", id2title.get(word))
    #print(ent_ids)
    
    ent1 = gaussian_distribution(qcluster_centre, ent_ids, id2title)
    ent = []
    for word in ent1:
        ent.append(word)
    for word in ent:
        #calculating distance between centroid and vector
        ent_vec = transe_vec_for_entity(word)
        weight = 0.0
        query_gauss_similarity = gaussian(distance.euclidean(ent_vec, qcluster_centre), sig=sig)
        weight = query_gauss_similarity
        table = str.maketrans(dict.fromkeys(string.punctuation))
        #if weight > 0.5:
        title = id2title.get(word)
        title = title.translate(table) #removing punctuations
        title = title.replace('(', '').replace(')', '').replace(',',' ').replace('?',' ').replace('/',' ').replace('-',' ')                .replace('\\',' ').replace('!',' ').replace('_',' ').replace('=',' ')        .replace('\'','').replace('&',' ').replace(':', ' ').replace('+',' ').replace('.', ' ').replace('  ', ' ').replace('\'', ' ').replace('\"', ' ').replace(';', ' ')

        selected_ent2weight[title] = str(round(weight, 2))

    return selected_ent2weight


# In[10]:


#Retrieving feedback documents for a Single query
def get_top_k_doc(query_number, query_text, index_filepath, k): # query = [query_number,query_string] , k = number of documents to retrieve
    xml_string = "<parameters>\n"
    print("Processing query number: " + str(query_number), " and query: ", str(query_text))

    # Removing punctuations as Indri doesnt support these characters
    table = str.maketrans(dict.fromkeys(string.punctuation))
    term = query_text.translate(table) #removing punctuations
    terms = term.split(' ')
    while ("" in terms):
        terms.remove("")

    # Forming Query
    xml_string += "    <query>\n        <number>" + str(query_number) + "</number>\n        <text>#combine("
       
    for term in terms:
        xml_string += " " + str(term)
    xml_string += " )</text>\n    </query>\n"
    
    '''*******  Change Path to Indri index Accordingly ********************'''
    xml_string += "    <index>"+index_filepath+"</index>\n    <runID>runName</runID>\n    <trecFormat>true</trecFormat>\n    <count>" + str(k) + "</count>\n</parameters>"

    filename = "query/query_temp.xml"
    xml_file = open(filename, "w", encoding='utf8')
    n = xml_file.write(xml_string)
    xml_file.close()

    ''' Running INDRIRUNQUERY through cmd
     ****************** Might be different for Linux ***********************************'''
    result = subprocess.Popen([indrirunquery_command, filename],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
    stdout, stderr = result.communicate()
    s = '\n'
    lst = stdout.split()
    lst_len = len(lst)
    output = []
    i = 0

    # Forming structured result from raw data
    # Might be different for other datasets
    while True:
        idx = 6 * i + 2
        if (idx >= lst_len):
            break
        output.append(lst[idx].decode('ISO-8859-1'))
#        print(lst[idx].decode('ISO-8859-1'))
        i += 1
    return output


# In[11]:

#fetching the vector representation for a given entity_id
def transe_vec_for_entity(entity_id): #here entity_id is e.g. Q76, fun. provides the vector for the entity
    index = entity2index.get(entity_id, -1)
#    print("code in transe")
    if index==-1:
        entity_vec = vec[entity2index.get("Q543287")] #vector for Null
        #print(entity_id," from null :", entity_vec)
    else:
        entity_vec = vec[index]
    return entity_vec


# In[12]:
#converting a Wikipedia title into its corresponding Wikidata entity ID,
def get_wikiData_title2id(title):
    title = str(title).replace(' ','_')
    try:
        wiki_id = mapper.title_to_id(title)
#        print("entity title: ", title, " id ", wiki_id)
        return wiki_id # returns str type wiki_id
    except:
        print('Error in title: {}'.format(title))
        return -1


# In[13]:

#performing clustering on vectors using k-means
def kmeans(arr, clusters): #clustering on vectors 
    if(len(arr) < clusters):
        kmeans = KMeans(n_clusters=len(arr), n_init= 10, init= 'k-means++', random_state= 5)
    else:
        kmeans = KMeans(n_clusters=clusters, n_init= 10, init= 'k-means++', random_state= 5)
    kmeans.fit(arr)
#    print(kmeans.cluster_centers_)
#    print(kmeans.labels_)
    return kmeans.labels_, kmeans.cluster_centers_


# In[14]:

#will generate file having entity_ids and their vectors
def entity_list_file(entity_list): 
    arr =[]
    entity = entity_list.split(" ")
    print("Code running")
    for i in entity:
        ent_vec = transe_vec_for_entity(i)
        arr.append(ent_vec)
    return np.array(arr), entity    


# In[15]:

#getting context terms (specific named entities) for the given text 
def get_context_terms(text:str):
    count = 0
    size = 80000
    context = OrderedDict()
    id2title = OrderedDict()
    words = text.split(" ")
#    print(len(words))
    size = 10000
    for a in range(0, len(words), size):
        tmp = " ".join(words[a:a+size])
        doc = nlp(tmp)
        not_req = ["DATE","ORDINAL", "CARDINAL", "WORK_OF_ART", "NORP", "GPE", "PERSON", "ORG", "FAC"]
        for ent in doc.ents:
            if ent.label_ not in not_req:
                wiki_id = get_wikiData_title2id(ent)
                if wiki_id != None:
                    ent_vec = transe_vec_for_entity(wiki_id)
#                    print("context term: ", ent.label_, " wiki_id: ", ent_vec)
                    context[str(wiki_id)] = ent_vec
                    id2title[str(wiki_id)] = str(ent).strip()
#    print("complete context terms: ", context)
    
    return context, id2title


# In[16]:
#finding the named entities present in a query
def query_annotate(text):
    annotate = tagme.annotate(text)
    arr =[]
    annotations = OrderedDict()
    id2title = OrderedDict()
    try:
        for ent in annotate.get_annotations():
            title = ent.entity_title
            title = title.encode('ascii', 'ignore')
            title = title.decode('ascii')
            wiki_id = get_wikiData_title2id(title) #getting ids for wikidata entity names
            print("query entity: ", ent.entity_title, " wiki_id: ", wiki_id)
            ent_vec = transe_vec_for_entity(wiki_id)
#            print("entity id: ", wiki_id, " vec: ", ent_vec)
            arr.append(ent_vec)
            annotations[str(wiki_id)] = ent_vec
            id2title[str(wiki_id)] = str(title).strip()
    except:
        print('Error in TagMe document annotating')
        pass
    qcluster_label, qcluster_centers = kmeans(arr, 1) # 1 is no. of cluster
    return annotations, id2title, qcluster_centers


# In[17]:

#performing a search on Wikipedia for a given query, retrieving links from the search result page, and attempting to map these links to their Wikidata entity IDs. 
def wikipedia_search(query, selected_ent2weight, qcluster_center):
    dic_ent2vec = OrderedDict() #entity_title:vector e.g Q76: 0.2 0.4 ... 0.9
    title = wikipedia.search(query, results=1)
    links = wikipedia.page(title).links
    wiki_title = "".join(title)
    links.append(wiki_title)
    for link in links:
        link = str(link).replace(' ','_') #converting as per wikipedia input format
        try:
            wiki_id = mapper.title_to_id(link)
#            print("entity title: ", link, " id ", wiki_id)
            link = str(link).replace('_',' ') #converting as per ESF input format
#            dic_ent2vec[link] = transe_vec_for_entity(wiki_id)
    #        return wiki_id # returns str type wiki_id
        except:
            print('Error in title: {}'.format(link))
    #        return -1
    for key, value in dic_ent2vec.items():
        entity_score = 1 - spatial.distance.cosine(qcluster_center, value)
        if entity_score > 0.40:
            selected_ent2weight[key] = entity_score
#        print(key)
    return selected_ent2weight


# In[18]:

#will provide aliases for the fetched entity
def get_aliases_entities(id:str): 
    client = Client()
    entity = client.get(id, load=True)
    print("code here")
    if entity.state != EntityState.loaded :
        return None
    entity_aliases = entity.data.get('aliases')
    property_keys = entity_aliases.get('en')
    aliases_entities = []
    for property in property_keys:
        property_instance = property.get('value')
#        print("aliases for the entity: ", property_instance)
        name = property_instance
        aliases_entities.append(name)
    return aliases_entities


# In[19]:

#searching Wikipedia for pages related to a given query and processing the result to extract links to other Wikipedia pages.
def wikipedia_search_only(query):
    
    dic_id2vec = OrderedDict() #entity_id:vector e.g Q76: 0.2 0.4 ... 0.9
    dic_id2title = OrderedDict() #entity_id:title e.g Q76:barack obama
    link_title_chek = ""
    results = wikipedia.search(query, results=1)
    print("required results: ", results)
    for title in results:
        print("wiki page: ", title)
        title = title.strip()
        title = str(title).replace(' ','_')
        try:
            wiki_page = "https://en.wikipedia.org/wiki/" + title
            html = urlopen(wiki_page)
            bsObj = BeautifulSoup(html, features = "lxml") #this line will give warning
            lst = ["wikipedia", "Wikipedia", "wikidata", "File", "Category", "Help", "#", "Timeline", "List", "wikisource", "wikimedia", "%", "\\", "!", "Special", ":", "ISBN", "disambiguation", "-" ]
            unique_wiki_links = set()
            for link in bsObj.findAll("a"):
                if 'href' in link.attrs:
                    temp = link.attrs['href']
                    if "/wiki/" in temp and all((ar not in temp and ar.lower() not in temp) for ar in lst): 
                 #       print(link.attrs['href'])
                        unique_wiki_links.add(temp)
            #adding the main page, bcz wiki page is itself a link
            for link_title in unique_wiki_links:
                link_title = str(link_title[6:]).replace(' ','_') #removing /wiki/
                link_title_chek = link_title
                wiki_id = get_wikiData_title2id(link_title)
                link_title = str(link_title).replace('_',' ') #converting as per ESF input format
                dic_id2vec[str(wiki_id)] = transe_vec_for_entity(wiki_id)
                dic_id2title[str(wiki_id)] = link_title
        except:
            print("no related wikidata page found for: ", link_title_chek )
   
    return dic_id2vec, dic_id2title


# In[20]:
#retrieving and processing documents related to a given query to extract and annotate entities. 
def prf_annotate(query_number, query_text, index_filepath):
    docs = get_top_k_doc(query_number, query_text, index_filepath, 5) #retreiving top feedback docs
    prf_text = ""
    prf_annotations = OrderedDict() #wiki_id:vec e.g. Q76: 0.2 0.4 .. 0.9
    prf_id2title = OrderedDict() #wiki_id:title e.g. Q76: barack obama
    files_text = []
    for file in docs:
        text = [get_file(file, index_filepath)]   
        files_text += text
    prf_text = " ".join(files_text)
    table = str.maketrans(dict.fromkeys(string.punctuation))
    prf_text = prf_text.translate(table) #removing punctuations
    prf_text = prf_text.replace("\n", " ")
    prf_text = re.sub('\s{2,}', ' ', prf_text)
      
#    print("file texts for docs is: ", prf_text)
#    fetching entity annotations  from prf text
    words = prf_text.split(" ")
#    print(len(words))
    size = 2000
    for a in range(0, len(words), size):
        tmp = " ".join(words[a:a+size])
        annotate = tagme.annotate(tmp)
        try:
            for ent in annotate.get_annotations(0.5):
                title = ent.entity_title
                title = title.encode('ascii', 'ignore')
                title = title.decode('ascii')
                wiki_id = get_wikiData_title2id(title)
#                print(type(title))
#                print("entity title from PRF: ", ent.entity_title, " wiki_id: ", wiki_id)
                ent_vec = transe_vec_for_entity(wiki_id)
                prf_annotations[str(wiki_id)] = ent_vec
                prf_id2title[str(wiki_id)] = str(title).strip()
        except:
            print('Error in TagMe document annotating')
            continue
#    print("prf annotations: ", prf_annotations)
    
    context, id2title = get_context_terms(prf_text)
    prf_annotations.update(context)
    prf_id2title.update(id2title)
    start = time.time()
    wiki_id2vec, wiki_id2title = wikipedia_search_only(query_text)   
    if len(wiki_id2vec) >= 1 and len(wiki_id2title) >= 1:
        prf_annotations.update(wiki_id2vec)
        prf_id2title.update(wiki_id2title)
    end = time.time()
    global complete_time
    complete_time = complete_time + (end -start)
#    print("updated prf annotations: ", prf_annotations)
#    print("the current execution time is: ", end - start)
    return prf_annotations, prf_id2title


# In[21]:
#Creating queries in INDRI format to run 
def query_reformulation(query_number, query_text, dataset, index_filepath, selected_ent2weight):
    ifac = 0.80
#     file_name =  "query/"+ dataset +"/"+ dataset +".ti"
#     file1 = open(file_name, "a+")
#    if dataset == "msmarco":
#        ifac = 0.80
    words = ""
    for key, value in list(selected_ent2weight.items())[0:20]:
        val = round(float(value),2)
        words = words + " " + str(val)
        words = words + " #combine(" + str(key) + ")"
        
#     file1.write("\n<query>\n")
    qno_string = "\n<query>\n<number>" + query_number + "</number>\n<text>"
#    file1.write(qno_string)
    query = "\n<query>\n<number>" + query_number + "</number>\n<text> #weight(" + str(ifac) + " #combine("+ query_text + ") " + str(round(float(1-ifac),2)) + " #combine(" + words.strip().lower() + "))</text>\n</query>"
#    file1.write(query)
#    file1.close()
    print(query)
    return query


# In[22]:

#Performing retrieval operation on queries dataset wise
def read_query(query_file, dataset, index_filepath):
        
    preamble_model_string = "<parameters>\n<index>"+index_filepath+"</index>\n<runID>runName</runID>\n<trecFormat>true</trecFormat>\n<rule>method:dirichlet;mu:1500</rule>\n<count>1000</count>"
    file_name = "query_G/prob_dist_ESF/"+dataset+"/"+ dataset+"_ti"
    file1 = open(file_name, "a+")
#    if dataset == "cw12" or dataset == "msmarco" or dataset == "gov2" or dataset == "trec678":
#        file1.write(preamble_model_string)
    
    f = open(query_file,"r")
    for line in f.readlines():
        
        annotations = OrderedDict() #will have entity_id:entity_vec, e.g. Q76:0.2 0.4 ..0.909
        id2title = OrderedDict() #will have entity_id:entity_title, e.g. Q76:barack_obama
        arr = [] #will have vecs of all ids e.g. 0.2 0.4 ... 0.9
        ids = [] #will have all ids e.g. Q76
        dic_id2clusterlabel = OrderedDict() #will have entity_id:cluster_label e.g. Q76:2
        
        query_number = line.split("=>")[0].strip()
        query_text = line.split("=>")[1].lower().strip()
        table = str.maketrans(dict.fromkeys(string.punctuation))
        query_text = query_text.translate(table) #removing punctuations
        
#       getting annotations and query_entities_cluster_center
        query_annotations, query_id2title, qcluster_center = query_annotate(query_text) 
        prf_annotations, prf_id2title = prf_annotate(query_number, query_text, index_filepath)       
        annotations.update(query_annotations) #e.g. Q76:0.1 0.12 ...0.8
        
        annotations.update(prf_annotations) #same format as query annotations
        id2title.update(query_id2title) #e.g. Q76: barack obama
        id2title.update(prf_id2title) #same as query id2title
#        print(id2title) # entity_id:entity_title,e.g. Q76:barack obama
#        print(annotations) # entity_id:entity_vec,e.g. Q76:0.2 0.4 .. 0.9
        
    
        selected_ent2weight = cluster(annotations, qcluster_center, id2title, sig=1.0)
        sorted_selected_ent2weight = dict(sorted(selected_ent2weight.items(), key=lambda item: item[1], reverse = True))
#        selected_ent2weight = wikipedia_search(query_text, selected_ent2weight, qcluster_center)
#        print(selected_ent2weight)
        query_string = query_reformulation(query_number, query_text, dataset, index_filepath, sorted_selected_ent2weight)
        file1.write(query_string)
    f.close()
    file1.write("\n</parameters>")
    file1.close()
    global complete_time
    print("complete execution time: ", complete_time)
    complete_time = 0.0
    #        print(query_number,query_text)
#    print(str(len(query_list)) + ' topics read')
#    return query_list


# In[23]:


def main():
    choice = 4 #change this value for different datasets 1 = gov2, 2 = cw12, 3 = trec678, 4 = robust, 5 = cw09
    start = 0
    end = 0
    dataset= ""
    query_file=""
    index_filepath= ""
    while choice < 7:
        if choice == 14:
            start = 0 # query starting position in query_strings list
            end = 150 # query ending position ('end' not included) in query_strings list
            dataset= "robust" #change for different datasets, will be used in filenames
            query_file= "Topics/robust-topics" #'Topics/gov2-topics' #'Topics/cw-topics'
            index_filepath = "/home/pankaj/indri-5.12/Queryformulation/indri_trec678_testindx.idx/"
            read_query(query_file, dataset, index_filepath)
        if choice == 1:
            start = 0 # query starting position in query_strings list
            end = 200 # query ending position ('end' not included) in query_strings list
            dataset= "cw09" #change for different datasets, will be used in filenames
            query_file= "Topics/cw09" #'Topics/gov2-topics' #'Topics/cw-topics'
            index_filepath = "/home/pankaj/indri-5.12/Queryformulation/indri_clueWeb09B_Index.idx/"
            read_query(query_file, dataset, index_filepath)
        if choice == 2:
            print("cw12 Topics running")
            start = 0 # query starting position in query_strings list
            end = 100 # query ending position ('end' not included) in query_strings list
            dataset= "cw12" #"trec678" #change for different datasets, will be used in filenames
            query_file= 'Topics/cw12-topics'
            index_filepath = "/home/pankaj/indri-5.12/Queryformulation/indri_clueweb12B_Index_00.idx/"
            read_query(query_file, dataset, index_filepath)
        if choice == 3:
            print("gov2 Topics running")
            start = 0 # query starting position in query_strings list
            end = 150 # query ending position ('end' not included) in query_strings list
            dataset= "gov2" #"cw" #"trec678" #change for different datasets, will be used in filenames
            query_file= 'Topics/gov2-topics' #"Topics/678-topics" #'Topics/gov2-topics' 
            index_filepath = "/home/pankaj/indri-5.12/Queryformulation/indri_gov2_index_01.idx/"
            read_query(query_file, dataset, index_filepath)
        if choice == 5:
            start = 0 # query starting position in query_strings list
            end = 100 # query ending position ('end' not included) in query_strings list
            dataset= "msmarco" #"trec678" #change for different datasets, will be used in filenames
            query_file= 'Topics/msmarco-topics'
            index_filepath = "/home/pankaj/indri-5.12/Queryformulation/indri_msmarco_fulldocsnew.idx/"
            read_query(query_file, dataset, index_filepath)
        if choice == 6:
            start = 0 # query starting position in query_strings list
            end = 3 # query ending position ('end' not included) in query_strings list
            dataset= "trec678" #change for different datasets, will be used in filenames
            query_file= "Topics/678-topics" #'Topics/gov2-topics' #'Topics/cw-topics'
            index_filepath = "/home/pankaj/indri-5.12/Queryformulation/indri_trec678_testindx.idx/"
            read_query(query_file, dataset, index_filepath)
        
        choice = choice + 1

    print("code end")
    


# In[24]:


if __name__=="__main__":
    main()

