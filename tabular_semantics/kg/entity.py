'''
Created on 20 Mar 2019

@author: ejimenez-ruiz
'''
from enum import Enum


class KG(Enum):
        DBpedia = 0
        Wikidata = 1
        Google = 2        
        All = 3

class URI_KG(object):
    
    dbpedia_uri_resource = 'http://dbpedia.org/resource/'
    
    dbpedia_uri_property = 'http://dbpedia.org/property/'
    
    dbpedia_uri = 'http://dbpedia.org/ontology/'
    wikidata_uri ='http://www.wikidata.org/entity/'
    schema_uri = 'http://schema.org/' 
    
    uris = list()
    uris.append(dbpedia_uri)
    uris.append(wikidata_uri)
    uris.append(schema_uri)
    
    uris_resource = list()
    uris_resource.append(dbpedia_uri_resource)
    uris_resource.append(wikidata_uri)
    
    wikimedia_disambiguation_concept=wikidata_uri+'Q4167410'
    
    
    avoid_predicates=set()
    avoid_predicates.add("http://dbpedia.org/ontology/wikiPageDisambiguates")
    avoid_predicates.add("http://dbpedia.org/ontology/wikiPageRedirects")
    avoid_predicates.add("http://dbpedia.org/ontology/wikiPageWikiLink")
    avoid_predicates.add("http://dbpedia.org/ontology/wikiPageID")
    
    
    #Large amount of text
    avoid_predicates.add("http://dbpedia.org/ontology/abstract")
    avoid_predicates.add("http://www.w3.org/2000/01/rdf-schema#comment")
    
    
    
    
    avoid_predicates.add("http://dbpedia.org/ontology/wikiPageRevisionID")
    avoid_predicates.add("http://dbpedia.org/ontology/wikiPageExternalLink")
    avoid_predicates.add("http://purl.org/dc/terms/subject") #Link to categories
    
    avoid_predicates.add("http://www.w3.org/2000/01/rdf-schema#seeAlso")
    avoid_predicates.add("http://purl.org/linguistics/gold/hypernym")
    avoid_predicates.add("http://xmlns.com/foaf/0.1/primaryTopic")
    #avoid_predicates.add("http://www.w3.org/2002/07/owl#differentFrom")
    #avoid_predicates.add("http://www.w3.org/2002/07/owl#sameAs")
    avoid_predicates.add("http://dbpedia.org/property/related")

    avoid_top_concepts = set()
    avoid_top_concepts.add("http://www.w3.org/2002/07/owl#Thing")
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q35120")  # entity
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q830077")  # subject
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q18336849")  # item with given name property
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q23958946")  # individual/instance
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q26720107")  # subject of a right
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q488383")  # object
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q4406616")  # concrete object
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q29651224")  # natural object
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q223557")  # physical object
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q16686022")  # natural physical object

    # new additions
    # classes, objects, etc.
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q16889133")  # class
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q5127848")  # class
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q7184903")  # abstract object
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q16686448")  # artificial entity
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q151885")  # concept
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q8205328")  # artificial physical object
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q29651519")  # mental object
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q24017414")  # first-order metaclass
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q23960977")  # (meta)class
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q19478619")  # metaclass
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q23959932")  # fixed-order metaclass
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q21522864")  # class or metaclass of Wikidata ontology
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q19361238")  # Wikidata metaclass

    # general high-level concepts
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q595523")  # notion
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q1969448")  # term
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q4393498")  # representation
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q2145290")  # mental representation
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q1923256")  # intension
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q131841")  # idea
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q24229398")  # agent
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q1190554")  # occurrence
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q1914636")  # activity
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q26907166")  # temporal entity
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q4026292")  # action
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q9332")  # behavior
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q6671777")  # structure
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q1347367")  # capability
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q937228")  # property
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q930933")  # relation
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q1207505")  # quality
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q39875001")  # measure
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q337060")  # perceptible object
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q483247")  # phenomenon
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q16722960")  # phenomenon
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q602884")  # social phenomenon
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q769620")  # social action
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q1656682")  # event
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q386724")  # work
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q17537576")  # creative work
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q15621286")  # intellectual work
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q15401930")  # product (result of work/effort)
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q28877")  # goods
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q58778")  # system

    # groups and collections
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q99527517")  # collection entity
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q98119401")  # group or class of physical objects
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q36161")  # set
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q20937557")  # series
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q16887380")  # group
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q61961344")  # group of physical objects
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q16334295")  # group of humans
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q16334298")  # group of living things
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q874405")  # social group
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q3533467")  # group action

    # geographic terms
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q618123")  # geographical object
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q20719696")  # physico-geographical object
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q1503302")  # geographic object
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q58416391")  # spatial entity
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q15642541")  # human-geographic territorial entity
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q1496967")  # territorial entity
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q27096213")  # geographic entity
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q27096235")  # artificial geographic entity
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q2221906")  # geographic location
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q58415929")  # spatio-temporal entity
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q5839809")  # regional space
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q1251271")  # geographic area
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q82794")  # geographic region
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q35145263")  # natural geographic object
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q27096220")  # natural geographic entity

    # units
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q3563237")  # economic unit
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q2198779")  # unit
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q5371079")  # emic unit
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q20817253")  # linguistic unit
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q3695082")  # sign
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q7887142")  # unit of analysis
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q15198957")  # aspect of music
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q271669")  # landform
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q12766313")  # geomorphological unit
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q15989253")  # part

    # others (less general)
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q17334923")  # location
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q3257686")  # locality
    # avoid_top_concepts.add("http://www.wikidata.org/entity/Q486972")  # human settlement
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q43229")  # organization
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q177634")  # community
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q15324")  # body of water
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q863944")  # land waters
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q2507626")  # water area
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q3778211")  # legal person
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q4330518")  # research object
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q56061")  # administrative territorial entity
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q1048835")  # political territorial entity
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q1799794")  # administrative territorial entity of a specific level
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q12076836")  # administrative territorial entity of a single country
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q7210356")  # political organisation
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q155076")  # juridical person
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q12047392")  # legal form
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q1063239")  # polity
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q32178211")  # musical organization
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q4897819")  # role
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q1781513")  # position
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q1792379")  # art genre
    avoid_top_concepts.add("http://www.wikidata.org/entity/Q483394")  # genre


    def __init__(self):
        pass
        

class KGEntity(object):
    
    def __init__(self, enity_id, label, description, types, source):
        
        self.ident = enity_id
        self.label = label
        self.desc = description #sometimes provides a very concrete type or additional semantics
        self.types = types  #set of semantic types
        self.source = source  #KG of origin: dbpedia, wikidata or google kg
        
    
    def __repr__(self):
        return "<id: %s, label: %s, description: %s, types: %s, source: %s>" % (self.ident, self.label, self.desc, self.types, self.source)

    def __str__(self):
        return "<id: %s, label: %s, description: %s, types: %s, source: %s>" % (self.ident, self.label, self.desc, self.types, self.source)
    
    
    def getId(self):
        return self.ident
    
    '''
    One can retrieve all types or filter by KG: DBpedia, Wikidata and Google (Schema.org)
    '''
    def getTypes(self, kgfilter=KG.All):
        if kgfilter==KG.All:
            return self.types
        else:
            kg_uri = URI_KG.uris[kgfilter.value]
            filtered_types = set()
            for t in self.types:
                if t.startswith(kg_uri):
                    filtered_types.add(t)
            
            return filtered_types 
    
    def getLabel(self):
        return self.label
    
    def getDescription(self):
        return self.desc
    
    def getSource(self):
        return self.sourcec
    
    
    def addType(self, cls):
        self.types.add(cls)
    
    def addTypes(self, types):
        self.types.update(types)
        
        
        
if __name__ == '__main__':
    print(URI_KG.uris[KG.DBpedia.value])
    print(KG.DBpedia.value)
          
    