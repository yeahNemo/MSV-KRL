# 用于提取多视图的完整数据，而非随机游走的数据。# 有问题
# 问题：projection.entity_to_labels_dict 只有2130项，但其实entity远大于这个数，导致ind_type全是不完整的。


import os
import json
import torch
import rdflib
import pickle
import argparse
import numpy as np
from ordered_set import OrderedSet
from collections import defaultdict
from rdflib.namespace import RDF, RDFS
from lib.Projection import Projection
from lib.Subgraph import Subgraph
from lib.Label import URI_parse, pre_process_words, label_item

# 解析配置文件
parser = argparse.ArgumentParser()
try:
    with open('./configs/helis_config.json', 'r') as json_file:
    # with open('./configs/go_config.json', 'r') as json_file:
        loaded_args = json.load(json_file)
        args = argparse.Namespace(**loaded_args)
except FileNotFoundError:
    args = parser.parse_args()

# 设置基本参数
ontology_file_name_suffix = ".nt" if args.ontology_name in ["foodon", "go"] else ".xml"
device = "cuda" if torch.cuda.is_available() else "cpu"
ontology_name = args.ontology_name
ontology_file_path = args.ontology_file_path
related_file_save_path = args.related_file_save_path

# 初始化投影对象
projection = Projection(ontology_file_path=ontology_file_path)

def extract_full_data():
    """提取完整的多视图数据，而非随机游走的数据"""
    
    print("开始提取完整的多视图数据...")
    
    ## (1). 将本体文件转换为RDF/XML文件和N-Triple文件（便于调试）
    projection.extract_and_projection()
    projection.save_rdf_graph_to_file(rdf_file_dir=related_file_save_path, ontology_name=ontology_name)
    
    ## (2). 提取类、个体、对象属性和数据属性的URI
    projection.extract_uris()
    obj_properties = projection.get_obj_props()
    data_properties = projection.get_data_props()
    anno_properties = projection.get_anno_props()
    classes = projection.get_classes()
    individuals = projection.get_individuals()
    
    # 保存实体和属性为pickle
    with open(related_file_save_path+"entities_props_dict_full.pkl", 'wb') as f:
        entities_props_dict = {"classes": classes, "individuals": individuals, "obj_props": obj_properties, "data_props": data_properties}
        pickle.dump(entities_props_dict, f)
    
    # 合并所有实体和属性
    entities_and_props = classes.union(individuals).union(obj_properties).union(anno_properties).union(data_properties).union((str(RDF.type), str(RDFS.subClassOf)))
    # 保存实体和属性到文件
    with open(related_file_save_path+"entities_props_full.txt", 'w') as f:
        for e in entities_and_props:
            f.write('%s\n' % e)
    
    ## (3). 提取Manchester语法的公理（包括subClassOf/equivalentClass限制、对象/数据属性、subClassOf和type）
    projection.create_manchester_syntax_axiom()
    with open(related_file_save_path+"axioms_full.txt", 'w') as f:
        for ax in projection.axioms_manchester:
            f.write('%s\n' % ax)
    
    ## (4). 提取标签、定义、注释和其他注释属性
    print("开始索引注释属性值...")
    uri_labels_dict = dict()
    annotations_list = list()   
    projection.get_annotations()
    print(f"带标签注释的实体数量: {len(projection.entity_to_labels_dict)}, 带所有注释属性的实体数量: {len(projection.entity_to_all_annotations_dict)}")
    
    for eop in entities_and_props:
        if eop in projection.entity_to_labels_dict and len(projection.entity_to_labels_dict[eop]) > 0:
            labels_list = list(projection.entity_to_labels_dict[eop])[0]
            # uri_labels_dict:{'uri': label}
            uri_labels_dict[eop] = pre_process_words(words=labels_list.split())
    
    for eop in entities_and_props:
        if eop in projection.entity_to_all_annotations_dict:
            for value in projection.entity_to_all_annotations_dict[eop]:
                if (value is not None) and (not (eop in projection.entity_to_labels_dict and value in projection.entity_to_labels_dict[eop])):
                    annotation = [eop] + value.split()  
                    annotations_list.append(annotation)
    
    # 保存注释属性值到文件
    with open(related_file_save_path+"annotations_full.txt", 'w') as f:
        for e in projection.entity_to_labels_dict:
            for v in projection.entity_to_labels_dict[e]:
                f.write('%s label %s\n' % (e, v))
        for a in annotations_list:
            f.write('%s\n' % ' '.join(a))
    print("完成索引注释属性值")
    
    ## (5). 生成URI句子
    print("开始生成URI句子")
    axiom_sentences = list()
    
    # 获取公理句子
    if os.path.exists(related_file_save_path+'axioms_full.txt'):
        for line in open(related_file_save_path+'axioms_full.txt').readlines():
            axiom_sentence = [item for item in line.strip().split()]
            axiom_sentences.append(axiom_sentence)
    print(f'提取了 {len(axiom_sentences)} 个公理句子')
    
    ## (6). 生成基于URI句子的注释句子并生成uri2anno字典
    print("开始生成所有注释句子...")
    entity2annotations_dict = projection.entity_to_all_annotations_dict
    entity_to_text_dict = dict()
    
    # 为所有实体和属性生成文本表示
    for uri in entities_and_props:
        anno_temp = ""
        try:
            if ontology_name == "helis":
                if uri in entity2annotations_dict:
                    annos = entity2annotations_dict[uri]
                    max_len = 0
                    for anno in annos:
                        if len(anno) > max_len:
                            anno_temp = anno
                            max_len = len(anno)
            elif ontology_name in ["foodon", "go"]:
                if uri in uri_labels_dict:
                    annos = uri_labels_dict[uri]    
                    anno_temp = " ".join(annos)
        except:
            anno_temp = label_item(item=uri, uri_labels_dict=uri_labels_dict)
            anno_temp = " ".join(anno_temp)
        
        if ontology_name == "helis":
            if "#" in uri:
                entity_to_text_dict[uri] = anno_temp 
        elif ontology_name in ["foodon", "go"]:
            if "http://" in uri:
                entity_to_text_dict[uri] = anno_temp 
    
    with open(related_file_save_path+'entity_to_text_dict_full.pkl', 'wb') as f:
        pickle.dump(entity_to_text_dict, f)
    print("完成生成所有注释句子")
    
    ## (7). 生成子图并提取完整的子图数据
    # 步骤1: 生成子图
    print("开始生成RDF文件的5个子图...")
    subgraph = Subgraph(related_file_save_path + args.ontology_name + ontology_file_name_suffix)
    subgraph.generate_subgraphs(classes, individuals, obj_properties, data_properties, related_file_save_path+'subgraphs/')
    print("完成生成子图")
    
    # 步骤2: 提取子图的完整数据（而非随机游走）
    print("开始提取子图的完整数据...")
    
    # 提取subClassOf子图的完整数据
    print("开始提取subClassOf子图的完整数据")
    subclassof_graph = rdflib.Graph().parse(related_file_save_path+'subgraphs/subclassof.xml')
    subclassof_triples = []
    for s, p, o in subclassof_graph:
        subclassof_triples.append([str(s), str(p), str(o)])
    
    # 生成subClassOf注释句子
    subclassof_anno_sentences = []
    for triple in subclassof_triples:
        if len(triple) < 3:
            continue
        temp_list = []
        for uri in triple:
            try:
                anno_info = entity_to_text_dict[uri]
                temp_list.append(anno_info)
            except:
                temp_list.append(uri)
        subclassof_anno_sentences.append(temp_list)
    
    with open(related_file_save_path+'subgraphs/subclassof_anno_sentences.txt', 'w') as f:
        for sentence in subclassof_anno_sentences:
            f.write(" [SEP] ".join(sentence) + '\n')
    print(f"完成提取subClassOf子图的完整数据，长度: {len(subclassof_anno_sentences)}")
    
    # 提取class_props子图的完整数据
    print("开始提取class_props子图的完整数据")
    class_props_graph = rdflib.Graph().parse(related_file_save_path+'subgraphs/class_props.xml')
    class_props_triples = []
    for s, p, o in class_props_graph:
        class_props_triples.append([str(s), str(p), str(o)])
    
    # 生成class_props注释句子
    class_props_anno_sentences = []
    for triple in class_props_triples:
        if len(triple) < 3:
            continue
        temp_list = []
        for uri in triple:
            try:
                anno_info = entity_to_text_dict[uri]
                temp_list.append(anno_info)
            except:
                temp_list.append(uri)
        class_props_anno_sentences.append(temp_list)
    
    with open(related_file_save_path+'subgraphs/class_props_anno_sentences.txt', 'w') as f:
        for sentence in class_props_anno_sentences:
            f.write(" [SEP] ".join(sentence) + '\n')
    print(f"完成提取class_props子图的完整数据，长度: {len(class_props_anno_sentences)}")
    
    # 提取ind_type子图的完整数据
    print("开始提取ind_type子图的完整数据")
    ind_type_graph = rdflib.Graph().parse(related_file_save_path+'subgraphs/ind_type.xml')
    ind_type_triples = []
    for s, p, o in ind_type_graph:
        ind_type_triples.append([str(s), str(p), str(o)])
    
    # 生成ind_type注释句子
    ind_type_anno_sentences = []
    for triple in ind_type_triples:
        if len(triple) < 3:
            continue
        temp_list = []
        for uri in triple:
            try:
                anno_info = entity_to_text_dict[uri]
                temp_list.append(anno_info)
            except:
                temp_list.append(uri)
        ind_type_anno_sentences.append(temp_list)
    
    with open(related_file_save_path+'subgraphs/ind_type_anno_sentences.txt', 'w') as f:
        for sentence in ind_type_anno_sentences:
            f.write(" [SEP] ".join(sentence) + '\n')
    print(f"完成提取ind_type子图的完整数据，长度: {len(ind_type_anno_sentences)}")
    
    # 提取obj_props子图的完整数据
    print("开始提取obj_props子图的完整数据")
    obj_props_graph = rdflib.Graph().parse(related_file_save_path+'subgraphs/obj_props.xml')
    obj_props_triples = []
    for s, p, o in obj_props_graph:
        obj_props_triples.append([str(s), str(p), str(o)])
    
    # 生成obj_props注释句子
    obj_props_anno_sentences = []
    for triple in obj_props_triples:
        if len(triple) < 3:
            continue
        temp_list = []
        for uri in triple:
            try:
                anno_info = entity_to_text_dict[uri]
                temp_list.append(anno_info)
            except:
                temp_list.append(uri)
        obj_props_anno_sentences.append(temp_list)
    
    with open(related_file_save_path+'subgraphs/obj_props_anno_sentences.txt', 'w') as f:
        for sentence in obj_props_anno_sentences:
            f.write(" [SEP] ".join(sentence) + '\n')
    print(f"完成提取obj_props子图的完整数据，长度: {len(obj_props_anno_sentences)}")
    
    # 提取data_props子图的完整数据
    print("开始提取data_props子图的完整数据")
    data_props_graph = rdflib.Graph().parse(related_file_save_path+'subgraphs/data_props.xml')
    data_props_triples = []
    for s, p, o in data_props_graph:
        data_props_triples.append([str(s), str(p), str(o)])
    
    # 生成data_props注释句子
    data_props_anno_sentences = []
    for triple in data_props_triples:
        if len(triple) < 3:
            continue
        temp_list = []
        for uri in triple:
            try:
                anno_info = entity_to_text_dict[uri]
                temp_list.append(anno_info)
            except:
                temp_list.append(uri)
        data_props_anno_sentences.append(temp_list)
    
    with open(related_file_save_path+'subgraphs/data_props_anno_sentences.txt', 'w') as f:
        for sentence in data_props_anno_sentences:
            f.write(" [SEP] ".join(sentence) + '\n')
    print(f"完成提取data_props子图的完整数据，长度: {len(data_props_anno_sentences)}")
    
    # 翻译公理到注释
    print("开始将公理句子翻译为注释")
    axiom_anno_sentences = []
    for sentence in axiom_sentences:
        if len(sentence) < 3:
            continue
        temp_list = []
        for uri in sentence:
            try:
                anno_info = entity_to_text_dict[uri]
                temp_list.append(anno_info)
            except:
                temp_list.append(uri)
        axiom_anno_sentences.append(temp_list)
    
    with open(related_file_save_path+'subgraphs/axiom_anno_sentences_sep.txt', 'w') as f:
        for sentence in axiom_anno_sentences:
            f.write(" [SEP] ".join(sentence) + '\n')
    print(f"完成将公理句子翻译为注释，长度: {len(axiom_anno_sentences)}")
    
    # 合并所有注释句子
    all_anno_sentences = subclassof_anno_sentences + class_props_anno_sentences + ind_type_anno_sentences + obj_props_anno_sentences + data_props_anno_sentences
    print(f"注释句子总数: {len(all_anno_sentences)}")
    all_anno_sentences = set([" [SEP] ".join(sentence) for sentence in all_anno_sentences])
    print(f"去除重复句子后，总数: {len(all_anno_sentences)}")
    
    with open(related_file_save_path+'subgraphs/all_anno_sentences_sep.txt', 'w') as f:
        for sentence in all_anno_sentences:
            f.write(sentence + '\n')
    print(f"完成生成所有子图的注释句子（已去除重复句子），总数: {len(all_anno_sentences)}")
    
    # 保存三元组数据用于后续处理
    all_triples = subclassof_triples + class_props_triples + ind_type_triples + obj_props_triples + data_props_triples
    with open(related_file_save_path+'all_triples.pkl', 'wb') as f:
        pickle.dump(all_triples, f)
    
    print("完成提取完整的多视图数据")

if __name__ == "__main__":
    extract_full_data()