import  pandas as pd
from datasets import Dataset,load_dataset,DatasetDict,concatenate_datasets
from datasets import Features, Sequence, ClassLabel, Value
from huggingface_hub import login
import os
# import requests
# import librosa
import numpy as np
import argparse
import json
# from huggingface_hub import login
# from datasets import Dataset,load_dataset,concatenate_datasets, Audio
# from itertools import islice
import asyncio
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset From Datacreation tool")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file.")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path) as f:
        config = json.load(f)
    return config

two_word_tags = {
        "pos_tags": [22, 22],
        "chunk_tags": [11, 12],
        "ner_tags": [1, 2]
    }
single_word_tags_lastname = {
        "pos_tags": [22],
        "chunk_tags": [12],
        "ner_tags": [2]
    }

single_word_tags_firstname = {
        "pos_tags": [22],
        "chunk_tags": [11],
        "ner_tags": [1]
    }



# Function to determine the appropriate tags
def get_tags(words,value):
    if value == "lastname":
        single_words = single_word_tags_lastname
    else:
        single_words = single_word_tags_firstname
    if len(words) == 1:
        return single_words
    elif len(words) == 2:
        return two_word_tags
    else:
        # Handle more cases if needed, here using a simple replication for demonstration
        return {
            "pos_tags": [22] * len(words),
            "chunk_tags": [11, 12] + [11] * (len(words) - 2),
            "ner_tags": [1, 2] + [1] * (len(words) - 2)
        }




single_words_city_street = {
        "pos_tags": [22],
        "chunk_tags": [11],
        "ner_tags": [5]
    }

two_word_tags_city_street = {
        "pos_tags": [22,22],
        "chunk_tags": [11,12],
        "ner_tags": [5,6]
    }

def get_tags_street_city(words,value):
    if len(words) == 1:
        return single_words_city_street
    elif len(words) == 2:
        return two_word_tags_city_street
    else:
        # Handle more cases if needed, here using a simple replication for demonstration
        return {
            "pos_tags": [22] * len(words),
            "chunk_tags": [11, 12] + [11] * (len(words) - 2),
            "ner_tags": [5, 6] + [5] * (len(words) - 2)
        } 

def make_dict_for_list(list_of_name,list_of_name_containing_multiple_words):
    data = {
        'id': [str(i) for i in range(1, len(list_of_name) + 1)],  # Convert each id to a string
        'tokens': [[name.lower()] for name in list_of_name]
    }
   
    data_1 = {
        'id': [str(i) for i in range(1, len(list_of_name_containing_multiple_words) + 1)],  # Convert each id to a string
        'tokens': [[name.split()[0], ' '.join(name.split()[1:])] if len(name.split()) > 1 else [name] for name in list_of_name_containing_multiple_words]
    }
    

    data_2 = {
        'id': [str(i) for i in range(1, len(list_of_name_containing_multiple_words) + 1)],  # Convert each id to a string
        'tokens': [[' '.join(name.split()[:-1]), name.split()[-1]] if len(name.split()) > 1 else [name] for name in list_of_name_containing_multiple_words]
    }
    


    return data,data_1,data_2


def addPosChunkNer_with_dataset(dataset,tag_param):
    # Lists to store the new column values
    pos_tags_column = []
    chunk_tags_column = []
    ner_tags_column = []

    # Iterate through each row in the dataset and determine the tags
    for words in dataset['tokens']:
        if tag_param == "firstname" or tag_param == "lastname":
            # print("ENter to this firstname lastname param")
            tags = get_tags(words,tag_param)
        elif tag_param == "street" or tag_param == "city":
            # print("ENter to this city street param")
            tags = get_tags_street_city(words,tag_param)
        pos_tags_column.append(tags['pos_tags'])
        chunk_tags_column.append(tags['chunk_tags'])
        ner_tags_column.append(tags['ner_tags'])

    # Add the new columns to the dataset
    dataset = dataset.add_column("pos_tags", pos_tags_column)
    dataset = dataset.add_column("chunk_tags", chunk_tags_column)
    dataset = dataset.add_column("ner_tags", ner_tags_column)

    # Print the modified dataset
    # print(dataset)
    return dataset

def make_dataset_from_lastname(df):
    #get only first columns values
    first_column_values = df.iloc[:,0].tolist()
    # print("FirstColumnsValues",first_column_values)
    names_list = first_column_values

    count_all = 0
    # print("Previous length of list:",len(names_list))
    # Iterate through each word in the list
    for word in names_list:
        # Check if both '(' and ')' are present in the word
        if "." in word or '(' in word or ')' in word or '-' in word or "'" in word:
            count_all += 1

    # print("Number of words containing apostropy:", count_all)

    # Cleaned list to store new values
    cleaned_values = []

    # Iterate through each item in the original list
    for value in names_list:
        if "." in value or '(' in value or ')' in value or '-' in value or "'" in value:
            # Remove apostrophes and dashes
            cleaned_value = value.replace("'", "").replace("-", " ").replace("(", "").replace(")", "").replace("."," ")
            # Append the cleaned value to the cleaned_values list
            cleaned_values.append(cleaned_value)

    # Append cleaned values back to the original list
    names_list.extend(cleaned_values)

    # Convert original values to lowercase and append to the cleaned_values list
    list_of_name = [value.lower() for value in names_list]
    list_of_name_containing_multiple_words = [name for name in list_of_name if len(name.split()) > 1]
    
    # print(list_of_name)
    print("length of name list:",len(list_of_name))
    print("length of list_of_name_containing_multiple_words list:",len(list_of_name_containing_multiple_words))

    data,data_1,data_2 = make_dict_for_list(list_of_name,list_of_name_containing_multiple_words)

    
    # Create a DataFrame
    df = pd.DataFrame(data)
    # df = df.drop_duplicates()
    df_1 = pd.DataFrame(data_1)
    # df_1 = df_1.drop_duplicates()
    df_2 = pd.DataFrame(data_2)
    # Concatenate DataFrames
    df_combined = pd.concat([df, df_1,df_2], axis=0)
    # Drop duplicates
    # try:
    #     df_combined = df_combined.drop_duplicates()
    #     print("DataFrame after dropping duplicates:")
    #     print(df_combined)
    # except Exception as e:
    #     print(f"Error during dropping duplicates: {e}")
        
    dataset = Dataset.from_pandas(df_combined)
    
    update_dataset = addPosChunkNer_with_dataset(dataset,"lastname")
    return update_dataset


def find_numberof_non_character_contain(list):
    count_all = 0

    # Iterate through each word in the list
    for word in list:
        # Check if both '(' and ')' are present in the word
        if "." in word or '(' in word or ')' in word or '-' in word or "'" in word:
            count_all += 1

    print("Number of words containing apostropy, dash, bracket, fullstop:", count_all)
    return count_all

def remove_non_character_and_append_again(list):
    # Cleaned list to store new values
    cleaned_values = []

    # Iterate through each item in the original list
    for value in list:
        # Remove apostrophes and dashes
        if "." in value or '(' in value or ')' in value or '-' in value or "'" in value:
            cleaned_value = value.replace("'", "").replace("-", " ").replace("(", "").replace(")", "").replace("."," ")
            # Append the cleaned value to the cleaned_values list
            cleaned_values.append(cleaned_value)

    # Append cleaned values back to the original list
    list.extend(cleaned_values)

    # Convert original values to lowercase and append to the cleaned_values list
    list = [value.lower() for value in list]

    return list

    
#Make Dataset for city list
def get_list_from_city_street_csv(df):
    print("Enter to Street City function")
    street_list = df.iloc[:,0].tolist()
    city_list = df.iloc[:,1].tolist()
    print("number of street",len(street_list))
    print("number of city",len(city_list))


    #find the number of city containing apostropy('), dash(-), bracket( () ), fullstop(.) (This is not necessary to do)
    total_non_character_contain_in_street = find_numberof_non_character_contain(street_list)
    total_non_character_contain_in_city = find_numberof_non_character_contain(city_list)
    print("total_non_character_contain_in_street",total_non_character_contain_in_street)
    print("total_non_character_contain_in_city",total_non_character_contain_in_city)

    list_of_street = remove_non_character_and_append_again(street_list)
    list_of_city = remove_non_character_and_append_again(city_list)
    print("number of street",len(list_of_street))
    print("number of city",len(list_of_city))
    list_of_street_containing_multiple_words = [name for name in list_of_street if len(name.split()) > 1]
    print("number of street multiple words",len(list_of_street_containing_multiple_words))
    list_of_city_containing_multiple_words = [name for name in list_of_city if len(name.split()) > 1]
    print("number of city multiple words",len(list_of_city_containing_multiple_words))
    

    data_street_1,data_street_2,data_street_3 = make_dict_for_list(list_of_street,list_of_street_containing_multiple_words)
    data_city_1,data_city_2,data_city_3 = make_dict_for_list(list_of_city,list_of_city_containing_multiple_words)

    print("AAAAAAAAAAAAAAAAAAAAAA")
    df_street_1 = pd.DataFrame(data_street_1)
    # df_street_1 = df_street_1.drop_duplicates()
    print("BBBBBBBBBBBBBBBBBBBBBBBB")
    df_street_2 = pd.DataFrame(data_street_2)
    # df_street_2 = df_street_2.drop_duplicates()
    df_city_1 = pd.DataFrame(data_city_1)
    # df_city_1 = df_city_1.drop_duplicates()
    df_city_2 = pd.DataFrame(data_city_2)
    # df_city_2 = df_city_2.drop_duplicates()

    
    dataset_street_1 = Dataset.from_pandas(df_street_1)
    dataset_street_2 = Dataset.from_pandas(df_street_2)
    print("Street dataset before update is:",dataset_street_2)

    update_dataset_street_1 = addPosChunkNer_with_dataset(dataset_street_1,"street")
    update_dataset_street_2 = addPosChunkNer_with_dataset(dataset_street_2,"street")
    print("Street dataset after update is:",update_dataset_street_2)

    
    # for i in range(5):
    #     print("streetDataset:",update_dataset_street_2[i])



    dataset_city_1 = Dataset.from_pandas(df_city_1)
    dataset_city_2 = Dataset.from_pandas(df_city_2)
    print("City dataset before update is:",dataset_city_2)
    update_dataset_city_1 = addPosChunkNer_with_dataset(dataset_city_1,"city")
    update_dataset_city_2 = addPosChunkNer_with_dataset(dataset_city_2,"city")
    print("City dataset after update is:",update_dataset_city_2)

    # for i in range(5):
    #     print("cityDataset:",update_dataset_city_2[i])

    update_dataset = concatenate_datasets([update_dataset_street_1,update_dataset_street_2,update_dataset_city_1,update_dataset_city_2])
    print("Combined dataset of street and city:",update_dataset)
    return update_dataset

def dataset_for_firstnamelist(first_names_list):
    count_all = 0

    # Iterate through each word in the list
    for word in first_names_list:
        # Check if both '(' and ')' are present in the word
        if "." in word or '(' in word or ')' in word or '-' in word or "'" in word:
            count_all += 1

    # print("Number of First names containing apostropy, fullstop, dash, bracket:", count_all)
    # Cleaned list to store new values
    cleaned_values = []
    # Iterate through each item in the original list
    for value in first_names_list:
        if "'" in value or "-" in value or "(" in value or ")" in value or "." in value:
            # Remove apostrophes and dashes
            cleaned_value = value.replace("'", "").replace("-", " ").replace("(", "").replace(")", "").replace("."," ")
            # Append the cleaned value to the cleaned_values list
            cleaned_values.append(cleaned_value)

    # Append cleaned values back to the original list
    first_names_list.extend(cleaned_values)

    # Convert original values to lowercase and append to the cleaned_values list
    list_of_firstname = [value.lower() for value in first_names_list]
    list_of_name_containing_multiple_words = [name for name in list_of_firstname if len(name.split()) > 1]
    # print(list_of_firstname)
    print("Total length of firstname list",len(list_of_firstname))
    print("Total length of firstname list containing multiple words",len(list_of_name_containing_multiple_words))

    data,data_1,data_2 = make_dict_for_list(list_of_firstname,list_of_name_containing_multiple_words)
    df = pd.DataFrame(data)
    # df = df.drop_duplicates()
    df_1 = pd.DataFrame(data_1)
    # df_1 = df_1.drop_duplicates()
    df_2 = pd.DataFrame(data_2)
    df_combined = pd.concat([df, df_1,df_2], axis=0)

    dataset = Dataset.from_pandas(df_combined)
    update_dataset = addPosChunkNer_with_dataset(dataset,"firstname")
    return update_dataset




def dataset_for_lastnamelist(last_names_list):    
    # cleaned_values = []

    # # Iterate through each item in the original list
    # for value in last_names_list:
    #     if "'" in value or "-" in value or "(" in value or ")" in value or "." in value:
    #         # Remove apostrophes and dashes
    #         cleaned_value = value.replace("'", "").replace("-", " ").replace("(", "").replace(")", "").replace("."," ")
    #         # Append the cleaned value to the cleaned_values list
    #         cleaned_values.append(cleaned_value)

    # # Append cleaned values back to the original list
    # last_names_list.extend(cleaned_values)

    # # Convert original values to lowercase and append to the cleaned_values list
    # list_of_lastname = [value.lower() for value in last_names_list]

    list_of_lastname = remove_non_character_and_append_again(last_names_list)

    list_of_name_containing_multiple_words = [name for name in list_of_lastname if len(name.split()) > 1]
    # print(list_of_lastname)
    print("Total length of lastname list",len(list_of_lastname))
    print("Total length of lastname list containing multiple words",len(list_of_name_containing_multiple_words))


   
    data,data_1,data_2 = make_dict_for_list(list_of_lastname,list_of_name_containing_multiple_words)
    df = pd.DataFrame(data)
    # df = df.drop_duplicates()
    df_1 = pd.DataFrame(data_1)
    # df_1 = df_1.drop_duplicates()
    df_2 = pd.DataFrame(data_2)
    df_combined = pd.concat([df, df_1,df_2], axis=0)

    dataset = Dataset.from_pandas(df_combined)
    update_dataset = addPosChunkNer_with_dataset(dataset,"lastname")
    return update_dataset



def get_list_from_firstname_lastname_excell(df):
    # print("Enter to get_list_from_firstname_lastname_excell")
    # print("NExt dataframe is:",df.head(20))
    filtered_lastname = df[df['category'] == 'Lastname']['name']
    filtered_firstname = df[df['category'] == 'Firstname']['name']

    # print("Firstname list:",filtered_firstname)
    # print("Lastname list:",filtered_lastname)
    first_names_list = filtered_firstname.tolist()
    last_names_list = filtered_lastname.tolist()

    dataset_firstname = dataset_for_firstnamelist(first_names_list)
    dataset_lastname= dataset_for_lastnamelist(last_names_list)

    print("This is the dataset firstname",dataset_firstname)
    print("This is the dataset lastname",dataset_lastname)

    for i in range(5):
        print(dataset_lastname[i])
    return concatenate_datasets([dataset_firstname,dataset_lastname])
    

    
async def data_push_to_hub(dataset,dataset_name):
    dataset.push_to_hub(dataset_name)

async def update_feature_of_new_dataset(update_dataset,existing_conll2003_dataset):
    conll2003_dataset = load_dataset(existing_conll2003_dataset)
    # print("Conll2003 Dataset features are",dataset["train"].features)
    # dataset1 = load_dataset(config["dataset_name_for_append_above_generated_dataset"])


    pos_labels = conll2003_dataset['train'].features['pos_tags'].feature
    chunk_labels = conll2003_dataset['train'].features['chunk_tags'].feature
    ner_labels = conll2003_dataset['train'].features['ner_tags'].feature

    

    def convert_int_to_classlabel(dataset, feature_name, class_label):
        def map_int_to_classlabel(example):
            example[feature_name] = [class_label.int2str(label) for label in example[feature_name]]
            return example
        return dataset.map(map_int_to_classlabel)

    def update_dataset_features(dataset, pos_labels, chunk_labels, ner_labels):
        features = Features({
            'id': Value(dtype='string', id=None),
            'tokens': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
            'pos_tags': Sequence(feature=pos_labels, length=-1, id=None),
            'chunk_tags': Sequence(feature=chunk_labels, length=-1, id=None),
            'ner_tags': Sequence(feature=ner_labels, length=-1, id=None),
        })
        return dataset.cast(features)

    # Assuming update_dataset is your new dataset
    new_dataset = convert_int_to_classlabel(update_dataset, 'pos_tags', pos_labels)
    new_dataset = convert_int_to_classlabel(update_dataset, 'chunk_tags', chunk_labels)
    new_dataset = convert_int_to_classlabel(update_dataset, 'ner_tags', ner_labels)
    update_dataset = update_dataset_features(update_dataset, pos_labels, chunk_labels, ner_labels)
    print("Updated Dataset is:",update_dataset)
    print("Updated Feature is:",update_dataset["train"].features)
    return update_dataset


def split_to_train_test_validation(dataset):
    split_dataset = dataset.train_test_split(test_size=0.2)
    # Further split the temp dataset into validation (50% of 20%) and test (50% of 20%)
    temp_dataset = split_dataset['test'].train_test_split(test_size=0.5)

    # Combine into a DatasetDict
    new_dataset_dict = DatasetDict({
        'train': split_dataset['train'],
        'validation': temp_dataset['train'],
        'test': temp_dataset['test']
    })
    update_dataset = new_dataset_dict
    return update_dataset


def make_dataset_from_excell_or_csv(df,config):
    # print("CONifg is:",config)
    # return "hello"
    column_values_to_list = df.iloc[:,int(config["column_want_to_load"])-1].tolist()
    # print("FirstColumnsValues",column_values_to_list)
    list_of_column = remove_non_character_and_append_again(column_values_to_list)
    list_of_column_containing_multiple_words = [lst for lst in list_of_column if len(lst.split()) > 1]

    print("lenght of list column is ",len(list_of_column))
    print("lenght of list containing multiple word column is ",len(list_of_column_containing_multiple_words))
    # return ("hello")
    data,data_1,data_2 = make_dict_for_list(list_of_column,list_of_column_containing_multiple_words)
    df = pd.DataFrame(data)
    df_1 = pd.DataFrame(data_1)
    df_2 = pd.DataFrame(data_2)
    df_combined = pd.concat([df, df_1,df_2], axis=0)
    update_dataset = Dataset.from_pandas(df_combined)
    print(update_dataset)

    #add tags

    two_word = {
        "pos_tags": [22, 22],
        "chunk_tags": [11, 12],
        "ner_tags": config["ner_tag_for_multiple_word"]
    }
    single_word = {
            "pos_tags": [22],
            "chunk_tags": [12],
            "ner_tags": config["ner_tag_for_single_word"]
        }


    def get_tags_for_datas(words):
        if len(words) == 1:
            return single_word
        elif len(words) == 2:
            return two_word
        else:
            # Handle more cases if needed, here using a simple replication for demonstration
            return {
                "pos_tags": config["pos_tag_for_single_word"] * len(words),
                "chunk_tags": config["chunk_tag_for_multiple_word"] + config["chunk_tag_for_single_word"] * (len(words) - 2),
                "ner_tags": config["ner_tag_for_multiple_word"] + config["ner_tag_for_single_word"] * (len(words) - 2)
            } 
    
    def addPosChunkNer_for_generarted_dataset(dataset):
        # Lists to store the new column values
        pos_tags_column = []
        chunk_tags_column = []
        ner_tags_column = []

        # Iterate through each row in the dataset and determine the tags
        for words in dataset['tokens']:
            # if tag_param == "firstname" or tag_param == "lastname":
            #     # print("ENter to this firstname lastname param")
            #     tags = get_tags(words,tag_param)
            # elif tag_param == "street" or tag_param == "city":
            #     # print("ENter to this city street param")
            #     tags = get_tags_for_datas(words,tag_param)
            tags = get_tags_for_datas(words)
            pos_tags_column.append(tags['pos_tags'])
            chunk_tags_column.append(tags['chunk_tags'])
            ner_tags_column.append(tags['ner_tags'])

        # Add the new columns to the dataset
        dataset = dataset.add_column("pos_tags", pos_tags_column)
        dataset = dataset.add_column("chunk_tags", chunk_tags_column)
        dataset = dataset.add_column("ner_tags", ner_tags_column)

        # Print the modified dataset
        # print(dataset)
        return dataset

    
    print("Dataset before adding pos chunk ner",update_dataset)
    update_dataset = addPosChunkNer_for_generarted_dataset(update_dataset)
    print("Dataset after adding pos chunk ner",update_dataset)
    return update_dataset

async def main():
    args = parse_args()
    config = load_config(args.config)
    
    try:
        # print("Logging to hub")
        login(token=config["huggingface_hub_token"])
        # print("Your huggingface token is:",config["huggingface_hub_token"])
    except Exception as e:
        print("Error logging into hub:", str(e))

    total_datasets = []

    if config["make_dutch_surname_dataset"]:
        try:
            df = pd.read_excel(config["path_for_dutch_surname_excell"])
            dataset_1 = make_dataset_from_lastname(df)
            update_dataset = dataset_1.remove_columns('__index_level_0__')
            update_dataset = split_to_train_test_validation(update_dataset)
            update_dataset = await update_feature_of_new_dataset(update_dataset,config["existing_conll2003_dataset"][0])
            await data_push_to_hub(update_dataset,config["dataset_name_for_surname_dataset"])
            time.sleep(5)
            total_datasets.append(config["dataset_name_for_surname_dataset"])
        except Exception as e:
            # Catch any other exceptions
            print(f"An unexpected error occurred 1 : {e}")
    
    if config["make_dutch_firstname_lastname_dataset"]:
        try:
            df_next = pd.read_excel(config["path_for_dutch_firstname_lastname_excell"])
            dataset_2 = get_list_from_firstname_lastname_excell(df_next)
            update_dataset = dataset_2.remove_columns('__index_level_0__')
            update_dataset = split_to_train_test_validation(update_dataset)
            print("Existing Conll2003 Dataset is :",config["existing_conll2003_dataset"][0])
            update_dataset = await update_feature_of_new_dataset(update_dataset,config["existing_conll2003_dataset"][0])
            await data_push_to_hub(update_dataset,config["dataset_name_for_firstname_lastname_dataset"])
            time.sleep(5)
            total_datasets.append(config["dataset_name_for_firstname_lastname_dataset"])
        except Exception as e:
            # Catch any other exceptions
            print(f"An unexpected error occurred 2 : {e}")

    if config["make_dutch_city_street_dataset"]:
        try:
            df_city_street = pd.read_csv(config["path_for_dutch_city_street_excell"])
            city_street_dataset = get_list_from_city_street_csv(df_city_street)
            print("city_street_dataset is:",city_street_dataset)
            update_dataset = split_to_train_test_validation(city_street_dataset)
            update_dataset = await update_feature_of_new_dataset(update_dataset,config["existing_conll2003_dataset"][0])
            print("KKKKKKKKKKK")
            await data_push_to_hub(update_dataset,config["dataset_name_for_city_street_dataset"])
            time.sleep(5)
            total_datasets.append(config["dataset_name_for_city_street_dataset"])
        except Exception as e:
            # Catch any other exceptions
            print(f"An unexpected error occurred 3 : {e}")

    if config["load_from_excell_or_csv_and_make_new_dataset"]:
        try:
            split_path = config["path_for_excell_or_csv"].rsplit(".", 1)
            file_extension = split_path[1] if len(split_path) > 1 else ''
            print("file extension is:",file_extension)

            if file_extension.lower() == "csv":
                print("csv file printed")
                df = pd.read_csv(config["path_for_excell_or_csv"])

            elif file_extension.lower() == "xlsx":
                print("excell file printed")
                df = pd.read_excel(config["path_for_excell_or_csv"])
            else:
                print("Only CSV and Excell File is supported. Also you may have input wrong path in config.json")
            
            dataset = make_dataset_from_excell_or_csv(df,config)
            update_dataset = dataset.remove_columns('__index_level_0__')
            # print("updated dataset is is:",update_dataset)
            # return "HELLO"
            update_dataset = split_to_train_test_validation(update_dataset)
            update_dataset = await update_feature_of_new_dataset(update_dataset,config["existing_conll2003_dataset"][0])
            print("KKKKKKKKKKK")
            await data_push_to_hub(update_dataset,config["dataset_name_for_load_from_excell_or_csv"])
            time.sleep(5)
            total_datasets.append(config["dataset_name_for_load_from_excell_or_csv"])
        except Exception as e:
            # Catch any other exceptions
            print(f"An unexpected error occurred 3 : {e}")

    print("totaldataset is:",total_datasets)

    time.sleep(5)

    if total_datasets:
        if config["append_above_generated_dataset"]:
            loaded_datasets = {split: [] for split in ['train', 'test', 'validation']}
            # Load each dataset and organize them by their splits
            for repo_id in total_datasets:
                time.sleep(5)
                dataset = load_dataset(repo_id)
                for split in dataset.keys():
                    loaded_datasets[split].append(dataset[split])

            # Concatenate the datasets for each split
            concatenated_datasets = {split: concatenate_datasets(loaded_datasets[split]) for split in loaded_datasets}
            update_dataset = DatasetDict(concatenated_datasets)
            print(update_dataset)
            await data_push_to_hub(update_dataset,config["dataset_name_for_append_above_generated_dataset"])
        else:
            print("Generated Datasets are not appended with each other")
    else:
        print("The dataset is empty. No operations to perform.")

        
    if config["append_existing_datasets"] or config["AppendWithConll2003Dataset"]:

        def load_and_concatenate_datasets(dataset_names,split):
            datasets = [load_dataset(name,split=split) for name in dataset_names]
            time.sleep(1)
            return concatenate_datasets(datasets)

        if total_datasets and config["append_existing_datasets"] and config["AppendWithConll2003Dataset"]:
            dataset_names = [config["dataset_name_for_append_above_generated_dataset"]] +config["existing_dataset_names_list"] + config["existing_conll2003_dataset"]
        elif total_datasets and config["append_existing_datasets"]:
            dataset_names = [config["dataset_name_for_append_above_generated_dataset"]] +config["existing_dataset_names_list"]
        elif config["append_existing_datasets"] and config["AppendWithConll2003Dataset"]:
            dataset_names = config["existing_dataset_names_list"] + config["existing_conll2003_dataset"]
        elif total_datasets and config["AppendWithConll2003Dataset"]:
            dataset_names = [config["dataset_name_for_append_above_generated_dataset"]] + config["existing_conll2003_dataset"]
        elif config["existing_dataset_names_list"]:
            dataset_names = config["existing_dataset_names_list"]
        else:
            dataset_names = config["existing_conll2003_dataset"]
        
        train_dataset = load_and_concatenate_datasets(dataset_names,"train")
        validation_dataset = load_and_concatenate_datasets(dataset_names,"validation")
        test_dataset = load_and_concatenate_datasets(dataset_names,"test")
        combined_dataset = concatenate_datasets([train_dataset,test_dataset,validation_dataset])
        update_dataset = split_to_train_test_validation(combined_dataset)
        await data_push_to_hub(update_dataset,config["final_dataset_name_after_appending_with_existing_or_conll2003_dataset"])

if __name__ == "__main__":
    asyncio.run(main())



