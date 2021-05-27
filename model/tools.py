import json

def write_vectors_to_file(vectors, filename):
    print("start writing file")
    with open(filename, "w+") as f:
        json.dump(vectors, f)
    print("write the file %s completely..."%(filename))


def read_vectors_from_file(filename):
    print("start reading file",flush=True)
    with open(filename, 'r') as load_f:
        load_dict = json.load(load_f)
    print("reading the file %s completely..." % (filename),flush=True)

    return load_dict


def tranform_key_type(dicts):
    if type(dicts).__name__ != 'dict':
        return dicts
    new_dict = {}
    for key, value in dicts.items():
        new_key = (int)(key)
        new_dict[new_key] = value
    return new_dict