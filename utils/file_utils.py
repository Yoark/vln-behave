import json
import gzip

def write_gz_jsonlines(filename, data_to_write):
    """
    Write to a gzip compressed jsonl file
    filename: 'xxx.jsonl.gz'
    """
    with gzip.open(filename, 'wb') as f:
        for item in data_to_write:
            it_string = json.dumps(item)+'\n'
            it_string = it_string.encode('utf-8')
            f.write(it_string) # type: ignore

def read_gz_jsonlines(filename):
    # *
    data = []
    with open(filename, 'rb') as f:
        for args in map(json.loads, gzip.open(f)):
            data.append(args)
    return data

def save_jsonl(filename, dataset):
    # *
    import json
    with open(filename, 'wb') as f:
        for item in dataset:
            it_string = json.dumps(item)+'\n'
            it_string = it_string.encode('utf-8')
            f.write(it_string)

def load_jsonl(filename):
    # *
    data = []
    import json
    with open(filename, 'rb') as f:
        data = [json.loads(item) for item in f]
    return data
def load_json(filename):
    # *
    with open(filename) as f:
        data = json.load(f)
    return data

def save_json(filename, data):
    # *
    with open(filename, 'w') as f:
        json.dump(data, f)
        
def write_to_record_file(data, file_path, verbose=True):
    if verbose:
        print(data)
    record_file = open(file_path, 'a')
    record_file.write(data+'\n')
    record_file.close()

# colormap = ["#fdfdfd", "#1d1d1d", "#ebce2b", "#702c8c", "#db6917", "#96cde6", "#ba1c30", "#c0bd7f”, "#7f7e80", "#5fa641", "#d485b2", "#4277b6", "#df8461", "#463397”, "#e1a11a”, "#91218c”, "#e8e948”, "#7e1510”, "#92ae31”, "#6f340d”, "#d32b1e”, "#2b3514”]
kelly_colors = ['#F2F3F4', '#222222', '#F3C300', '#875692', '#F38400', '#A1CAF1', '#BE0032', '#C2B280', '#848482', '#008856', '#E68FAC', '#0067A5', '#F99379', '#604E97', '#F6A600', '#B3446C', '#DCD300', '#882D17', '#8DB600', '#654522', '#E25822', '#2B3D26']
sasha_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
# *
my_colors = ['#9BBEDB', '#F18C8D', '#C0BF80', '#60A33F', '#D387B7', '#DEA315', '#8C1F88', '#DB6B1D' ]