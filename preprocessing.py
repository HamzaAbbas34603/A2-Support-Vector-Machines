
def read_data():
    with open("bank-additional.csv") as f:
        data = f.readlines()
        result = []
        for item in data[1:]:
            result.append(item.replace('\n','').split(','))
    return result

def numerize_data(data):
    numeric = []
    result = []
    nf = len(data[0])
    for col in data[0]:
        numeric.append([])
    for i,item in enumerate(data):
        f = False
        for col in item:
            if col == 'jesaispas':
                f = True
        if not f:
            result.append(item)
    for item in result:
        print('\t'.join(item))

data = read_data()
numerize_data(data)