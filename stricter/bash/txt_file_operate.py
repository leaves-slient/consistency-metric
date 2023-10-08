def load(file_name, strip=True):
    """
    只能从txt文件中加载数据,默认经过strip

    Returns:
        res: list
    """
    try:
        with open(file_name,'r') as f:
            if strip:
                res = [i.strip() for i in f]
            else:
                res = [i for i in f]
    except:
        with open(file_name,'r',errors='ignore') as f:
            if strip:
                res = [i.strip() for i in f]
            else:
                res = [i for i in f]
    return res

def save(file_name,res):
    """
    只能保存txt数据,res可以为str或list
    """
    with open(file_name,'w') as f:
        if isinstance(res, str):
            f.write(res)
        elif isinstance(res,list):
            for i in res:
                f.write(str(i) + '\n')

def merge_file(a, b):
    res = []
    for i, j in zip(a,b):
        res.append(i+' ||| '+j)
    return res