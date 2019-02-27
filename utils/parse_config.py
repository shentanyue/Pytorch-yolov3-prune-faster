#coding=utf-8
# 解析yolov3的cfg文件，并将每个块存储为dict。块的属性及其值作为键值对存储在字典中。
# 当我们解析cfg时，我们会继续将这些由block代码中的变量表示的dicts附加到列表中blocks。我们的函数将返回此块。
def parse_model_config(cfgfile_path):
    file = open(cfgfile_path, 'r')
    lines = file.read().split('\n')  # 以回车符为标准读取内容
    lines = [x for x in lines if len(x) > 0]  # 去除空行
    lines = [x for x in lines if x[0] != '#']  # 去除#开头的语句
    lines = [x.rstrip().lstrip() for x in lines]  # 去除语句两边的空格

    block = {}  # 储存每一个网络块
    blocks = []  # 最后返回的模型列表

    for line in lines:
        if line[0] == '[':
            if len(block) != 0:  # 一直到下一个‘[’,进入该循环，储存进去
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            value = value.strip()
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)  # 用来存储最后一个块

    return blocks


def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3,4,5,6,7,8,9'
    options['num_workers'] = '20'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options



if __name__ == '__main__':
    cfgfile = "../cfg/yolov3.cfg"
    blocks = parse_model_config(cfgfile)
    print(blocks)
