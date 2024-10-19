from pandas._libs import json


def to_hex(char_file):
    all_content, train_content, test_content = [], [], []
    with open(char_file, 'r') as f:
        content = f.read()
    train_list = list(content[1000:])
    test_list = list(content[:1000])
    for ch in train_list:
        hex_value = hex(ord(ch))[2:].upper()
        train_content.append(hex_value)
    for ch in test_list:
        hex_value = hex(ord(ch))[2:].upper()
        test_content.append(hex_value)
    all_content = test_content + train_content
    return all_content, train_content, test_content


if __name__ == '__main__':
    char_file = "../datasets/char_all_15000.txt"
    all_content, train_content, test_content = to_hex(char_file)
    print(train_content[:2])
    print(test_content[:2])
    print(all_content[:2])
    with open('train_unis.json', 'w') as f:
        f.write(json.dumps(train_content))
    with open('val_unis.json', 'w') as f:
        f.write(json.dumps(test_content))
    with open('trian_val_all_characters.json', 'w') as f:
        f.write(json.dumps(all_content))
