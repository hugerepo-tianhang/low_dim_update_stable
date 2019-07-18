
import json
a = [("a", 1), ("b", 2)]
with open("test.json", 'w') as fp:
    json.dump(a, fp)


with open("test.json", 'r') as fp:
    b = json.load(fp)

