import pandas as pd

list1 = [1,2,3]
list2 = ['a','b','c']

cols = ['col1', 'col2']

df = pd.DataFrame(
    {'k1': list1,
     'k2': list2,
    })
print(df)
