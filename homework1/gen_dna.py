import numpy as np
import pandas as pd

length = int(input('DNA长度：'))
jj = 'ACTG'
lst = np.random.randint(0, 4, size=length)
lst1 = [jj[x] for x in lst]
lst2 = [jj[(x + 2) % 4] for x in lst]

print('list1:', lst1)
print('list2:', lst2)

data = pd.DataFrame([lst1, lst2])
print(data)
data.to_csv('result.csv', header=0, index=0)
print('文件写出成功.')
