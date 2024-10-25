# 用于我们提出的方法的对比
import torch
import numpy as np
from utils import get_geneid_symbol_mapping,get_coverage,normalized,evaluate
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt


# 组方评价网络结构
class jinyu_Net(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(jinyu_Net, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H, bias=True)
        self.linear2 = torch.nn.Linear(H, D_out, bias=False)
        #self.linear3 = torch.nn.Linear(H//10,D_out,bias=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        y_pred = self.linear2(self.relu(self.linear1(x)))
        y_pred = self.sigmoid(y_pred)
        return y_pred  

with open('my_model.pkl', 'rb') as f:
    model = pickle.load(f)   
with open('test_recipes.txt', 'rb') as handle:
    test_x = pickle.load(handle)
handle.close()
with open('test_label.txt', 'rb') as handle:
    test_y = pickle.load(handle)
handle.close()


# 读取显著表达的靶点
file_name = '../Dataset/nodes.txt'
f = open(file_name,'r')
lines = f.readlines()
nodes = list(map(lambda x: x.strip(), lines))
geneid_to_name, name_to_geneid = get_geneid_symbol_mapping("../Dataset/Homo_sapiens.gene_info")
for i in range(len(nodes)):
    nodes[i]=int(name_to_geneid[nodes[i]])
# 获取超图结构(0,1表示->下标表示)
with open('../Output/afterH.txt', 'rb') as handle:
    H = pickle.load(handle)
handle.close()
H = list(set([tuple(hyper) for hyper in H]))
hypergraph = []
for i in range(len(H)):
    hyper = []
    for j in range(len(H[i])):
        if H[i][j]==1:
            hyper.append(j)
    hypergraph.append(hyper)


test_x,test_y = np.array(test_x),np.array(test_y)
test_x = normalized(test_x,len(nodes))
test_x = torch.tensor(get_coverage(test_x,hypergraph))
model.eval()
with torch.no_grad():
    pred_y = model(test_x)

    
fpr, tpr, thresholds = metrics.roc_curve(test_y, pred_y)
auc = metrics.auc(fpr, tpr)
lw = 2
# framework
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='TS-TCM (%0.2f)' % auc)
# kmeans
res2 = [-1.253,6.835,-2.644,9.267,4.753,5.199,-7.088,-14.997,1.254,-0.290,9.106,-0.266,6.296,-17.833,-12.570,1.639,-7.110,-0.178]
res = []
max_value,min_value = max(res2),min(res2)
for data in res2:
    res.append(1-(data-min_value)/(max_value-min_value))  
fpr, tpr, thresholds = metrics.roc_curve(test_y, res)
auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='blue',lw=lw, label='K-means (%0.2f)' % auc)

# spectral
res3 =[-19.589,-11.619,-8.480,-12.879,-19.829,-17.209,17.814,23.890,-12.236,2.170,-16.428,-9.150,-3.799,5.317,5.778,-0.696,6.915,-9.847]
res = []
max_value,min_value = max(res3),min(res3)
for data in res3:
    res.append(1-(data-min_value)/(max_value-min_value))  
fpr, tpr, thresholds = metrics.roc_curve(test_y, res)
auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr-0.005, color='green',lw=lw, label='Spectral (%0.2f)' % auc)

# netwoek distance
res1 = [-3.376,-3.875,-2.956,-3.601,-4.021,-3.906,-4.487,-6.273,-3.319,-4.711,-4.294,-3.625,-2.706,-3.816,-3.959,-4.435,-5.705,-3.716]
res = []
max_value,min_value = max(res1),min(res1)
for data in res1:
    res.append(1-(data-min_value)/(max_value-min_value))  
fpr, tpr, thresholds = metrics.roc_curve(test_y, res)
auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='black',lw=lw, label='PPI Distance (%0.2f)' % auc)

# gcn
res1 = [-0.106,-0.317,-0.009,-0.064,-0.004,-0.124,0.197,0.355,-0.124,-0.086,-0.201,-0.088,-0.198,0.072,-0.063,0.008,0.215,-0.063]
res = []
max_value,min_value = max(res1),min(res1)
for data in res1:
    res.append(1-(data-min_value)/(max_value-min_value))  
fpr, tpr, thresholds = metrics.roc_curve(test_y, res)
auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr+0.005, color='red',lw=lw, label='GCN (%0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves of different methods on prescriptions')
plt.legend(loc="lower right")
plt.savefig('fig1.png')
plt.show()
exit()