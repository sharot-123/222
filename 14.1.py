#coding:utf-8
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score,f1_score
from sklearn.metrics import  classification_report
import sklearn.svm as svm
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
#plt.rcParams['font.sans-serif'] = ['simhei']
plt.rcParams['axes.unicode_minus'] = False

df= pd.read_csv('fenshu20240112.csv', encoding='ISO-8859-1',delimiter=',')
#print(df.head())

df=df[['attend_score','hw_score','ans_score','discus_score','smart_cr','s&v_num','hw_num','ans_num','discus_num','fe_tag','explan_t']]
#归一化
# df['attend_score'] = minmax_scale(df['attend_score'])
# df['hw_score'] = minmax_scale(df['hw_score'])
# df['ans_score'] = minmax_scale(df['ans_score'])
# df['discus_score'] = minmax_scale(df['discus_score'])
#df['smart_cr'] = minmax_scale(df['smart_cr'])
df['s&v_num'] = minmax_scale(df['s&v_num'])
df['hw_num'] = minmax_scale(df['hw_num'])
df['ans_num'] = minmax_scale(df['ans_num'])
df['discus_num'] = minmax_scale(df['discus_num'])
#df['explan_t'] = minmax_scale(df['explan_t'])

# 对标识进行数值化处理
le=preprocessing.LabelEncoder()
df['flag_2'] = le.fit_transform(df['fe_tag'])
#print(df.head(5))

X = df[['attend_score','hw_score','ans_score','discus_score','smart_cr','ans_num','s&v_num']]#'hw_num','discus_num','explan_t'
cols=['attend_score','hw_score','ans_score','discus_score','smart_cr','s&v_num','ans_num']
Y = df['flag_2']
x_train,x_test,y_train,y_test=train_test_split(X,Y,train_size=0.7,test_size=0.3)

# print(y_train.value_counts())
# print(y_test.value_counts())

# #向量机
# #比较核函数（kernel)
# for kernel in ["linear","poly","rbf","sigmoid"]:
#     clf = SVC(kernel = kernel
#              ,gamma="auto"
#              ,degree = 1
#              ,cache_size = 5000  #设定越大表示算法允许使用越多的内存来进行计算
#              ).fit(x_train, y_train)
#     ypred = clf.predict(x_test)  #获取模型的预测结果
    
#     accuracy = clf.score(x_test,y_test)#接口score返回的是准确度accuracy
#     precision = precision_score(y_test,ypred,average = "macro")
#     recall = recall_score(y_test, ypred,average = "macro")
#     f1 = f1_score(y_test,ypred,average = "macro")
#     print("%s 's testing accuracy is %f,precision is %f, recall is %f', f1 is %f" %(kernel,accuracy,precision,recall,f1))
# #poly，rbf，sigmoid正确率相当

# #gamma
# import matplotlib.pyplot as plt
# gamma_range = np.linspace(0.01,10,100) #在指定的起始点和结束点之间生成等间隔的数值序列
# precisionall = []
# recallall = []
# f1all = []
# accuracyall = []
# for gamma in gamma_range:
#     clf = SVC(kernel = "rbf",gamma=gamma,cache_size = 5000).fit(x_train, y_train)
#     pred = clf.predict(x_test)
#     accuracy = clf.score(x_test,y_test)
#     precision = precision_score(y_test, pred, average = "macro")
#     recall = recall_score(y_test, pred, average = "macro")
#     f1 = f1_score(y_test,pred, average = "macro")
#     precisionall.append(precision)
#     recallall.append(recall)
#     f1all.append(f1)
#     accuracyall.append(accuracy)
#     #print("under gamma %f, testing accuracy is %f,precision is %f,recall is %f',f1 is %f" %(gamma,accuracy,precision,recall,f1))
     
# print(max(accuracyall),gamma_range[accuracyall.index(max(accuracyall))])
# # plt.figure()
# plt.figure(figsize=[20,5])
# plt.plot(gamma_range,precisionall,c="blue",label="precision")
# plt.plot(gamma_range,recallall,c="red",label="recall")
# plt.plot(gamma_range,f1all,c="black",label="f1")
# plt.plot(gamma_range,accuracyall,c="orange",label="accuracy")
# plt.legend()
# plt.show()
# #结论：rbf的gamma=9.596363636363636,精度较优

# #C值
# import matplotlib.pyplot as plt
# C_range = np.linspace(0.01,30,30)
# precisionall = []
# recallall = []
# f1all = []
# accuracyall = []
# for C in C_range:
    
#     clf = SVC(kernel = "rbf",gamma=9.596363636363636,C=C,cache_size = 5000).fit(x_train, y_train)
#     pred = clf.predict(x_test)
#     accuracy = clf.score(x_test,y_test)
#     precision = precision_score(y_test,pred,average = "macro")
#     recall = recall_score(y_test, pred,average = "macro")
#     f1 = f1_score(y_test,pred,average = "macro")
#     accuracyall.append(accuracy)
#     recallall.append(recall)
#     precisionall.append(precision)
#     f1all.append(f1)
#     #print("under C %f, testing accuracy is %f,precision is %f,recall is %f', f1 is %f" %(C,accuracy,precision,recall,f1))
   
    
# print(max(accuracyall),C_range[accuracyall.index(max(accuracyall))])
# plt.figure()
# plt.plot(C_range,precisionall,c="blue",label="precision")
# plt.plot(C_range,recallall,c="red",label="recall")
# plt.plot(C_range,f1all,c="black",label="f1")
# plt.plot(C_range,accuracyall,c="orange",label="accuracy")
# plt.legend()
# plt.show()
# #0.6814814814814815 c=2.078275862068965

 
#  #decision_function_shape
# for decision_function_shape in ["ovo","ovr"]:
#     clf = SVC(kernel = "rbf"
#              ,gamma=9.596363636363636
# #            ,c=2.078275862068965
#              ,cache_size = 5000  #设定越大表示我们算法允许使用越多的内存来进行计算
#              ,decision_function_shape = decision_function_shape
#              ).fit(x_train, y_train)
#     pred = clf.predict(x_test)  #获取模型的预测结果
    
#     accuracy = clf.score(x_test,y_test)#接口score返回的是准确度accuracy
#     precision = precision_score(y_test,pred,average = "macro")
#     recall = recall_score(y_test, pred,average = "macro")
#     f1 = f1_score(y_test,pred,average = "macro")
#     print("%s 's testing accuracy is %f,precision is %f, recall is %f', f1 is %f" %(decision_function_shape,accuracy,precision,recall,f1))
    
# 汇总参数&显示混淆矩阵&评估报告
# clf = SVC(kernel = "rbf",gamma=9.596363636363636
# #           ,C=2.078275862068965
#           ,cache_size = 5000,decision_function_shape="ovr"
#          ).fit(x_train, y_train)
# ypred = clf.predict(x_test)
# accuracy = clf.score(x_test,y_test)
# precision = precision_score(y_test,ypred,average = "macro")
# recall = recall_score(y_test,ypred,average = "macro")
# f1 = f1_score(y_test,ypred,average = "macro")
# print("testing accuracy %f,presicion is %f,recall is %f,f1 is %f" % (accuracy,precision,recall,f1))
# #testing accuracy 0.614815,presicion is 0.744877,recall is 0.471673,f1 is 0.452735

from sklearn.metrics import confusion_matrix

# # 混淆矩阵，对角线为正确，其他为误分类
# confusion_matrix(y_test,ypred,labels=(2,1,0))
# from sklearn.metrics import ConfusionMatrixDisplay
# from matplotlib import pyplot as plt
# # 使用sklearn工具包中的ConfusionMatrixDisplay可视化混淆矩阵，参考plot_confusion_matrix
# classes = ('high','medium','low')
# disp = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test,ypred,labels=(2,1,0)),display_labels = classes)
# disp.plot(
#     include_values = True,            # 混淆矩阵每个单元格上显示具体数值
#     cmap = plt.get_cmap("Greys"),                 # 不清楚啥意思，没研究，使用的sklearn中的默认值
#     ax = None,                        # 同上
#     xticks_rotation = "horizontal",   # 同上
#     values_format = "d"# 显示的数值格式
# )
# plt.show()

# #评估报告
# print(classification_report(y_test,ypred))

##决策树与随机森林

from sklearn import tree

# clf = tree.DecisionTreeClassifier(max_depth=7,criterion="entropy")  #C4.5
# clf = clf.fit(x_train,y_train)  #fit
# ypred = clf.predict(x_test)
# # score = clf.score(x_test,y_test)  #返回预测准确度accuracy
# # print(score)
# # print(clf.feature_importances_)

# # 混淆矩阵，对角线为正确，其他为误分类
# confusion_matrix(y_test,ypred,labels=(2,1,0))
# from sklearn.metrics import ConfusionMatrixDisplay
# from matplotlib import pyplot as plt
# # 使用sklearn工具包中的ConfusionMatrixDisplay可视化混淆矩阵，参考plot_confusion_matrix
# classes = ('high','medium','low')
# disp = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test,ypred,labels=(2,1,0)),display_labels = classes)
# disp.plot(
#     include_values = True,            # 混淆矩阵每个单元格上显示具体数值
#     cmap = plt.get_cmap("Greys"),                 # 不清楚啥意思，没研究，使用的sklearn中的默认值
#     ax = None,                        # 同上
#     xticks_rotation = "horizontal",   # 同上
#     values_format = "d"# 显示的数值格式
# )
# plt.show()

feature_name = ['出勤率','作业分','课堂答题分','小组活动分','是否智慧教室','视频及讲故事次数','作业次数','课堂答题次数','小组活动次数','知识点讲解时间占用百分比']

# #print(*zip(feature_name,clf.feature_importances_))

# #棒棒糖图
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# df = pd.DataFrame({'group':cols, 'values':clf.feature_importances_})
# my_dpi=96    #画出来图形会窄一点
# plt.figure(figsize=(480/my_dpi, 480/my_dpi), dpi=my_dpi)
# ordered_df = df.sort_values(by='values')
# plt.figure(figsize=(15, 5))
# plt.hlines(y=range(len(cols)), xmin=0, xmax=ordered_df['values'], color='skyblue')
# plt.plot(ordered_df['values'], range(len(cols)), "o")
# plt.yticks(range(len(cols)), ordered_df['group'])
# plt.title("Lolipop chart for feature importances", loc='left')
# plt.xlabel('Importance of the variable')
# plt.ylabel('Variable')
# plt.show()


## #最优超参数曲线进行判断最优剪枝参数
# #max_depth为7
# import matplotlib.pyplot as plt
# test = []
# for i in range(20):
#     clf = tree.DecisionTreeClassifier(max_depth=i+1
#                                       ,criterion="entropy"
#                                       ,random_state=0
#                                       ,splitter="random"
#                                       )
#     clf = clf.fit(x_train, y_train)
#     score = clf.score(x_test, y_test)
#     test.append(score)
# plt.plot(range(1,21),test,color="red",label="max_depth")
# plt.legend()
# plt.show()
# #从目前来看，是7

# from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
# evaluates = ['accuracy','precision','recall','f1']
# y_train_pred = clf.predict(x_train)
# y_test_pred = clf.predict(x_test)
# # accuracy
# train_accuracy = clf.score(x_train, y_train)
# test_accuracy = clf.score(x_test, y_test)
# # precision
# train_precision_macro = precision_score(y_train, y_train_pred,average="macro")
# test_precision_macro = precision_score(y_test, y_test_pred,average="macro")
# train_precision_micro = precision_score(y_train, y_train_pred,average="micro")
# test_precision_micro = precision_score(y_test, y_test_pred,average="micro")
# # recall
# train_recall_macro = recall_score(y_train, y_train_pred,average="macro")
# test_recall_macro = recall_score(y_test, y_test_pred,average="macro")
# train_recall_micro = recall_score(y_train, y_train_pred,average="micro")
# test_recall_micro = recall_score(y_test, y_test_pred,average="micro")
# # f1
# train_f1_macro = f1_score(y_train, y_train_pred,average="macro")
# test_f1_macro = f1_score(y_test, y_test_pred,average="macro")
# train_f1_micro = f1_score(y_train, y_train_pred,average="micro")
# test_f1_micro = f1_score(y_test, y_test_pred,average="micro")

# print("训练集macro:accuracy is %f,precision is %f,recall is %f,f1 is %f"%(train_accuracy,train_precision_macro,train_recall_macro,train_f1_macro))
# print("测试集macro:accuracy is %f,precision is %f,recall is %f,f1 is %f"%(test_accuracy,test_precision_macro,test_recall_macro,test_f1_macro))
# print("训练集micro:accuracy is %f,precision is %f,recall is %f,f1 is %f"%(train_accuracy,train_precision_micro,train_recall_micro,train_f1_micro))   
# print("测试集micro:accuracy is %f,precision is %f,recall is %f,f1 is %f"%(test_accuracy,test_precision_micro,test_recall_micro,test_f1_micro))


##随机森林
from sklearn.ensemble import RandomForestClassifier#随机森林分类器
from sklearn.model_selection import GridSearchCV#网格搜索
from sklearn.model_selection import cross_val_score#交叉验证

# clf = RandomForestClassifier(n_estimators=20,random_state=3).fit(x_train,y_train) #决策树的数量、
# score_pre = cross_val_score(rfc,df[cols],df['flag_2'],cv=10).mean()#交叉验证，全数据集
# print(score_pre)

# 定义参数网格
# param_grid = {
#          'n_estimators': [i for i in range(0, 100)],
#      'random_state': [i for i in range(3, 10, 1)]
# }

# # 使用GridSearchCV进行参数调优
# grid_search = GridSearchCV(clf, param_grid, cv=3)
# grid_search.fit(x_train, y_train)
#
# # 输出最佳参数
# print("Best Parameters:", grid_search.best_params_)


##神经网络
from sklearn.neural_network import MLPClassifier # 导入神经网络

# X = np.array(X)
# Y = np.array(Y)
# x_train,x_test,y_train,y_test=train_test_split(X,Y,train_size=0.7,test_size=0.3)

# clf = MLPClassifier(solver='lbfgs',activation='tanh',
#                    hidden_layer_sizes=(20,10,20), random_state=0,alpha=1)
# clf.fit(x_train,y_train) #拟合

# 定义参数网格
# param_grid = {
#     'random_state': [i for i in range(3, 10, 1)],
#     'alpha': [1e-5,1e-4,1e-3, 1e-2, 0.1, 1, 100,1000,10000,100000]
# }

# # 使用GridSearchCV进行参数调优
# grid_search = GridSearchCV(clf, param_grid, cv=3)
# grid_search.fit(x_train, y_train)
#
# # 输出最佳参数
# print("Best Parameters:", grid_search.best_params_)


# y_predict = clf.predict(x_test)
# score=clf.score(x_test,y_test,sample_weight=None)
# print('y_predict = ')  
# print(y_predict)  
# print('y_test = ')
# print(y_test)    
# print('Accuracy:',score)
# print('layers nums :',clf.n_layers_)
#隐藏层为5

# # alpha为正则化强度,调参
# alphas = [1e-5,1e-4,1e-3, 1e-2, 0.1, 1, 100,1000,10000,100000]
# for alpha in alphas:
#     ann_model = MLPClassifier(hidden_layer_sizes=[20,10], activation='tanh', solver='lbfgs', random_state=0,
#                              alpha=alpha)
#     ann_model.fit(x_train, y_train)
#     print('alpha={}，准确率：{:.3f}'.format(alpha, ann_model.score(x_test, y_test)))
# #alphas目前看来取1较好

##xgboost
import xgboost as XGBC
import warnings
from xgboost import XGBClassifier as XGBC
import shap


# X = df[['attend_score','hw_score','ans_score','discus_score','smart_cr','s&v_num','hw_num','ans_num','explan_t','discus_num']]
# cols=['attend_score','hw_score','ans_score','discus_score','smart_cr','s&v_num','hw_num','ans_num','explan_t','discus_num']
# Y = df['flag_2']
# x_train,x_test,y_train,y_test=train_test_split(X,Y,train_size=0.7,test_size=0.3)

# # 设置模型默认参数
# clf=XGBC(
#     #     通用参数
#     silent=1,
#     nthread=4,# cpu 线程数 默认最大
#     learning_rate= 0.3, # 如同学习率
#     min_child_weight=1,
#     # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#     #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
#     max_depth=6, # 构建树的深度，越大越容易过拟合
#     gamma=0,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
#     subsample=1, # 随机采样训练样本 训练实例的子采样比
#     colsample_bytree=1, # 生成树时进行的列采样
#     reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
#     reg_alpha=0, # L1 正则项参数
#     scale_pos_weight=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
#     objective= 'multi:softmax', #多分类的问题 指定学习任务和相应的学习目标
#     num_class=3, # 类别数，多分类与 multisoftmax 并用
#     n_estimators=100, #树的个数
#     seed=1000#随机种子
# ).fit(x_train, y_train)

# ##调参
from sklearn.model_selection import GridSearchCV

# 定义参数网格
# param_grid = {
#     'learning_rate': [0.3, 0.1, 0.01],
#      'n_estimators': [i for i in range(0, 100)],
#      'colsample_bytree': [i / 10.0 for i in range(6, 10)],
#      'reg_lambda': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 100]
#      'max_depth': [i for i in range(3, 10, 1)],
#       'min_child_weight': [i for i in range(1, 6, 1)],
#     'gamma': [i / 10.0 for i in range(0, 5)],
#      'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
# }

# # 使用GridSearchCV进行参数调优
# grid_search = GridSearchCV(clf, param_grid, cv=3)
# grid_search.fit(x_train, y_train)
#
# # 输出最佳参数
# print("Best Parameters:", grid_search.best_params_)
##Best Parameters: {'gamma': 0.4, 'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 20}

# ypred = clf.predict(x_test)

from sklearn.metrics import accuracy_score
# accuracy_pre = accuracy_score(y_test, ypred)
# print("Accuracy: %.2f%%" % (accuracy_pre * 100.0))



# from sklearn.model_selection import GridSearchCV 
# from sklearn.model_selection import cross_val_score  
# param_test1 = {
#  'max_depth':range(3,10,1),
#  'min_child_weight':range(1,6,1)
# }
# gsearch1 = GridSearchCV(estimator = XGBC(booster = 'gbtree',nthread =4,silent =1,num_feature =10,learning_rate =0.1, n_estimators=20,
#                                          max_depth=4,min_child_weight=1, gamma=0.4, subsample=1,colsample_bytree=1,
#                                          objective= 'multi:softmax',scale_pos_weight=1, seed=1000,num_class = 3),
#                         param_grid = param_test1,scoring=None,n_jobs=-1,cv=3)
# gsearch1.fit(x_train,y_train)
# print(gsearch1.cv_results_, gsearch1.best_params_,gsearch1.best_score_)
##max_depth=4,min_child_weight=2
#

# param_test4 = {
#     'subsample': [i / 10.0 for i in range(6, 10)],
#     'colsample_bytree': [i / 10.0 for i in range(6, 10)]
# }
#
# gsearch4 = GridSearchCV(
#     estimator=XGBC(booster = 'gbtree',nthread = 4,silent = 1,num_feature =10,learning_rate=0.1, n_estimators=20,
#                    max_depth=4, min_child_weight=2, gamma=0.4,subsample=1, colsample_bytree=1, objective='multi:softmax',
#                             scale_pos_weight=1, seed=1000,num_class =3), param_grid=param_test4, scoring='accuracy', n_jobs=-1,cv=3)
#
# gsearch4.fit(x_train,y_train)
# print(gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_)
# ##colsample_bytree': 0.7, 'subsample': 0.6

# param_test5 = {
#  'reg_lambda':[1e-5,1e-4,1e-3, 1e-2, 0.1, 1, 100]
# }
# gsearch4 = GridSearchCV(
#     estimator=XGBC(booster = 'gbtree',nthread =4,silent =1,num_feature =10,learning_rate=0.1, n_estimators=20,
#                    max_depth=4, min_child_weight=2, gamma=0.4,subsample=0.6, colsample_bytree=0.7, objective='multi:softmax',
#                             scale_pos_weight=1, seed=1000,num_class =3), param_grid=param_test5, scoring='accuracy', n_jobs=-1,cv=3)
#
# gsearch4.fit(x_train,y_train)
# print(gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_)
# ##'reg_lambda': 1

##汇总调参之后

# #加入order_logic的xgboost代码
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# X_train = pd.read_csv('X_train_best.csv')
# y_train = pd.read_csv('y_train_best.csv').values.ravel()
# X_test = pd.read_csv('X_test_best.csv')
# y_test = pd.read_csv('y_test_best.csv').values.ravel()

# 自定义有序损失函数
def ordered_logistic_loss(y_true, y_pred):
    """
    自定义有序逻辑回归损失函数
    """
    y_true = y_true.astype(int)
    num_classes = 3  # 明确类别数
    num_samples = len(y_true)

    # 确保 y_pred 是一维数组
    if y_pred.ndim == 2:
        # 这里假设取第一个类别的预测分数作为示例，你可以根据实际情况调整
        y_pred = y_pred[:, 0]

    # 初始化概率矩阵
    probas = np.zeros((num_samples, num_classes))

    # 计算每个类别的概率
    probas[:, 0] = 1 / (1 + np.exp(-y_pred))
    probas[:, 2] = 1 - 1 / (1 + np.exp(-y_pred))
    probas[:, 1] = probas[:, 0] - (1 / (1 + np.exp(-(y_pred + 1))))

    # 确保概率值在合理范围内
    probas = np.clip(probas, 1e-15, 1 - 1e-15)

    # 计算对数概率
    log_probas = np.log(probas)

    # 计算损失
    loss = -np.mean(log_probas[np.arange(num_samples), y_true])

    # 计算梯度和海森矩阵
    grad = np.zeros((num_samples, num_classes))
    hess = np.zeros((num_samples, num_classes))
    for i in range(num_samples):
        for j in range(num_classes):
            if j == y_true[i]:
                grad[i, j] = probas[i, j] - 1
            else:
                grad[i, j] = probas[i, j]
            hess[i, j] = probas[i, j] * (1 - probas[i, j])
    grad = grad.ravel()
    hess = hess.ravel()
    return grad, hess
#
# #初始化 XGBClassifier 模型
clf = XGBClassifier(
# #     通用参数
  #silent = 0,  # 设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
  nthread = 4,  # cpu 线程数 默认最大
  learning_rate = 0.1,  # 如同学习率
  min_child_weight = 2,
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
# 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
  max_depth = 4,  # 构建树的深度，越大越容易过拟合
  gamma = 0.4,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
  subsample = 0.6,  # 随机采样训练样本 训练实例的子采样比
  colsample_bytree = 0.7,  # 生成树时进行的列采样
  reg_lambda = 1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
  reg_alpha = 0.001,  # L1 正则项参数
  #scale_pos_weight = 2,  # 如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
  objective = 'ordered_logistic_loss',  # 多分类的问题 指定学习任务和相应的学习目标
  num_class = 3,  # 类别数，多分类与 multisoftmax 并用
  n_estimators = 20,  # 树的个数
  seed = 1000  # 随机种子
)

# 进行预测
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# 计算准确率
train_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)

# 定义评估指标和平均方法
evaluates = ['macro', 'micro']
metrics = {
    'precision': precision_score,
    'recall': recall_score,
    'f1': f1_score
}

# 循环计算并打印评估指标
for avg_type in evaluates:
    # train_metrics = {}
    test_metrics = {}
    for metric_name, metric_func in metrics.items():
        # train_metrics[metric_name] = metric_func(y_train, y_train_pred, average=avg_type)
        test_metrics[metric_name] = metric_func(y_test, y_test_pred, average=avg_type)

    # print(f"训练集{avg_type}: accuracy is {train_accuracy:.6f}, precision is {train_metrics['precision']:.6f}, recall is {train_metrics['recall']:.6f}, f1 is {train_metrics['f1']:.6f}")
    print(f"测试集{avg_type}: accuracy is {test_accuracy:.6f}, precision is {test_metrics['precision']:.6f}, recall is {test_metrics['recall']:.6f}, f1 is {test_metrics['f1']:.6f}")

# from xgboost import plot_importance
# plot_importance(clf)
# plt.show()

# #混淆矩阵
# from sklearn.metrics import confusion_matrix

# ypred = clf.predict(x_test)

# # 混淆矩阵，对角线为正确，其他为误分类
# confusion_matrix(y_test,ypred,labels=(2,1,0))
# from sklearn.metrics import ConfusionMatrixDisplay
# from matplotlib import pyplot as plt
# # 使用sklearn工具包中的ConfusionMatrixDisplay可视化混淆矩阵，参考plot_confusion_matrix
# classes = ('high','medium','low')
# disp = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test,ypred,labels=(2,1,0)),display_labels = classes)
# disp.plot(
#     include_values = True,            # 混淆矩阵每个单元格上显示具体数值
#     cmap = plt.get_cmap("Greys"),                 # 不清楚啥意思，没研究，使用的sklearn中的默认值
#     ax = None,                        # 同上
#     xticks_rotation = "horizontal",   # 同上
#     values_format = "d"# 显示的数值格式
# )
# plt.show()

#shap分析
import shap
shap.initjs()

#shap计算
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(x_train)
#print(np.array(shap_values).shape)#(3, 314, 9)

# #特征统计值
# shap.summary_plot(shap_values,df[cols],cmap = plt.get_cmap("gist_gray"))
#蜂窝图
shap.summary_plot(shap_values[0],x_train, max_display=20)

# # 训练集第39个样本对于输出结果的SHAP解释
# print(x_train.iloc[39,:])
# print(y_train.iloc[39])
# shap.force_plot(explainer.expected_value[0], shap_values[0][39,:], x_train.iloc[39,:],matplotlib = True)
##决策图训练样本39-50
# shap.decision_plot(explainer.expected_value[0],shap_values[0][39:50,:],feature_names = cols)

# 部分依赖图分析：单个特征与模型预测结果的关系hw_score
# shap.dependence_plot('hw_score', shap_values[2], x_train[cols],interaction_index = None,show = False,cmap = plt.get_cmap("gist_gray"))
# plt.tick_params(labelsize = 15)
# plt.xlabel("homework score",fontsize = 18)  
# plt.ylabel("SHAP value for\homework score",fontsize=18)
# plt.axhline(y=0,ls="--",c="grey") #添加水平直线
# plt.axvline(x=0.86,ls="--",c="grey")#添加垂直直线
# plt.annotate('safe point:(0.86,0)',xy=(0.86,0),xytext=(0.5,0.5),weight='light',color='black',fontsize=10,fontproperties="simhei",
#   arrowprops=dict(arrowstyle='->',connectionstyle='arc3',color='black'),
#   bbox=dict(boxstyle='round,pad=1', fc='grey', ec='k',lw=1 ,alpha=0.3))
# plt.show()
# #第二个特征ans_score
# shap.dependence_plot('ans_score', shap_values[2], x_train[cols],interaction_index = None,show = False,cmap = plt.get_cmap("gist_gray"))
# plt.tick_params(labelsize = 15)
# plt.xlabel("class answer score",fontsize = 18)  
# plt.ylabel("SHAP value for\class answer score",fontsize=18)
# plt.axhline(y=0,ls="--",c="grey") #添加水平直线
# plt.axvline(x=0.88,ls="--",c="grey")#添加垂直直线
# plt.annotate('safe point:(0.88,0)',xy=(0.88,0),xytext=(0.5,0.5),weight='light',color='black',fontsize=10,fontproperties="simhei",
#   arrowprops=dict(arrowstyle='->',connectionstyle='arc3',color='black'),
#   bbox=dict(boxstyle='round,pad=1', fc='grey', ec='k',lw=1 ,alpha=0.3))
# plt.show()
# #第三个特征discus_score
# shap.dependence_plot('discus_score', shap_values[2], x_train[cols],interaction_index = None,show = False,cmap = plt.get_cmap("gist_gray"))
# plt.tick_params(labelsize = 15)
# plt.xlabel("group discuss score",fontsize = 18)  
# plt.ylabel("SHAP value for\group discuss score",fontsize=18)
# plt.axhline(y=0,ls="--",c="grey") #添加水平直线
# plt.axvline(x=0.88,ls="--",c="grey")#添加垂直直线
# plt.annotate('safe point:(0.88,0)',xy=(0.88,0),xytext=(0.5,0.5),weight='light',color='black',fontsize=10,fontproperties="simhei",
#   arrowprops=dict(arrowstyle='->',connectionstyle='arc3',color='black'),
#   bbox=dict(boxstyle='round,pad=1', fc='grey', ec='k',lw=1 ,alpha=0.3))
# plt.show()
# # #第四个特征attend_score
# shap.dependence_plot('attend_score', shap_values[2], x_train[cols],interaction_index = None,show = False,cmap = plt.get_cmap("gist_gray"))
# plt.tick_params(labelsize = 15)
# plt.xlabel("attendence score",fontsize = 18)  
# plt.ylabel("SHAP value for\ attendence score",fontsize=18)
# plt.axhline(y=0,ls="--",c="grey") #添加水平直线
# plt.axvline(x=0.91,ls="--",c="grey")#添加垂直直线
# plt.annotate('safe point:(0.91,0)',xy=(0.91,0),xytext=(0.5,0.5),weight='light',color='black',fontsize=10,fontproperties="simhei",
#   arrowprops=dict(arrowstyle='->',connectionstyle='arc3',color='black'),
#   bbox=dict(boxstyle='round,pad=1', fc='grey', ec='k',lw=1 ,alpha=0.3))
# plt.show()
# #第五个特征smart_cr
# shap.dependence_plot('smart_cr', shap_values[2], x_train[cols],interaction_index = None,show = False,cmap = plt.get_cmap("gist_gray"))
# plt.tick_params(labelsize = 15)
# plt.xlabel("smart classroom",fontsize = 18)  
# plt.ylabel("SHAP value for\ smart classroom",fontsize=18)
# plt.axhline(y=0,ls="--",c="grey") #添加水平直线
# plt.axvline(x=0.91,ls="--",c="grey")#添加垂直直线
# plt.annotate('safe point:(0.91,0)',xy=(0.91,0),xytext=(0.5,0.5),weight='light',color='black',fontsize=10,fontproperties="simhei",
#   arrowprops=dict(arrowstyle='->',connectionstyle='arc3',color='black'),
#   bbox=dict(boxstyle='round,pad=1', fc='grey', ec='k',lw=1 ,alpha=0.3))
# plt.show()
# #第六个特征s&v_num
# shap.dependence_plot('s&v_num', shap_values[2], x_train[cols],interaction_index = None,show = False,cmap = plt.get_cmap("gist_gray"))
# plt.tick_params(labelsize = 15)
# plt.xlabel("story&video",fontsize = 18)  
# plt.ylabel("SHAP value for\ story&video",fontsize=18)
# plt.axhline(y=0,ls="--",c="grey") #添加水平直线
# plt.axvline(x=0.91,ls="--",c="grey")#添加垂直直线
# plt.annotate('safe point:(0.91,0)',xy=(0.91,0),xytext=(0.5,0.5),weight='light',color='black',fontsize=10,fontproperties="simhei",
#   arrowprops=dict(arrowstyle='->',connectionstyle='arc3',color='black'),
#   bbox=dict(boxstyle='round,pad=1', fc='grey', ec='k',lw=1 ,alpha=0.3))
# plt.show()
# #第七个特征ans_num
# shap.dependence_plot('ans_num', shap_values[2], x_train[cols],interaction_index = None,show = False,cmap = plt.get_cmap("gist_gray"))
# plt.tick_params(labelsize = 15)
# plt.xlabel("class answer number",fontsize = 18)  
# plt.ylabel("SHAP value for\ class answer number",fontsize=18)
# plt.axhline(y=0,ls="--",c="grey") #添加水平直线
# plt.axvline(x=0.91,ls="--",c="grey")#添加垂直直线
# plt.annotate('safe point:(0.91,0)',xy=(0.91,0),xytext=(0.5,0.5),weight='light',color='black',fontsize=10,fontproperties="simhei",
#   arrowprops=dict(arrowstyle='->',connectionstyle='arc3',color='black'),
#   bbox=dict(boxstyle='round,pad=1', fc='grey', ec='k',lw=1 ,alpha=0.3))
# plt.show()
