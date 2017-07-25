# ライブラリの取り込み
import pandas as pd
from sklearn import svm, metrics, cross_validation

#　アヤメのCSVのデータを取り出す
csv = pd.read_csv('iris.csv')

#　アヤメのCSVのデータを取り出す
csv_data = csv[["SepalLength","SepalWidth","PetalLength","PetalWidth"]]
csv_label = csv["Name"]

#　学習用のデータとテスト用に分割する
train_data, test_data, train_label, test_label = \
cross_validation.train_test_split(csv_data, csv_label)

clf = svm.SVC()
clf.fit(train_data, train_label)
pre = clf.predict(test_data)

# 正解率を求める
ac_score = metrics.accuracy_score(test_label, pre)
print("正解率=", ac_score)
