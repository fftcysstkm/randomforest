#%%
import pandas as pd
import numpy as np

###############################################################################
#自作のクラス
#■コンストラクタ：環境データとプランクトンデータをデータフレームとしてメンバーに格納。
#■setColForSite():不要なデータ別地点のデータ削除
#■removeChar()：文字列が含まれればNaNに置き換える。
#■interpolateData()：NaNをスプライン補間する。
#■addMoveAve()：移動平均値をデータフレームに追加する。
#■setMicroClass()：プランクトンの細胞数をクラスにする。
###############################################################################
class Damdata:
    #パス設定、シート名渡す。→データフレームを設定。
    #現状、必要な項目だけ残す。
    def __init__(self,path,sh_name1,sh_name2):
        try:
            if sh_name1 != "env":
                raise Exception
            self.path = path
            # df_env-----sh_name1の環境のデータ
            self.df_env = self.readFile(sh_name1)
            self.df_env['date'] = self.df_env['date'].map(pd.to_datetime)#datetime型にする。
            self.df_env.set_index('date',inplace=True)#日付をDataframeのindexにする。

            # df_pla----sh_name2のプランクトンの細胞数のデータ。NaNは消しておく。
            self.df_pla = self.readFile(sh_name2)
            self.df_pla['date'] = self.df_pla['date'].map(pd.to_datetime)
            self.df_pla.set_index('date',inplace=True)
            self.df_pla = self.df_pla['●●●●●●● sp.']#プランクトンは●●●●●（種名）だけ残す。
            self.df_pla = self.df_pla.dropna()#プランクトンデータは欠損値をここで削除

        except AttributeError:
            print('エラーメッセージ')
        

    #ファイル読み込み。↑コンストラクタで使用される。
    def readFile(self,sheet_name):
        try:
            #最初の0~3行目は不要行なのでカット。
            return pd.read_excel(self.path,sheet_name,skiprows=range(4),index_col=None,header = 0)
        except FileNotFoundError:
            print('読み込みファイルエラー')
        
    
    #環境のデータのみ、地点名に応じて不要な列を削除。
    def setColForSite(self,site_name):
        try:
            if site_name == "siteA":#siteAならsiteB関係のデータ削除
                self.df_env.drop(['wtemp_B','wtgradient_B','wtemp_max_B','wtgradient_max_B'],axis=1,inplace=True)
            elif site_name == "siteB":#siteBならsiteA関係のデータ削除
                self.df_env.drop(['wtemp_A','wtgradient_A','wtemp_max_A','wtgradient_max_A'],axis=1,inplace=True)
            else:
                raise Exception
            
            #★★★余分な列削除。リンのうち、LQ式の方はここで削除している。濁度換算の方は生かす。★★★★★
            self.df_env.drop(['elevation','dif_elevation','capacity',
                          '301capacity','inflow','KRYM_intake','BP_outflow',
                          'donduit_outflow','down_outflow','KDIJ_fow',
                          'turnover','humidity','wind_max','wind_direction',
                          'D-T-P_LQ','D-PO4-P_LQ'],axis=1,inplace=True)
            

        except AttributeError :
            print('引数が間違っています。"siteA"か"siteB"のどちらかを入力してください。')


    #文字列除去(NaNに置換)
    def removeChar(self):
        str2nan = lambda x: np.nan if type(x) == str else x#lambda式。データ型が文字列はNaNに置換
        self.df_env = self.df_env.applymap(str2nan)

    #環境データのNaNをスプライン補間（3次）。内挿。元のDFを書き換え。
    #そこそこ時間かかる。
    def interpolateData(self):
        self.df_env.interpolate(method='spline',order=3,limit_area='inside',inplace=True)


    #環境データの移動平均を計算し、元のDFに付け加える。
    def addMoveAve(self):
        #移動平均を計算する項目のリスト = データフレームの項目名
        env_list = list(self.df_env.columns)
        #移動平均の期間のリスト
        duration_list = [2,3,5,7,10,15,20,30]
        for env in env_list:
            for dur in duration_list:
                self.df_env[env+str(dur)+"days"] =self.df_env[env].rolling(window = dur,min_periods=dur,center=False).mean()

    #プランクトンの細胞数をクラス分け。
    def setMicroClass(self):
        def value_to_label(value):
            class_list = [0,35,100]
            if value == class_list[0]:
                return 0
            elif value < class_list[1]:
                return 1
            elif value < class_list[2]:
                return 2
            else:
                return 3
        self.df_pla = self.df_pla.map(value_to_label)

#%%
# RandomForestモデルの木構造の視覚化に必要なパッケージ
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import japanize_matplotlib

###########################################################
#ラベル（プランクトン細胞数）と特徴量データに分ける。
#■入力：データフレーム（ラベルと特徴量データ合体）
#■返り値：ラベルのdf、特徴量のdfの２つ
###########################################################
def sepLabelData(original_df):
    df_label = original_df['Plankton sp.'] #ラベル列抽出
    df_features = original_df.drop('Plankton sp.', axis=1) #ラベル以外の列抽出
    return df_label, df_features
#%%
###########################################################
#重要度のデータフレーム、ファイル（フルパス）を入力
#パスに重要度のグラフを格納
###########################################################
def savePlotImportance(df_importance,file_name):
    plt.figure(figsize=(15,10)) #グラフサイズ
    plt.barh(df_importance['features'], df_importance['importance']) #plt.barh(x, y)#barh横棒グラフ
    plt.title("特徴量の重要度",fontsize=18)
    plt.grid(True)
    plt.rcParams['axes.axisbelow'] = True
    plt.xlabel("重要度",fontsize=16)
    plt.ylabel("特徴量（説明変数）",fontsize=16)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig(file_name,dpi=300)

#初回のランダムフォレストの時に使用
def savePlotImportance_first(df_importance,file_name):
    plt.figure(figsize=(15,30)) #グラフサイズ
    plt.barh(df_importance['features'], df_importance['importance']) #plt.barh(x, y)#barh横棒グラフ
    plt.title("特徴量の重要度",fontsize=18)
    plt.grid(True)
    plt.rcParams['axes.axisbelow'] = True
    plt.xlabel("重要度",fontsize=16)
    plt.ylabel("特徴量（説明変数）",fontsize=16)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig(file_name,dpi=300)

#%%
#クラスの中に、環境データとプランクトンデータが含まれる。
file_path = r"C:\Users\●●●●●●●●●●●●\data.xlsx"
siteB_data = Damdata(file_path,"env","plankton_siteB")#環境＆プランクトンデータをDFとしてセット
siteB_data.setColForSite("siteB")#環境データはsiteB用のもののみ残す。
siteB_data.removeChar()#環境データの文字列削除　→　NaNへ置換
siteB_data.interpolateData()#NaNを補間
siteB_data.addMoveAve()#移動平均値をDFへ追加
siteB_data.setMicroClass()#プランクトンの細胞データをクラスのラベルに変換

#%%
#クラスの環境データとプランクトンデータを結合し、分析に用いるデータフレームを作成
original_df = pd.concat([siteB_data.df_pla,siteB_data.df_env],axis=1,join='inner')
original_df

original_df = original_df["2001":"2017"]
original_df = original_df[original_df.index.month.isin([6,7,8,9,10])]
original_df

#%%
#ラベルと特徴量データに分ける。
df_label, df_features = sepLabelData(original_df)

#%%
# モデルを作成(RandomForest)学習データとテストデータを分ける
df_train, df_test, label_train, label_test = train_test_split(df_features, df_label, test_size=0.25, random_state = 777)
#学習
clf = RandomForestClassifier(n_estimators=150, random_state = 777) #(決定木の数,ランダム固定(任意の数))
clf.fit(df_train, label_train) #fit()メソッドでモデルに学習(学習データ)
print("予測の精度")
print(clf.score(df_test, label_test))

#%%
#重要度のデータフレーム
#特徴量データの列名と、重要度の値を使ってデータフレームを作成
#降順で並び替え・インデックスでスライス・特徴量取り出し。
#★注意★：グラフ描写の都合で、データフレーム内は昇順になっている。
df_importance = pd.DataFrame({'features':df_train.columns,'importance':clf.feature_importances_}).sort_values(by='importance',ascending=True)

#%%
df_importance.to_excel(r'C:\Users\●●●●●●●●●●●●●\imp.xlsx')

#%%
#重要度のデータフレーム
#特徴量データの列名と、重要度の値を使ってデータフレームを作成
#降順で並び替え・インデックスでスライス・特徴量取り出し。
top8_list = list(df_importance.sort_values(by='importance',ascending=False)[0:8]["features"])
top8_list.append("Plankton sp.")#重要度のリストからはプランクトンのデータが抜けているので忘れずに追加。
top8_df = original_df[top8_list]

#%%
#ラベルと特徴量データに分ける。
dftop8_label, dftop8_features = sepLabelData(top8_df)

# モデルを作成(RandomForest)学習データとテストデータを分ける
df_train, df_test, label_train, label_test = train_test_split(dftop8_features, dftop8_label, test_size=0.25, random_state = 888)

#学習
clf = RandomForestClassifier(n_estimators=150, random_state = 888) #(決定木の数,ランダム固定(任意の数))
clf.fit(df_train, label_train) #fit()メソッドでモデルに学習(学習データ)

print("予測の精度")
print(clf.score(df_test, label_test))