
# coding: utf-8

# Import Dependencies

# In[1]:


from __future__ import division
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


# Import Data

# In[2]:


ATL = pd.read_csv('../ATL.txt')
BKN = pd.read_csv('../BKN.txt')
BOS = pd.read_csv('../BOS.txt')
CHI = pd.read_csv('../CHI.txt')
CHO = pd.read_csv('../CHO.txt')
CLE = pd.read_csv('../CLE.txt')
DAL = pd.read_csv('../DAL.txt')
DEN = pd.read_csv('../DEN.txt')
DET = pd.read_csv('../DET.txt')
GSW = pd.read_csv('../GSW.txt')
HOU = pd.read_csv('../HOU.txt')
IND = pd.read_csv('../IND.txt')
LAC = pd.read_csv('../LAC.txt')
LAL = pd.read_csv('../LAL.txt')
MEM = pd.read_csv('../MEM.txt')
MIA = pd.read_csv('../MIA.txt')
MIL = pd.read_csv('../MIL.txt')
MIN = pd.read_csv('../MIN.txt')
NOP = pd.read_csv('../NOP.txt')
NYK = pd.read_csv('../NYK.txt')
OKC = pd.read_csv('../OKC.txt')
ORL = pd.read_csv('../ORL.txt')
PHI = pd.read_csv('../PHI.txt')
PHO = pd.read_csv('../PHO.txt')
POR = pd.read_csv('../POR.txt')
SAC = pd.read_csv('../SAC.txt')
SAS = pd.read_csv('../SAS.txt')
TOR = pd.read_csv('../TOR.txt')
UTA = pd.read_csv('../UTA.txt')
WAS = pd.read_csv('../WAS.txt')


# sample input

# In[3]:


OKC.head(3)


# Crate a dictionary out of data

# In[4]:


t_dict = {'Atlanta Hawks' : ATL,'Brooklyn Nets' : BKN,'Boston Celtics' : BOS,'Chicago Bulls' : CHI, 'Charlotte Hornets' : CHO,
        'Cleveland Cavaliers' : CLE,'Dallas Mavericks' : DAL, 'Denver Nuggets' : DEN,'Detroit Pistons' : DET,'Golden State Warriors' : GSW, 
        'Houston Rockets' : HOU,'Indiana Pacers' : IND,'Los Angeles Clippers' : LAC, 'Los Angeles Lakers' : LAL,'Memphis Grizzlies' : MEM,
        'Miami Heat' : MIA, 'Milwaukee Bucks' : MIL,'Minnesota Timberwolves' : MIN,'New Orleans Pelicans' : NOP, 'New York Knicks' : NYK,
        'Oklahoma City Thunder' : OKC,'Orlando Magic' : ORL, 'Philadelphia 76ers' : PHI,'Phoenix Suns' : PHO,'Portland Trail Blazers' : POR, 
        'Sacramento Kings' : SAC,'San Antonio Spurs' : SAS,'Toronto Raptors' : TOR, 'Utah Jazz' : UTA,'Washington Wizards' : WAS}


# Adjustments and Feature Engineering

# In[6]:


for team_name, team in t_dict.iteritems():
        team['Away'] = np.where(team['Unnamed: 5']=='@',1,0)
        team['Roadtrip'] = team['Away'] * (team['Away'].groupby((team['Away'] != team['Away'].shift()).cumsum()).cumcount() + 1)
        team.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4','Unnamed: 5','Unnamed: 8','Notes'], axis=1, inplace = True)
        team['diff'] = (team['Tm']-team['Opp'])
        team['+/-'] = team['diff'].cumsum()
        team['pct'] = np.round(team['W']/(team['W']+team['L']),3)
        team['win'] = np.where(team['Unnamed: 7']=='W', 1, 0)
        team['Res'] = team['Streak'].str[:1]
        team['WS'] = np.where(team['Res']=='W', 1, 0)*team['Streak'].str[2:]
        team['LS'] = np.where(team['Res']=='L', 1, 0)*team['Streak'].str[2:]
        team['WStreak'] = np.where(team['WS']=='', 0,team['WS'])
        team['LStreak'] = np.where(team['LS']=='', 0,team['LS'])
        team['WStreak'] = pd.to_numeric(team['WStreak'])
        team['LStreak'] = pd.to_numeric(team['LStreak'])
        team.drop(['Res', 'WS', 'LS','diff', 'Unnamed: 7','Streak','Tm','Opp','W','L'], axis=1, inplace = True)
        team['Prevdate'] = (pd.to_datetime(team['Date'])-pd.to_datetime(team['Date']).shift(+1)) #prev game in int
        team.drop(team.index[:1], inplace=True)
        team['Rest'] = (team['Prevdate']/ np.timedelta64(1, 'D')).astype(int)
        team.drop(['Prevdate'], axis=1, inplace = True)


# In[7]:


OKC.head(3)


# Shift some features to next game as the model can only have info that existed before the game

# In[8]:


for team_name, team in t_dict.iteritems():
    team[['Roadtrip','+/-','pct','win','WStreak','LStreak']] =     team[['Roadtrip','+/-','pct','win','WStreak','LStreak']].shift(+1)
    team.drop(team.index[team['Date'].str.contains('Oct')], inplace=True)
    team.drop(team.index[team['Date'].str.contains('Feb')], inplace=True)
    team.reset_index(drop=True, inplace=True)


# Match every row with opponent, obtain data for opponent and add to same row (make sure it is only info up to the date)

# In[9]:


for team_name, team in t_dict.iteritems():
    team['OppRoadtrip'] = np.nan
    team['Opp+/-'] = np.nan
    team['Opppct'] = np.nan
    team['OppPrevRes'] = np.nan
    team['OppWStreak'] = np.nan
    team['OppLStreak'] = np.nan
    team['OppRest'] = np.nan
    for index, row in team.iterrows():
        opp = row['Opponent']
        opp_team = t_dict.get(opp)
        date_comp = team.loc[index,'Date']
        opp_index_list = opp_team.index[opp_team['Date'] == date_comp].tolist()
        opp_index = 0
        opp_index = opp_index_list.pop()
        team.loc[index,'OppRoadtrip'] = opp_team.loc[opp_index,'Roadtrip']
        team.loc[index,'Opp+/-'] = opp_team.loc[opp_index,'+/-']
        team.loc[index,'Opppct'] = opp_team.loc[opp_index,'pct']
        team.loc[index,'OppPrevRes'] = opp_team.loc[opp_index,'win']
        team.loc[index,'OppWStreak'] = opp_team.loc[opp_index,'WStreak']
        team.loc[index,'OppLStreak'] = opp_team.loc[opp_index,'LStreak']
        team.loc[index,'OppRest'] = opp_team.loc[opp_index,'Rest']


# In[10]:


OKC.head(3)


# Remove away games and concatenate all teams together

# In[11]:


data = pd.DataFrame(columns = ['Roadtrip','+/-','pct','win','WStreak','LStreak','Rest','OppRoadtrip', 
                               'Opp+/-','Opppct','OppPrevRes','OppWStreak','OppLStreak','OppRest'])
for team_name, team in t_dict.iteritems():
    team.drop(team.index[team['Away']==1], inplace=True)
    team.drop(['G','Date','Away','Opponent'], axis=1, inplace = True)
    data = pd.concat([data, team], ignore_index=True)
    
data.rename(columns={'win':'PrevRes'}, inplace=True)


# In[12]:


data.head(3)


# REPEAT SIMILAR PROCESS IN ORDER TO OBTAIN LABELS(COULD NOT DO WITH EXISTING DATA BECAUSE OF THE SHIFT)

# In[13]:


ATL = pd.read_csv('../ATL.txt')
BKN = pd.read_csv('../BKN.txt')
BOS = pd.read_csv('../BOS.txt')
CHI = pd.read_csv('../CHI.txt')
CHO = pd.read_csv('../CHO.txt')
CLE = pd.read_csv('../CLE.txt')
DAL = pd.read_csv('../DAL.txt')
DEN = pd.read_csv('../DEN.txt')
DET = pd.read_csv('../DET.txt')
GSW = pd.read_csv('../GSW.txt')
HOU = pd.read_csv('../HOU.txt')
IND = pd.read_csv('../IND.txt')
LAC = pd.read_csv('../LAC.txt')
LAL = pd.read_csv('../LAL.txt')
MEM = pd.read_csv('../MEM.txt')
MIA = pd.read_csv('../MIA.txt')
MIL = pd.read_csv('../MIL.txt')
MIN = pd.read_csv('../MIN.txt')
NOP = pd.read_csv('../NOP.txt')
NYK = pd.read_csv('../NYK.txt')
OKC = pd.read_csv('../OKC.txt')
ORL = pd.read_csv('../ORL.txt')
PHI = pd.read_csv('../PHI.txt')
PHO = pd.read_csv('../PHO.txt')
POR = pd.read_csv('../POR.txt')
SAC = pd.read_csv('../SAC.txt')
SAS = pd.read_csv('../SAS.txt')
TOR = pd.read_csv('../TOR.txt')
UTA = pd.read_csv('../UTA.txt')
WAS = pd.read_csv('../WAS.txt')


# In[14]:


label_dict = {'Atlanta Hawks' : ATL,'Brooklyn Nets' : BKN,'Boston Celtics' : BOS,'Chicago Bulls' : CHI, 'Charlotte Hornets' : CHO,
        'Cleveland Cavaliers' : CLE,'Dallas Mavericks' : DAL, 'Denver Nuggets' : DEN,'Detroit Pistons' : DET,'Golden State Warriors' : GSW, 
        'Houston Rockets' : HOU,'Indiana Pacers' : IND,'Los Angeles Clippers' : LAC, 'Los Angeles Lakers' : LAL,'Memphis Grizzlies' : MEM,
        'Miami Heat' : MIA, 'Milwaukee Bucks' : MIL,'Minnesota Timberwolves' : MIN,'New Orleans Pelicans' : NOP, 'New York Knicks' : NYK,
        'Oklahoma City Thunder' : OKC,'Orlando Magic' : ORL, 'Philadelphia 76ers' : PHI,'Phoenix Suns' : PHO,'Portland Trail Blazers' : POR, 
        'Sacramento Kings' : SAC,'San Antonio Spurs' : SAS,'Toronto Raptors' : TOR, 'Utah Jazz' : UTA,'Washington Wizards' : WAS}


# In[15]:


for team_name, team in label_dict.iteritems():
        team['Away'] = np.where(team['Unnamed: 5']=='@',1,0)
        team['Roadtrip'] = team['Away'] * (team['Away'].groupby((team['Away'] != team['Away'].shift()).cumsum()).cumcount() + 1)
        team.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4','Unnamed: 5','Unnamed: 8','Notes'], axis=1, inplace = True)
        team['diff'] = (team['Tm']-team['Opp'])
        team['+/-'] = team['diff'].cumsum()
        team['pct'] = np.round(team['W']/(team['W']+team['L']),3)
        team['win'] = np.where(team['Unnamed: 7']=='W', 1, 0)
        team['Res'] = team['Streak'].str[:1]
        team['WS'] = np.where(team['Res']=='W', 1, 0)*team['Streak'].str[2:]
        team['LS'] = np.where(team['Res']=='L', 1, 0)*team['Streak'].str[2:]
        team['WStreak'] = np.where(team['WS']=='', 0,team['WS'])
        team['LStreak'] = np.where(team['LS']=='', 0,team['LS'])
        team['WStreak'] = pd.to_numeric(team['WStreak'])
        team['LStreak'] = pd.to_numeric(team['LStreak'])
        team.drop(['Res', 'WS', 'LS','diff', 'Unnamed: 7','Streak','Tm','Opp','W','L'], axis=1, inplace = True)
        team['Prevdate'] = (pd.to_datetime(team['Date'])-pd.to_datetime(team['Date']).shift(+1)) #prev game in int
        team.drop(team.index[:1], inplace=True)
        team['Rest'] = (team['Prevdate']/ np.timedelta64(1, 'D')).astype(int)
        team.drop(['Prevdate'], axis=1, inplace = True)


# In[16]:


for team_name, team in label_dict.iteritems():
    team.drop(team.index[team['Date'].str.contains('Oct')], inplace=True)
    team.drop(team.index[team['Date'].str.contains('Feb')], inplace=True)
    team.drop(team.index[team['Away']==1], inplace=True)
    team.reset_index(drop=True, inplace=True)
    team.drop(['G','Date', 'Opponent', 'Roadtrip','+/-', 'pct','WStreak','LStreak','Rest'], axis=1, inplace = True)
    team.reset_index(drop=True, inplace=True)


# In[17]:


labels = pd.DataFrame(columns = ['win'])
for team_name, team in label_dict.iteritems():
    team.drop(team.index[team['Away']==1], inplace=True)
    team.drop(['Away'], axis=1, inplace = True)
    labels = pd.concat([labels, team], ignore_index=True)


# In[18]:


labels.head(3)


# WE NOW HAVE THE TRAINING DATA AND THE TARGET SO WE CAN TRAIN THE MODEL

# In[19]:


X = data.values
y = labels.values
y=y.astype('int')
X_scaled = preprocessing.scale(X)


# In[20]:


training_size = 0.8
training_index = int(round(len(X)*0.8))
train_X = np.asarray(X_scaled[:training_index],dtype=np.float64)
test_X = np.asarray(X_scaled[training_index:],dtype=np.float64)
train_Y = np.asarray(y[:training_index],dtype=np.float64)
test_Y = np.asarray(y[training_index:],dtype=np.float64)


# Data is now split 80%/20% and scaled with a mean of 0 and standard deviation of 1

# I ran 100 models and had each model vote if the home team will win or not.
# The end result is equal to the probability of the home team winning.

# First 25 models are Neural Network(NN) with identity as activation function and stochastic gradient descent as the method
# Regularization parameter is quite large, and learning rate relatively low. Restriction to number of iterations at 100

# In[21]:


big = np.zeros(len(test_Y))
for i in range(25):
    nn = MLPClassifier(activation='identity',alpha=0.001, #identity activation function, large reg(gradient on full set)
                        batch_size=X.shape[0], #small dataset, optimising on full
                        hidden_layer_sizes=(30,50,30), #heuristically chosen
                        learning_rate_init=0.001, max_iter=100, #small learning rate and many iterations(small dataset)
                        solver='sgd') #stochastic grdient descent chosen(simplest method - no need for complexity)
    nn.fit(train_X,train_Y)
    pred = nn.predict(test_X)
    big += pred


# Second 25 models are also NN with very similar features but the Restriction to number of iterations is at 10000

# In[22]:


for i in range(25):
    nn = MLPClassifier(activation='identity',alpha=0.001, #identity activation function, large reg(gradient on full set)
                        batch_size=X.shape[0], #small dataset, optimising on full
                        hidden_layer_sizes=(30,50,30), #heuristically chosen
                        learning_rate_init=0.001, max_iter=10000, #small learning rate and many iterations(small dataset)
                        solver='sgd') #stochastic grdient descent chosen(simplest method - no need for complexity)
    nn.fit(train_X,train_Y)
    pred = nn.predict(test_X)
    big += pred


# Third 25 models are also NN but significantly different. Rectified Linear Unit is used and Adam as a solver

# In[23]:


for i in range(25):
    nn = MLPClassifier(activation='relu',alpha=0.001, #relu
                        batch_size=X.shape[0], #small dataset, optimising on full
                        hidden_layer_sizes=(30,50,30), #heuristically chosen
                        learning_rate_init=0.001, max_iter=100, #small learning rate and many iterations(small dataset)
                        solver='adam',
                        random_state=i) #stochastic grdient descent chosen(simplest method - no need for complexity)
    nn.fit(train_X,train_Y)
    pred = nn.predict(test_X)
    big += pred


# Last 25 models are Random Forest that doesn't allow the trees to grow deep in order to prevent overfitting

# In[24]:


for i in range(25):
    rf = RandomForestClassifier(max_depth=5, random_state=i) #don't allow trees to go deep, prevent overfitting
    rf.fit(train_X,train_Y)
    pred = rf.predict(test_X)
    big += pred


# In[25]:


def scale(X,new_min,new_max):
    old_max = max(X)
    old_min = min(X)
    factor = ((new_max-new_min)/(old_max-old_min))
    X_new = np.zeros(len(X))
    for i in range(len(X)):
        X_new[i] = int(factor*(X[i]-old_max)+new_max)
    return X_new

#Avoid probabilities of 100% and 0%
Prediction_vote = scale(big,10,90)


# Model has now calculated probabilites, need to map this back to the games (what follows is similar to previous work)

# In[26]:


ATL = pd.read_csv('../ATL.txt')
BKN = pd.read_csv('../BKN.txt')
BOS = pd.read_csv('../BOS.txt')
CHI = pd.read_csv('../CHI.txt')
CHO = pd.read_csv('../CHO.txt')
CLE = pd.read_csv('../CLE.txt')
DAL = pd.read_csv('../DAL.txt')
DEN = pd.read_csv('../DEN.txt')
DET = pd.read_csv('../DET.txt')
GSW = pd.read_csv('../GSW.txt')
HOU = pd.read_csv('../HOU.txt')
IND = pd.read_csv('../IND.txt')
LAC = pd.read_csv('../LAC.txt')
LAL = pd.read_csv('../LAL.txt')
MEM = pd.read_csv('../MEM.txt')
MIA = pd.read_csv('../MIA.txt')
MIL = pd.read_csv('../MIL.txt')
MIN = pd.read_csv('../MIN.txt')
NOP = pd.read_csv('../NOP.txt')
NYK = pd.read_csv('../NYK.txt')
OKC = pd.read_csv('../OKC.txt')
ORL = pd.read_csv('../ORL.txt')
PHI = pd.read_csv('../PHI.txt')
PHO = pd.read_csv('../PHO.txt')
POR = pd.read_csv('../POR.txt')
SAC = pd.read_csv('../SAC.txt')
SAS = pd.read_csv('../SAS.txt')
TOR = pd.read_csv('../TOR.txt')
UTA = pd.read_csv('../UTA.txt')
WAS = pd.read_csv('../WAS.txt')


# In[27]:


t_dict = {'Atlanta Hawks' : ATL,'Brooklyn Nets' : BKN,'Boston Celtics' : BOS,'Chicago Bulls' : CHI, 'Charlotte Hornets' : CHO,
        'Cleveland Cavaliers' : CLE,'Dallas Mavericks' : DAL, 'Denver Nuggets' : DEN,'Detroit Pistons' : DET,'Golden State Warriors' : GSW, 
        'Houston Rockets' : HOU,'Indiana Pacers' : IND,'Los Angeles Clippers' : LAC, 'Los Angeles Lakers' : LAL,'Memphis Grizzlies' : MEM,
        'Miami Heat' : MIA, 'Milwaukee Bucks' : MIL,'Minnesota Timberwolves' : MIN,'New Orleans Pelicans' : NOP, 'New York Knicks' : NYK,
        'Oklahoma City Thunder' : OKC,'Orlando Magic' : ORL, 'Philadelphia 76ers' : PHI,'Phoenix Suns' : PHO,'Portland Trail Blazers' : POR, 
        'Sacramento Kings' : SAC,'San Antonio Spurs' : SAS,'Toronto Raptors' : TOR, 'Utah Jazz' : UTA,'Washington Wizards' : WAS}


# In[28]:


for team_name, team in t_dict.iteritems():
        team['Team'] = team_name
        team['Away'] = np.where(team['Unnamed: 5']=='@',1,0)
        team['Roadtrip'] = team['Away'] * (team['Away'].groupby((team['Away'] != team['Away'].shift()).cumsum()).cumcount() + 1)
        team.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4','Unnamed: 5','Unnamed: 8','Notes'], axis=1, inplace = True)
        team['diff'] = (team['Tm']-team['Opp'])
        team['+/-'] = team['diff'].cumsum()
        team['pct'] = np.round(team['W']/(team['W']+team['L']),3)
        team['win'] = np.where(team['Unnamed: 7']=='W', 1, 0)
        team['Res'] = team['Streak'].str[:1]
        team['WS'] = np.where(team['Res']=='W', 1, 0)*team['Streak'].str[2:]
        team['LS'] = np.where(team['Res']=='L', 1, 0)*team['Streak'].str[2:]
        team['WStreak'] = np.where(team['WS']=='', 0,team['WS'])
        team['LStreak'] = np.where(team['LS']=='', 0,team['LS'])
        team['WStreak'] = pd.to_numeric(team['WStreak'])
        team['LStreak'] = pd.to_numeric(team['LStreak'])
        team.drop(['Res', 'WS', 'LS','diff', 'Unnamed: 7','Streak','Tm','Opp','W','L'], axis=1, inplace = True)
        team['Prevdate'] = (pd.to_datetime(team['Date'])-pd.to_datetime(team['Date']).shift(+1)) #prev game in int
        team.drop(team.index[:1], inplace=True)
        team['Rest'] = (team['Prevdate']/ np.timedelta64(1, 'D')).astype(int)
        team.drop(['Prevdate'], axis=1, inplace = True)


# In[29]:


for team_name, team in t_dict.iteritems():
    team[['Roadtrip','+/-','pct','win','WStreak','LStreak']] =     team[['Roadtrip','+/-','pct','win','WStreak','LStreak']].shift(+1)
    team.drop(team.index[team['Date'].str.contains('Oct')], inplace=True)
    team.drop(team.index[team['Date'].str.contains('Feb')], inplace=True)
    team.reset_index(drop=True, inplace=True)


# In[30]:


for team_name, team in t_dict.iteritems():
    team['OppRoadtrip'] = np.nan
    team['Opp+/-'] = np.nan
    team['Opppct'] = np.nan
    team['OppPrevRes'] = np.nan
    team['OppWStreak'] = np.nan
    team['OppLStreak'] = np.nan
    team['OppRest'] = np.nan
    for index, row in team.iterrows():
        opp = row['Opponent']
        opp_team = t_dict.get(opp)
        date_comp = team.loc[index,'Date']
        opp_index_list = opp_team.index[opp_team['Date'] == date_comp].tolist()
        opp_index = 0
        opp_index = opp_index_list.pop()
        team.loc[index,'OppRoadtrip'] = opp_team.loc[opp_index,'Roadtrip']
        team.loc[index,'Opp+/-'] = opp_team.loc[opp_index,'+/-']
        team.loc[index,'Opppct'] = opp_team.loc[opp_index,'pct']
        team.loc[index,'OppPrevRes'] = opp_team.loc[opp_index,'win']
        team.loc[index,'OppWStreak'] = opp_team.loc[opp_index,'WStreak']
        team.loc[index,'OppLStreak'] = opp_team.loc[opp_index,'LStreak']
        team.loc[index,'OppRest'] = opp_team.loc[opp_index,'Rest']


# In[31]:


games = pd.DataFrame(columns = ['Date','Team','Opponent','pct','Opppct','Rest','OppRest'])
for team_name, team in t_dict.iteritems():
    team.drop(team.index[team['Away']==1], inplace=True)
    team.drop(['Away','G','+/-','win','WStreak','LStreak','Roadtrip','OppRoadtrip',               
	           'Opp+/-','OppPrevRes','OppWStreak','OppLStreak'], axis=1, inplace = True)
    games = pd.concat([games, team], ignore_index=True)


# In[32]:


training_size = 0.8
training_index = int(round(len(games)*0.8))
train_games = np.asarray(games[:training_index])
test_games = np.asarray(games[training_index:])


# Attach probabilities to known games and publish result

# In[33]:


final = pd.DataFrame(test_games,Prediction_vote)
final.reset_index(drop=False, inplace=True)
final.columns = ['Home Win%', 'Date','OppRest','Away Team','Opppct','Rest','Home Team','pct']
cols = ['Date','Home Team','Away Team','Home Win%','pct','Rest','Opppct','OppRest']
final = final[cols]
final['deltapct_adj'] = final['pct']*final['Rest']-final['Opppct']*final['OppRest']
final['actual result'] = test_Y
indices = [6,30,58,75,95]
publish = final.loc[indices]
publish.drop(['OppRest','Rest','pct','Opppct','deltapct_adj'], axis=1, inplace = True)


# In[35]:


publish


# Interesting Visualisation

# In[36]:


plt.figure(figsize=(10,10))
plt.scatter(final['Home Win%'],final['deltapct_adj'])
plt.plot([10, 80], [-1.2, 1.2], '--')
plt.xlim(5,85)
plt.xlabel('Home Win Probability%')
plt.ylim(-1.5,1.5)
plt.ylabel('Win percentage timed by rest days')
plt.title('Adjusted Win Percentage VS Home Win Probability')
plt.show()

