from statistics import mean

import pandas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import stats
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, RepeatedKFold, cross_validate
from sklearn import metrics
import numpy as np
from numpy import arange, std, absolute
from hana_ml.algorithms.pal.preprocessing import variance_test
from matplotlib import pyplot
'''import os
os.environ['CUDA_VISIBLE_DEVICES']="-1"
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense'''
class Model:

    running = False

    def clean_pollbase(self):
        new_df = pd.read_csv("datasets/opinionpolls.csv") # Read excel file

        # Noticed that all unnamed columns did not contain necessary data, t.f. drop
        new_df = new_df.loc[:, ~new_df.columns.str.contains('^Unnamed')] #Delete unnamed columns

        new_df = new_df[new_df['Con'].notna()] #drop polls that do not hold voting intention for conservatives
        new_df = new_df[new_df['Published'].notna()] #drop polls that do not have a published date
        new_df = new_df[new_df['Month'].notna()] #drop polls that do not have the month published

        new_df.Month = new_df['Month'].str.strip() #delete whitespace in strings in month column

        new_df.iloc[0, 0] = "2020"

        date_df = new_df.iloc[:, 0:4] #Used for formatting published date
        date_df['Year'] = pd.to_numeric(date_df['Year'], errors='coerce')
        date_df['Year'] = date_df['Year'].fillna(0)
        date_df['Year'] = date_df['Year'].astype(int)

        days = []
        month = 0
        months = []
        date = ''
        dates = []

        for index, row in date_df.iterrows():
            if '-' in row['Published']:
                days.append(row['Published'][0:1])
                months.append(row['Published'][3:6])
            else:
                days.append(row['Published'])
                months.append(row['Month'])

        date_df.drop(['Month'], axis=1)
        date_df.drop(['Published'], axis=1)

        date_df['Month'] = months

        date_df['Published'] = days

        monthDict = {'Jan':'01', 'Feb':'02', 'Mar':'03', 'Apr':'04', 'May':'05', 'Jun':'06', 'Jul':'07', 'Aug':'08',
                     'Sep':'09', 'Oct':'10','Nov':'11', 'Dec':'12'}

        date_df['Month'] = date_df['Month'].astype("string")
        date_df['Month'].str.strip()
        date_df['Month'] = date_df['Month'].map(monthDict)

        date_df['Month'] = date_df['Month'].fillna(0)

        for index, row in date_df.iterrows():
            if '-' in row['Published']:
                day = row['Published'][0:1]
                month = row['Published'][3:6]

                date = str(row['Year']) + month + day
            elif '/' in row['Published']:
                day = row['Published'][0:1]
                month = "0" + row['Published'][2:3]
                date = str(row['Year']) + month + day
            else:
                date = str(row['Year']) + str(row['Month']) + str(row['Published'])

            if len(date) < 8:
                date = date[0:6] + "0" + date[6]

            dates.append(date)

        date_df['date'] = dates

        new_df.drop(['Year', 'Month', 'Fieldwork', 'Published', 'Ukip'], axis=1, inplace=True)
        new_df['date_published'] = dates

        new_df['date_published'] = pd.to_datetime(new_df['date_published'], format='%Y%m%d')

        ordinals = []
        for i in new_df['date_published']:
            ordinals.append(i.toordinal())

        new_df['date_ordinals'] = ordinals

        new_df.to_csv("datasets/CleanOpinionPolls.csv", index=False)


    def pre_process_pollbase(self):
        clean_df = pd.read_csv('datasets/CleanOpinionPolls(new).csv', index_col=0)
        clean_df = clean_df.iloc[:, 1:7]
        print(clean_df)
        clean_df.dropna(inplace=True)
        clean_df['date_published'] = pd.to_datetime(clean_df['date_published'], errors='coerce')


        clean_df['date_published'] = pd.to_datetime(clean_df['date_published'], format='%Y%m%d')

        ordinals = []
        for i in clean_df['date_published']:
            ordinals.append(i.toordinal())

        clean_df['date_ordinals'] = ordinals
        clean_df.sort_values(by="date_ordinals", inplace=True)


        # Deleting outliers

        opinion_polls = clean_df
        index = -1
        for j in opinion_polls.columns:
            if not "date" in j:
                index = index + 1
                s = opinion_polls.sort_values(by=j)
                party_votes = s[j]
                q1, q3 = np.percentile(party_votes, [25, 75])
                iqr = q3 - q1
                lower = q1 - (0.25 * iqr)
                upper = q3 + (0.25 * iqr)

                print(j, "Lower:", lower, "Higher:", upper)

                party_votes = 0
                i = -1
                print("index", index)
                while party_votes < lower:
                    i = i + 1
                    party_votes = s.iloc[i, index]

                outliers_lower = s.head(i + 1)

                non_outliers = s.iloc[i + 1:, :]

                party_votes = 0
                i = -1
                while party_votes < upper:
                    i = i + 1
                    party_votes = s.iloc[i, index]

                outliers_higher = s.tail(len(s) - i + 1)
                non_outliers = non_outliers.iloc[:i + 1, :]

                outliers = pd.concat([outliers_lower, outliers_higher])

                plt.clf()
                sns.scatterplot(data=non_outliers, x="date_ordinals", y=j, color='blue')
                sns.scatterplot(data=outliers, x="date_ordinals", y=j, color='red')
                if j == "BXP/Reform":
                    j = "Reform"
                string = 'static/media/outliers/' + j + '_outliers.png'
                plt.savefig(string)

                s = non_outliers

        '''for i in clean_df.columns:
            if not "date" in i:
                upper = clean_df[i].mean() + clean_df[i].std()
                lower = clean_df[i].mean() - clean_df[i].std()
                clean_df = clean_df.loc[(clean_df[i] < upper) & (clean_df[i] > lower)]'''

        s.to_csv('datasets/PreprocessedOpinionPolls.csv', index=False)

    def forecast_popular_vote(self):
        model = HuberRegressor()
        pp_df = pd.read_csv("datasets/PreprocessedOpinionPolls.csv")

        model_results = {
            "Polling": ["model"],
            "Con":[],
            "Lab":[],
            "LD":[],
            "Green":[],
            "BXP/Reform":[],
            "date_published":["2024-12-08"],
            "date_ordinals":[pd.to_datetime("2024-12-08").toordinal()]
        }

        for i in pp_df.columns:
            if not "date" in i:
                x = pp_df['date_ordinals'].tail(300).values.reshape(-1, 1)
                y = pp_df[i].tail(300).values.reshape(-1, 1)
                pred_date = "2024-12-08"
                pred_date = pd.to_datetime(pred_date)
                pred_date_ords = pred_date.toordinal()

                model.fit(x, y)
                pred = model.predict([[pred_date_ords]])
                model_results[i].append(float(pred))

        print(model_results)
        model_results = pd.DataFrame(model_results)
        clean_df = pd.read_csv("datasets/CleanOpinionPolls.csv")
        pp_df = pd.concat([pp_df, model_results.iloc[:, 1:]], ignore_index=True)
        clean_df = pd.concat([clean_df, model_results], ignore_index=True)
        pp_df.to_csv("datasets/PreprocessedOpinionPolls.csv", index=False)
        clean_df.to_csv("datasets/CleanOpinionPolls.csv", index=False)
        return model

    #TODO
    def gen_popvote_graph(self, model):
        clean_df = pd.read_csv("datasets/CleanOpinionPolls.csv", index_col=False)

        pp_df = pd.read_csv("datasets/PreprocessedOpinionPolls.csv", index_col=False)

        clean_df = clean_df.iloc[:, 1:]
        colours = ["blue", "red", "orange", "green", "cyan"]

        plt.clf()
        for i, c in zip(clean_df.columns, colours):
            if not "date" in i:
                x = pp_df["date_ordinals"]
                y = pp_df[i]

                xaxis = [x[len(x)-2], x[len(x)-1]]
                yaxis = [y[len(y) - 2], y[len(y) - 1]]

                plt.plot(xaxis, yaxis, color=c)
                plt.title("Popular Vote Forecast")

                ax = sns.scatterplot(data=clean_df, y=i, s=5, x="date_ordinals", color=c, label = i)

                plt.annotate(str(pp_df[i][len(pp_df) - 1]),
                             xy=(pp_df["date_ordinals"][len(pp_df) - 1] + 90, pp_df[i][len(pp_df) - 1]))

        plt.legend(fontsize="7")
        plt.xticks()
        ax.set(ylabel='% Vote')

        plt.savefig("static/media/forecasts/opinionpollgraph.png")
        plt.clf()



    def test_pop_vote(self):
        linear = LinearRegression()
        huber = HuberRegressor()
        ransac = RANSACRegressor(random_state=64)

        #Test each model based on Con Vote
        clean_df = pd.read_csv("datasets/CleanOpinionPolls.csv")
        pp_df = m.identify_outliers(pd.read_csv("datasets/PreprocessedOpinionPolls.csv"))
        print(len(pp_df))

        x = pp_df['date_ordinals'].values.reshape(-1, 1)
        y = pp_df['Lab'].values.reshape(-1, 1)

        #Linear Regression

        linear.fit(x, y)
        pred = linear.predict(x)
        scores = r2_score(y, pred)
        print(linear, scores.mean())

        plt.clf()
        plt.scatter(x, y, label="Labour voting Intention", s=5)
        plt.plot(x, pred, color='black', label="Linear")
        plt.annotate(text = str(np.round(pred[len(pred)-1], 2)), xy = (x[len(x)-1], pred[len(pred)-1]))

        results = self.evaluate_model(x, y, linear)

        evals =[]
        evals.append('Linear Regression Mean MAE: %.3f (%.3f)' % (mean(results), std(results)))

        # Huber Regression

        huber.fit(x, y)
        pred = huber.predict(x)
        scores = r2_score(y, pred)
        print('huber', scores.mean())

        plt.plot(x, pred, color='red', label="Huber")
        plt.annotate(text = str(np.round(pred[len(pred)-1], 2)), xy = (x[len(x)-1], pred[len(pred)-1]))

        results = self.evaluate_model(x, y, huber)
        evals.append('Huber Regression Mean MAE: %.3f (%.3f)' % (mean(results), std(results)))

        # RANSAC regression

        ransac.fit(x, y)
        pred = ransac.predict(x)

        scores = r2_score(y, pred)
        print("RANSAC", scores.mean())

        plt.plot(x, pred, color="orange", label="RANSAC")

        plt.annotate(text = str(np.round(pred[len(pred)-1], 2)), xy = (x[len(x)-1], pred[len(pred)-1]))

        results = self.evaluate_model(x, y, ransac)
        evals.append(str('RANSAC Regression Mean MAE: %.3f (%.3f)' % (mean(results), std(results))))

        plt.legend()
        plt.title("Comparison of Regression models on Labour Voting Intention")
        plt.savefig('static/media/tests/linear.png')
        for i in evals:
            print(i)

    def run_pop_forecast(self):
        m = Model()
        m.clean_pollbase()
        m.pre_process_pollbase()
        model = m.forecast_popular_vote()
        m.gen_popvote_graph(model)

#Taken from https://machinelearningmastery.com/robust-regression-for-machine-learning-in-python/
    def evaluate_model(self, x, y, model):
        # define model evaluation method
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        # evaluate model
        scores = cross_val_score(model, x, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        # force scores to be positive
        return absolute(scores)

    def identify_outliers(self, opinion_polls):
        index = -1
        for j in opinion_polls.columns:
            if not "date" in j:
                index = index + 1
                s = opinion_polls.sort_values(by=j)
                party_votes = s[j]
                q1, q3 = np.percentile(party_votes, [25, 75])
                iqr = q3 - q1
                lower = q1-(0.25*iqr)
                upper = q3+(0.25*iqr)

                print(j ,"Lower:", lower, "Higher:", upper)

                party_votes = 0
                i = -1
                print("index", index)
                while party_votes < lower:
                    i = i+1
                    party_votes = s.iloc[i, index]

                outliers_lower = s.head(i+1)

                non_outliers = s.iloc[i+1:, :]

                party_votes = 0
                i = -1
                while party_votes < upper:
                    i = i + 1
                    party_votes = s.iloc[i, index]

                outliers_higher = s.tail(len(s) - i+1)
                non_outliers = non_outliers.iloc[:i+1, :]

                outliers = pd.concat([outliers_lower, outliers_higher])

                plt.clf()
                sns.scatterplot(data=non_outliers, x="date_ordinals", y=j, color='blue')
                sns.scatterplot(data=outliers, x="date_ordinals", y=j, color='red')
                if j == "BXP/Reform":
                    j = "Reform"
                string = 'static/media/outliers/'+ j + '_outliers.png'
                plt.savefig(string)

                s = non_outliers

        return s # dataset without outliers
    def clean_socioeconomic_data(self):
        sec_df = pd.read_csv("datasets/sec_csv.csv") # Turn sec.csv into a dataset
        new_df = pd.DataFrame()

        rankAv = 0
        count = 0
        totalrank = {'ConstituencyName': [], 'Rank': []}

        '''Iterate through all neighbourhoods in sec dataset, calculate average SEC rank for the constituency, add 
        totalrank dictionary'''
        for i in range(1, len(sec_df)):
            if sec_df['ConstituencyName'][i] == sec_df['ConstituencyName'][i-1]:
                rankAv = rankAv + sec_df['ranking_total'][i]
                count = count + 1
            elif (sec_df['ConstituencyName'][i] != sec_df['ConstituencyName'][i-1]) or sec_df['ConstituencyName'][i] == "End":
                string = str(sec_df['ConstituencyName'][i-1]) + ": " + str((rankAv / count))
                totalrank['ConstituencyName'].append(sec_df['ConstituencyName'][i-1])
                totalrank['Rank'].append(rankAv/count)
                rankAv = sec_df['rank'][i]
                count = 1

        #new_df holds data from totalrank
        new_df['ConstituencyName'] = totalrank['ConstituencyName']
        new_df['Rank'] = totalrank['Rank']

        new_df.sort_values(by='ConstituencyName', inplace=True)

        new_df.to_csv("datasets/clean_sec.csv")

    def calc_bias(self):
        #clean election summary
        summary = pd.read_csv('datasets/historical_csv/election_history_summary.csv')
        summary.drop([col for col in summary.columns if 'Unnamed' in col], axis=1,  inplace = True)
        summary.dropna(inplace=True)

        summary.to_csv('datasets/historical_csv/election_history_summary.csv')
        csvs = ['Alliance Vote', 'Brexit Vote', 'Con Vote', 'DUP Vote', 'Green Vote', 'Labour Vote', 'Liberal Vote',
                'Other Vote', 'PC Vote', 'SDLP Vote', 'Sinn Fein Vote', 'SNP Vote', 'UUP Vote']

        # example with conservatives


        test_df = pd.read_csv('datasets/historical_csv/Con Vote.csv')
        test_df = test_df.iloc[:, 4:]
        bias = []
        row_bias = []
        b = 0

        #TODO
        for i in range(0, len(test_df)): # for all rows in test_df
            for j in range(0, len(test_df.columns)-1): # for all columns in test_df
                count = 0
                if not np.isnan(float(summary.iloc[j,2].strip('%'))):
                    b = float(summary.iloc[j,2].strip('%')) - (test_df.iloc[i, j+1]*100)
                    row_bias.append(b)
                else:
                    count = count + 1
            bias.append(sum(row_bias)/(len(row_bias)-count))
        print(bias)
        #for i in range(0, len(test_df)): # For all constituencies in con_vote df


    def clean_historical(self):

        # Get excel file instance of the historical dataset
        file = pd.ExcelFile('datasets/election_history_cut_nospaces.xlsx')
        election_df_list = []

        #Iterate through all the sheets in the historical dataset and clean them using clean_history_sheet()
        for i in file.sheet_names:
            x = self.clean_historical_sheet(pd.read_excel('datasets/election_history_cut_nospaces.xlsx', i))
            election_df_list.append(x)
            string = "datasets/historical_csv/" + i + ".csv"
            x.to_csv(string)

        # Create a new CSV for each party, showing their electoral history in each constituency
        party_vote_columns = [col for col in election_df_list[len(election_df_list)-1] if 'Vote' in col]

        party_df_temp = pd.DataFrame({
            'id':[], 'Constituency':[], 'Region':[], '1964 Vote Share':[], '1966 Vote Share':[], '1970 Vote Share': [],
            '1974 (F) Vote Share': [], '1974 (O) Vote Share': [], '1979 Vote Share': [], '1983 Vote Share': [],
            '1987 Vote Share': [], '1992 Vote Share': [], '1997 Vote Share': [], '2001 Vote Share': [],
            '2005 Vote Share': [], '2010 Vote Share': [], '2015 Vote Share': [], '2017 Vote Share': [],
            '2019 Vote Share': [],
        })

        party_df_temp['id'] = election_df_list[len(election_df_list)-1]['id']
        party_df_temp['Constituency'] = election_df_list[len(election_df_list)-1]['Constituency']
        party_df_temp['Region'] = election_df_list[len(election_df_list)-1]['Country/Region']

        # Test for 1964 Election
        # Test for Conservatives

        # For all rows in 1964 Election
        '''for i in range(0, len(party_df)):
            # For all row in party df
            istrue = False
            for j in range(0, len(election_df)-1):
                if party_df['Constituency'].iloc[i] == election_df['Constituency'].iloc[j]:
                    result_list.append(election_df['Con Vote'].iloc[j])
                    j = len(party_df)
                    istrue = True
            if not istrue:
                result_list.append("")'''

        '''for n in range(0, len(election_df_list)):
            result_list = []
            for i in range(0, len(party_df)):
                # For all row in party df
                istrue = False
                for j in range(0, len(election_df_list[n]) - 1):
                    if party_df['Constituency'].iloc[i] == election_df_list[n]['Constituency'].iloc[j]:
                        result_list.append(election_df_list[n]['Con Vote'].iloc[j])
                        j = len(party_df)
                        istrue = True
                if not istrue:
                    result_list.append("")

            print((len(election_df_list)-n), "to go")
            party_df.iloc[:, n+2] = result_list'''

        summary = pd.read_csv("datasets/historical_csv/election_history_summary.csv", index_col=0)

        for m in range(0, len(party_vote_columns)): # for all parties
            party_df = party_df_temp
            for n in range(0, len(election_df_list)): # for all elections
                result_list = []
                if (party_vote_columns[m] in election_df_list[n].columns): # If party is running in election
                    for i in range(0, len(party_df)): # for all rows in party df
                        # For all row in party df
                        istrue = False
                        for j in range(0, len(election_df_list[n]) - 1): # For all rows in all election_dfs
                            # if constituency is in both party df and election df
                            if party_df['Constituency'].iloc[i] == election_df_list[n]['Constituency'].iloc[j]:
                                result_list.append(election_df_list[n][party_vote_columns[m]].iloc[j]) #append party result in constituency
                                j = len(party_df)
                                istrue = True
                        if not istrue:
                            result_list.append("0")
                else:
                    for i in range(0, 650):
                        result_list.append("0")
                election_df_list[n] = election_df_list[n].fillna(0)

                print((len(election_df_list) - n), "to go")
                party_df.iloc[:, n + 3] = result_list
            party_df.to_csv('datasets/historical_csv/'+party_vote_columns[m]+".csv", index=0)

        con_vote = pd.read_csv('datasets/historical_csv/Con Vote.csv')
        con_vote = con_vote.fillna(0)
        con_vote.to_csv('datasets/historical_csv/Con Vote.csv', index=0)

    def clean_historical_sheet(self, sheet):

        headers = sheet.columns
        newheaders = []

        for i in range(0, len(headers)): #remove whitespace from column headers
            newheaders.append(headers[i].strip())

        sheet.set_axis(newheaders, axis=1, inplace=True) #replace column names with non-whitespace newheaders list

        sheet.drop(['County', 'Electorate', 'Turnout'], axis=1, inplace=True) #Drop County column

        # drop columns with names that contain 'Votes
        votes = [col for col in sheet.columns if 'Votes' in col]
        for i in votes:
            try:
                sheet.drop(i, axis=1, inplace=True)
            except:
                print("hello")

        return sheet #return the cleaned sheet

    def pre_process_historical_data(self):
        df = pd.read_csv('datasets/historical_csv/Con Vote.csv')
        new_df_temp = pd.DataFrame()

        headers = df['Constituency']
        for i in headers:
            new_df_temp[i] = ''

        date_list = ['1964-10-15', '1966-03-31', '1970-06-18', '1974-02-28', '1974-10-10', '1979-05-03', '1983-06-09',
                     '1987-06-11', '1992-04-09', '1997-05-01', '2001-06-07', '2005-05-05', '2010-05-06', '2015-05-07',
                     '2017-06-08', '2019-12-12']

        new_df_temp['Election Date'] = date_list
        new_df_temp['Election Date'] = pd.to_datetime(new_df_temp['Election Date'], format='%Y-%m-%d')
        ordinals = []
        for i in new_df_temp['Election Date']:
            ordinals.append(i.toordinal())

        new_df_temp['Date Ordinals'] = ordinals

        csv_names = ['Alliance Vote', 'Brexit Vote', 'Con Vote', 'DUP Vote', 'Green Vote', 'Labour Vote', 'Liberal Vote',
                     'Other Vote', 'PC Vote', 'SDLP Vote', 'Sinn Fein Vote', 'SNP Vote', 'UUP Vote']

        for f in csv_names:
            string = "datasets/historical_csv/" + f + ".csv"
            df = pd.read_csv(string)
            new_df = new_df_temp
            for i in range(0, len(new_df.columns)):
                if not "Date" in new_df.columns[i]:
                    cons_list = df.iloc[i, 3:].values.tolist()


                    new_df[new_df.columns[i]] = cons_list
            print(len(cons_list))
            new_df.to_csv("datasets/historical_csv/pre_processed/" + f + ".csv", index = 0)

        cons_df_temp = pd.DataFrame({'Alliance Vote':[], 'Brexit Vote':[], 'Con Vote':[], 'DUP Vote':[], 'Green Vote':[],
                        'Labour Vote':[], 'Liberal Vote':[], 'Other Vote':[], 'PC Vote':[], 'SDLP Vote':[],
                        'Sinn Fein Vote':[], 'SNP Vote':[], 'UUP Vote':[], "Election Date":[], "Date Ordinals":[]})

        cons_df_temp["Election Date"] = new_df_temp['Election Date']
        cons_df_temp['Date Ordinals'] = new_df_temp['Date Ordinals']

        # For all constituencies (list of constituencies = headers)


        for i in range(0, len(headers)): # for all constituencies
            cons_df = cons_df_temp
            for j in range(0, len(csv_names)): # for all parties
                string = "datasets/historical_csv/pre_processed/" + csv_names[j] + ".csv"
                party_df = pd.read_csv(string)
                cons_df[csv_names[j]] = party_df[headers[i]]
            string = 'datasets/historical_csv/pre_processed/constituencies/' + headers[i] + ".csv"
            cons_df.to_csv(string)
            print((650-i), "done")


    def seat_forecast(self):
        #test with ABERAVON
        cons_df = pd.read_csv('datasets/historical_csv/pre_processed/constituencies/ABERAVON.csv', index_col = 0)

        # splitting testing and training data
        x = cons_df.iloc[:, :13]
        y = cons_df.iloc[:, 14]

        x_train = x.iloc[:13, :]
        x_test = x.iloc[13:, :]

        y_train = y.iloc[:13]
        y_test = y.iloc[13:]

        model = Sequential()
        model.add(
            LSTM(10, activation='relu')
        )
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        num_epochs = 25
        model.fit_generator(x_train, y_train, epochs=num_epochs)


if __name__ == "__main__":
    m = Model()
    m.run_pop_forecast()


