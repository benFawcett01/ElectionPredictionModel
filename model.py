import time
from datetime import datetime, date
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.dtensor import optimizers
from keras.layers import Dense, Activation, SimpleRNN, Conv2D, Conv1D, LSTM
from keras.models import Sequential
from numpy import std, absolute
from numpy.random import seed
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import cross_val_score, RepeatedKFold, train_test_split
import tensorflow as tf
from tensorboard.summary.v1 import scalar
from tensorflow import losses
from tensorflow.python.framework.random_seed import set_random_seed


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
                lower = q1 - (0.5 * iqr)
                upper = q3 + (0.5 * iqr)

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
        model = RANSACRegressor(random_state = 120)
        pp_df = pd.read_csv("datasets/PreprocessedOpinionPolls.csv")
        pp_df = pp_df.sort_values(by='date_ordinals')

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
                x = pp_df['date_ordinals'].tail(50).values.reshape(-1, 1)
                y = pp_df[i].tail(50).values.reshape(-1, 1)

                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

                model.fit(x_train, y_train)
                yhat = model.predict(x_test)
                acc = r2_score(y_test, yhat)
                print("accuracy = %.3f" % acc)



                pred_date = "2024-06-08"
                pred_date = pd.to_datetime(pred_date)
                pred_date_ords = pred_date.toordinal()

                model.fit(x_train, y_train)

                pred = model.predict([[pred_date_ords]])
                model_results[i].append(float(pred))

        print(model_results)
        model_results = pd.DataFrame(model_results)
        clean_df = pd.read_csv("datasets/CleanOpinionPolls.csv")
        pp_df = pd.concat([pp_df, model_results.iloc[:, 1:]], ignore_index=True)
        clean_df = pd.concat([clean_df, model_results], ignore_index=True)
        pp_df.to_csv("datasets/PreprocessedOpinionPolls.csv", index=False)
        clean_df.to_csv("datasets/CleanOpinionPolls.csv", index=False)
        print(self.evaluate_model(x, y, model))
        return model

    def gen_popvote_graph(self, model):
        clean_df = pd.read_csv("datasets/CleanOpinionPolls.csv", index_col=False)

        pp_df = pd.read_csv("datasets/PreprocessedOpinionPolls.csv", index_col=False)

        clean_df = clean_df.iloc[:, 1:]
        clean_pp_df = pp_df.iloc[:]
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
                ax = sns.scatterplot(data=clean_pp_df, y=i, s=5, x="date_ordinals", color=c, label=i)

                labels = np.arange(min(clean_pp_df['date_ordinals'].astype(int)),
                                                max(clean_pp_df['date_ordinals'].astype(int)), 250)

                new_labels = []
                for j in range(0, len(labels)):
                    new_labels.append(date.fromordinal(labels[j]))

                ax.set_xticks(ticks = np.arange(min(clean_pp_df['date_ordinals'].astype(int)),
                                                max(clean_pp_df['date_ordinals'].astype(int)), 250), labels = new_labels)

                plt.xticks(rotation=30, size=8)
                print(i)
                plt.annotate(str(round(pp_df[i][len(pp_df) - 1], 2)),
                             xy=(clean_pp_df["date_ordinals"][len(pp_df) - 1] + 90, clean_pp_df[i][len(pp_df) - 1]), color = c)



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

        plt.clf()
        plt.scatter(x, y, label="Labour voting Intention", s=5)
        plt.plot(x, pred, color='black', label="Linear")
        plt.annotate(text = str(np.round(pred[len(pred)-1], 2)), xy = (x[len(x)-1], pred[len(pred)-1]))

        results = self.evaluate_model(x, y, linear)

        evals =[]
        evals.append(str('Linear MAE: ' + str(results)))

        # Huber Regression

        huber.fit(x, y)
        pred = huber.predict(x)
        scores = r2_score(y, pred)

        plt.plot(x, pred, color='red', label="Huber")
        plt.annotate(text = str(np.round(pred[len(pred)-1], 2)), xy = (x[len(x)-1], pred[len(pred)-1]))

        results = self.evaluate_model(x, y, huber)

        evals.append(str('Huber MAE: ' + str(results)))

        # RANSAC regression

        ransac.fit(x, y)
        pred = ransac.predict(x)

        scores = r2_score(y, pred)
        print("RANSAC", scores.mean())

        plt.plot(x, pred, color="orange", label="RANSAC")

        plt.annotate(text = str(np.round(pred[len(pred)-1], 2)), xy = (x[len(x)-1], pred[len(pred)-1]))

        results = self.evaluate_model(x, y, ransac)
        evals.append(str('RANSAC MAE: ' + str(results)))

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
        x_test, x_train, y_test, y_train = train_test_split(x, y, test_size=0.3)

        # evaluate model
        scores = cross_val_score(model, x_train, y_train, scoring='neg_mean_absolute_error', n_jobs=-1)

        # force scores to be positive
        return mean((absolute(scores)))

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

    def past_election_dataset_clean(self):
        df = pd.read_excel('datasets/general-elections-and-governments.xlsx', 3, index_col=0, skiprows=1)

        df = df.loc[df['Country'] == 'UK']


        elections = ['1918', '1922', '1923', '1924', '1929', '1931', '1935', '1945', '1950', '1951', '1955', '1959',
                     '1964', '1966', '1970', '1974F', '1974O', '1979', '1983', '1987', '1992', '1997', '2001', '2005',
                     '2010', '2015', '2017', '2019']

        df_template = pd.DataFrame({
            'Elections':[], 'CON Vote Share':[], 'CON Seats':[], 'LAB Vote Share':[], 'LAB Seats':[],'LD Vote Share':[],
            'LD Seats': [], 'PC/SNP Vote Share': [], 'PC/SNP Seats':[], 'Other Vote Share':[], 'Other Seats':[]
        })

        party_list = df['Party'].unique()

        party_df = df.loc[df['Party'] == party_list[1]]

        for i in range(0, len(party_list)):
            party_df = df.loc[df['Party'] == party_list[i]]
            party_df = party_df[['Vote share', 'Seats']]
            string = party_list[i] + " Vote Share"
            string2 = party_list[i] + " Seats"
            df_template[string] = party_df['Vote share'].round(2)
            df_template[string2] = party_df['Seats']

        df_template['Elections'] = ['1918-12-14','1922-11-15','1923-12-06','1924-10-29','1929-05-30','1931-10-27','1935-11-14','1945-07-05','1950-02-23','1951-10-25','1955-05-26','1959-10-08','1964-10-15', '1966-03-31', '1970-06-18', '1974-02-28', '1974-10-10', '1979-05-03', '1983-06-09',
                     '1987-06-11', '1992-04-09', '1997-05-01', '2001-06-07', '2005-05-05', '2010-05-06', '2015-05-07',
                     '2017-06-08', '2019-12-12']
        df_template['Elections'] = pd.to_datetime(df_template['Elections'], format='%Y-%m-%d')
        list = []

        for i in df_template['Elections']:
            list.append(i.toordinal())

        df_template['date_ordinal'] = list

        df_template.to_csv('datasets/past_elections_clean.csv', index=False)

    def plot_history(self):
        df = pd.read_csv('datasets/past_elections_clean.csv')

        list = [x for x in df.columns if 'Seats' in x]
        c_list = ['blue', 'red', 'orange', 'yellow', 'black']

        for i, c in zip(list, c_list):
            plt.plot(df[i], color=c)

        plt.xticks(ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                            25, 26,27], labels=df['Elections'], rotation=45)
        plt.axhline(y = 326, color = 'r', linestyle = '-')
        plt.savefig('static/media/tests/history.png')

    def forecast_seats(self):
        start = time.time()
        seed(1)
        set_random_seed(1)

        df = pd.read_csv('datasets/past_elections_clean.csv', index_col=0)

        pop_df = pd.read_csv('datasets/CleanOpinionPolls.csv')

        pop_forecast = pop_df.iloc[len(pop_df)-1, :].tolist()
        pop_forecast[0] = '2024'

        party_list = [x for x in df.columns if 'Share' in x]

        n = 1
        word = 50

        for i in range(n+1, len(pop_forecast) + 4*n, n + 1):
            pop_forecast.insert(i, word)

        pop_forecast = pop_forecast[:11]

        other = pop_forecast[7] + pop_forecast[9] # other = green + reform
        national = self.pred_nationalist() # predicts nationalist vote using linear regression

        pop_forecast[9] = other
        pop_forecast[7] = national[0][0]

        print(pop_forecast)
        for i in range(1, 10):
            pop_forecast[i] = round(pop_forecast[i] / 100, 3)
        print(pop_forecast)

        pop_forecast[0] = '2024-07-08'
        pop_forecast.append(pd.to_datetime('2024-07-08').toordinal())

        df.loc[len(df)] = pop_forecast
        df.fillna(0, inplace=True)

        vote_share = [x for x in df.columns if 'Vote Share' in x]
        seats = [x for x in df.columns if 'Seats' in x]

        # get past election columns (change in vote, seats)
        for j, k in zip(vote_share, seats):
            past_vote = [0]
            prev_seat = [0]
            for i in range(1, len(df[j])):
                past_vote_share = df[j].iloc[i] - df[j].iloc[i-1]
                previous_seat = df[k].iloc[i-1]
                past_vote.append(round(past_vote_share, 2))
                prev_seat.append(previous_seat)

            string = j + " Change"
            print(string)
            df[string] = past_vote
            string = k + " prev"
            df[string] = prev_seat


        margins = []
        margins_df = pd.DataFrame({
            'CON Margin': [],
            'LAB Margin': [],
            'LD Margin': [],
            'PC/SNP Margin': [],
            'Other Margin':[]
        })
        winners = []



        for i in range(0, len(df['CON Vote Share'])):
            highest_val = 0
            highest_column = ''
            for k in vote_share:
                if df[k].iloc[i] > highest_val:
                    highest_val = df[k].iloc[i]
                    highest_column = k
            winners.append(highest_column)
            for k in vote_share:
                margins.append(round(highest_val-df[k].iloc[i], 2))

        con = []
        lab = []
        ld = []
        nat = []
        oth = []

        for i in range(0, len(margins)):
            calc = i % 5
            if calc == 0:
                con.append(margins[i])
            elif calc == 1:
                lab.append(margins[i])
            elif calc == 2:
                ld.append(margins[i])
            elif calc == 3:
                nat.append(margins[i])
            else:
                oth.append(margins[i])

        margins_df['CON Margin'] = con
        margins_df['LAB Margin'] = lab
        margins_df['LD Margin'] = ld
        margins_df['PC/SNP Margin'] = nat
        margins_df['Other Margin'] = oth

        df = pd.concat([df, margins_df], axis=1)

        df.to_csv('datasets/past_elections_clean_with_2024.csv', index=False)

        x = df[['date_ordinal','CON Vote Share', 'CON Vote Share Change', 'CON Seats prev', 'CON Margin']]
        y = df[['CON Seats']]

        # Training the model
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
        model = self.nn()
        hist = model.fit(x_train, y_train, epochs=1000, validation_data=(x_train, y_train))

        print("x_test \n", x_test)
        '''
        train_predict = model.predict(x_train)
        test_predict = model.predict(x_test)

        pred_df = pd.DataFrame({
            'date_ordinals':[],
            'test_y':[],
            'pred_y':[]
        })

        pred_df['date_ordinals'] = x_test['date_ordinal']
        pred_df['test_y'] = y_test
        list = []
        for i in range(0, len(test_predict)):
            list.append(test_predict[i])

        pred_df['pred_y'] = list

        pred_df = pred_df.sort_values(by='date_ordinals')

        plt.plot(pred_df['date_ordinals'], pred_df['test_y'], color='b', label='actual')
        plt.plot(pred_df['date_ordinals'], pred_df['pred_y'], color='r', label='pred')
        plt.legend()
        plt.title("Train and Testing neural Network")
        plt.xlabel("Election Date (Ordinals)")
        plt.ylabel("No. Seats won")
        plt.savefig('static/media/tests/prediction.png')

        pred_df['pred_y'] = pred_df['pred_y'].astype('int') 

        #print(pred_df['date_ordinals'], pred_df['pred_y'])

        cvscores = []

        scores = model.evaluate(x_test, y_test, verbose=0)
        print(pred_df['pred_y'])'''

        # see with actual data

        color = ['b', 'r', 'orange', 'y', 'g']

        plt.clf()
        party_list = ['CON', 'LAB', 'LD', 'PC/SNP', 'Other']
        '''for i, c in zip(party_list, color):

            x = df[['date_ordinal', (i+' Vote Share'), (i+' Vote Share Change'), (i+' Seats prev'), (i +' Margin')]].iloc[2:len(df[(i+' Seats')])-1, :]
            print("x, y: ", len(x), len(y))
            y = df[[(i+' Seats')]].iloc[2:len(df[(i+' Seats')])-1]

            # Training the model
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
            hist = model.fit(x_train, y_train, epochs=1000, validation_data=(x_train, y_train))

            #pred_set = df[['date_ordinal', 'LAB Vote Share', 'LAB Vote Share Change', 'LAB Seats prev']].iloc[len(df['LAB Seats'])-1, :]
            #print("pred set:", pred_set)

            pred = model.predict([x])
            print(pred)

            print("y, date ordinals: ", len(y), len(x['date_ordinal']))

            plt.plot(x['date_ordinal'], y, color=c, label=(i +' actual'), alpha=0.5)
            plt.plot(x['date_ordinal'], pred, color=c, label=(i + ' pred'))
            plt.xlabel("Election Date (Ordinals)")
            plt.ylabel("No. Seats won")
            plt.title("Predicting number of seats won by each party from past elections")
            plt.hlines(y = 325, xmin = 700517, xmax = 737405)
            plt.legend(loc='lower left', fontsize=5)
        '''
        # Forecast next election's seat count:
        preds = pd.DataFrame({'CON Preds':[], 'LAB Preds':[], 'LD Preds':[], 'PC/SNP Preds':[], 'Other Preds':[]})

        for i in party_list:
            x = df[['date_ordinal', (i + ' Vote Share'), (i + ' Vote Share Change'), (i + ' Seats prev'),
                    (i + ' Margin')]].iloc[2:len(df['date_ordinal'])-1, :]
            print("x, y: ", len(x), len(y))
            y = df[[(i + ' Seats')]].iloc[2:len(df['date_ordinal'])-1]

            # Training the model
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
            hist = model.fit(x_train, y_train, epochs=1000, validation_data=(x_train, y_train))
            self.model_loss(hist, i)

            x_pred = df[['date_ordinal', (i + ' Vote Share'), (i + ' Vote Share Change'), (i + ' Seats prev'),
                         (i + ' Margin')]].iloc[:]
            y_pred = model.predict([x_pred]).tolist()
            string = i + ' Preds'
            preds[string] = y_pred


        seat_pred_unrounded = preds.iloc[len(preds)-1, :]
        print("s_p_u", seat_pred_unrounded)
        seat_pred = []
        for i in range(0, len(seat_pred_unrounded)):
            seat_pred.append(int(round(float(seat_pred_unrounded[i][0]), 0)))

        while sum(seat_pred) < 650:
            for i in range(0, 5):
                if sum(seat_pred) < 650:
                    seat_pred[i] = seat_pred[i]+1

        preds.loc[len(preds) - 1] = seat_pred

        print(preds.iloc[len(preds) - 1, :])
        preds.to_csv('datasets/seat_predictions.csv', index=False)

        plt.savefig('static/media/tests/act_prediction.png')
        plt.clf()
        self.represent_seat_forecast()

        winner = ''
        winner_seats = 0

        for i in range(0, len(seat_pred) - 1):
            if int(seat_pred[i]) > winner_seats:
                winner_seats = int(seat_pred[i])
                winner = party_list[i]

        end = time.time()
        print("Total Neural Network Runtime: ", (end-start))
        return winner

    def represent_seat_forecast(self):
        df = pd.read_csv('datasets/seat_predictions.csv')
        preds = df.iloc[len(df)-1]
        y = []
        for i in range(0, len(preds)):
            y.append(int(round(float(preds[i].replace('[', '').replace(']', '')), 0)))

        x = ['Con Seats', 'Lab Seats', 'LD Seats', 'PC/SNP Seats', 'Other Seats']
        print(x)
        print(y)

        colour = ['blue', 'red', 'orange', 'yellow', 'black']
        fig, ax = plt.subplots()
        ax.barh(x, y, color=colour)
        ax.set_yticklabels(labels=x, rotation=50)
        ax.set_xlabel("Seats Won")
        ax.set_ylabel("Party")
        plt.title(label='Seat Forecast by Party')



        for i, v in enumerate(y):
            ax.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
        plt.savefig("static/media/forecasts/seatforecast.png")



    def nn(self):
        seed(64)
        set_random_seed(64)

        model = Sequential([
            LSTM(25, input_shape=(5,1), activation="tanh", return_sequences=False),
            Dense(125, activation='relu'),
            Dense(125, activation='relu'),
            Dense(125, activation='relu'),
            Dense(125, activation='relu'),
            Dense(1, activation='softplus')
        ])

        adam = optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=adam, loss='log_cosh', metrics = ['mse', 'mae'])
        return model
    def pred_nationalist(self):
        df = pd.read_csv('datasets/past_elections_clean.csv', index_col=0)
        df.fillna(0, inplace=True)
        x = df['date_ordinal'].values.reshape(-1, 1)
        y = df['PC/SNP Vote Share'].values.reshape(-1, 1)

        model = LinearRegression()
        model.fit(x, y)
        p = model.predict([[pd.to_datetime('2024-07-08', format='%Y-%m-%d').toordinal()]])
        return p * 100

    def analyse_voteshare_seats(self):
        df = pd.read_csv('datasets/past_elections_clean_with_2024.csv', index_col=0)

        corr = df[['CON Seats', 'CON Seats Change']].corr()
        print(corr)

    # Taken from Towardsdatascience
    def model_loss(self, history, i):
        plt.clf()
        plt.figure(figsize=(8, 4))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Test Loss')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.legend(loc='upper right')
        if 'SNP' in i:
            i = 'PC-SNP'
        string = 'static/media/loss/' + i
        plt.savefig(string)
        plt.clf()


if __name__ == "__main__":
    m = Model()
    m.run_pop_forecast()
    m.forecast_seats()

