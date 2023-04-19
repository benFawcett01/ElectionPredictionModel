import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras import optimizers
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class model2:

    def forecast_seats(self):
        df = pd.read_csv('datasets/past_elections_clean.csv', index_col=0)

        pop_df = pd.read_csv('datasets/CleanOpinionPolls.csv')

        pop_forecast = pop_df.iloc[len(pop_df)-1, :].tolist()
        pop_forecast[0] = '2024'

        party_list = [x for x in df.columns if 'Share' in x]

        n = 1
        word = 0

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

        df.to_csv('datasets/past_elections_clean_with_2024.csv', index=False)

        #test nn

        x = df[['date_ordinal','CON Vote Share', 'CON Seats']].iloc[12:len(df)-1, :]
        y = df[['CON Seats']].iloc[12:len(df)-1]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8)
        model = self.nn(x_train, y_train)


        hist = model.fit(x, y, epochs=400, validation_data=(x_train, y_train))

        train_predict = model.predict(x_train)
        test_predict = model.predict(x_test)

        pred_df = pd.DataFrame({
            'date_ordinals':[],
            'test_y':[],
            'pred_y':[]
        })

        print(test_predict)
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
        plt.savefig('static/media/tests/prediction.png')

        for i, j in zip(x_test['date_ordinal'], test_predict):
            print(i, j)

        print(df.iloc[len(df)-1, :][['date_ordinal','CON Vote Share']].tolist())
    def nn(self, x, y):
        model = Sequential([
            LSTM(3, input_shape=(3,1), activation="relu"),
            Dense(10, activation='relu'),
            Dense(10, activation='relu'),
            Dense(10, activation='relu'),
            Dense(1)
        ])

        adam = adam = optimizers.Adam(learning_rate=0.001)
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

if __name__ == "__main__":
    m = model2()
    m.