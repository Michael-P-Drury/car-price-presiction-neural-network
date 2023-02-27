import pandas as pd
import numpy as np
import keras
from sklearn.preprocessing import OneHotEncoder


def taking_manufacture_input():

    response = 0
    response_given = False

    while not response_given:
        print('What manufacturer would you like to train the neural network for?')
        print('1 : Audi')
        print('2 : BMW')
        print('3 : Ford')
        print('4 : Mercedes')
        print('5 : Skoda')
        print('6 : Toyota')
        print('7 : Vauxhall')
        print('8 : VW')

        try:
            response = int(input())
            if response < 1 or response > 7:
                print('invalid input')
            else:
                response_given = True

        except:
            print('invalid input')

    dataset = choosing_dataset(response)
    return dataset


def choosing_dataset(input_integer):

    if input_integer == 1:
        dataset = pd.read_csv('audi.csv')
    if input_integer == 2:
        dataset = pd.read_csv('bmw.csv')
    if input_integer == 3:
        dataset = pd.read_csv('ford.csv')
    if input_integer == 4:
        dataset = pd.read_csv('merc.csv')
    if input_integer == 5:
        dataset = pd.read_csv('skoda.csv')
    if input_integer == 6:
        dataset = pd.read_csv('toyota.csv')
    if input_integer == 7:
        dataset = pd.read_csv('vauxhall.csv')
    if input_integer == 8:
        dataset = pd.read_csv('vw.csv')

    return dataset


def make_model(x, y):

    model = keras.Sequential()
    model.add(keras.layers.Dense(15, activation='relu', input_shape=(len(x.axes[1]),)))
    model.add(keras.layers.Dense(15, activation='relu'))
    model.add(keras.layers.Dense(15, activation='relu'))
    model.add(keras.layers.Dense(15, activation='relu'))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer='Adam', loss='mean_squared_error')

    return model


def get_rid_of_string(x):

    models_df = x[['model']]
    models_df = models_df.to_numpy()
    models = []
    for i in range(0, len(models_df)):
        if models_df[i] in models:
            pass
        else:
            models.append(models_df[i][0])

    fuel_df = x[['fuelType']]
    fuel_df = fuel_df.to_numpy()
    fuels = []
    for i in range(0, len(fuel_df)):
        if fuel_df[i] in fuels:
            pass
        else:
            fuels.append(fuel_df[i])

    transmission_df = x[['transmission']]
    transmission_df = transmission_df.to_numpy()
    transmissions = []
    for i in range(0, len(transmission_df)):
        if transmission_df[i] in transmissions:
            pass
        else:
            transmissions.append(transmission_df[i])

    new_df = x[['transmission', 'fuelType', 'model']]
    X = OneHotEncoder().fit_transform(new_df).toarray()
    ohe = pd.DataFrame(X)
    x = pd.concat([x, ohe], axis=1, join="inner")
    x = x.drop(columns=['transmission', 'fuelType', 'model'])

    return x, models, fuels, transmissions


def taking_inputs(models, fuels, transmissions, x_length):

    out_array = np.zeros(x_length)

    model_loop = True
    while model_loop:
        try:
            print('Enter model of car: ')
            for i in range(0, len(models)):
                print(str(i+1) + ' : ' + models[i])
            model_in = int(input())
            if model_in >= 1 and model_in <= len(models):
                model_loop = False
            else:
                print('input not in range')
        except:
            ('invalid input')

    year_loop = True
    while year_loop:
        try:
            year_in = int(input('Enter year of car: '))
            year_loop = False
        except:
            ('invalid input')

    trans_loop = True
    while trans_loop:
        try:
            print('Enter transmission of car: ')
            for i in range(0, len(transmissions)):
                print(str(i + 1) + ' : ' + transmissions[i])
            trans_in = int(input())
            if trans_in >= 1 and trans_in <= len(transmissions):
                trans_loop = False
            else:
                print('input not in range')
        except:
            ('invalid input')


    mile_loop = True
    while mile_loop:
        try:
            milage_in = float(input('Enter milage of car: '))
            mile_loop = False
        except:
            ('invalid input')

    fuel_loop = True
    while fuel_loop:
        try:
            print('Enter fuel type of car: ')
            for i in range(0, len(fuels)):
                print(str(i + 1) + ' : ' + fuels[i])
            fuel_in = int(input())
            if fuel_in >= 1 and fuel_in <= len(fuels):
                fuel_loop = False
            else:
                print('input not in range')
        except:
            ('invalid input')

    mpg_loop = True
    while mpg_loop:
        try:
            mpg_in = float(input('Enter mpg of your car: '))
            mpg_loop = False
        except:
            ('invalid input')


    engine_loop = True
    while engine_loop:
        try:
            engine_in = float(input('Enter engine size of your car (litres): '))
            engine_loop = False
        except:
            ('invalid input')

    out_array[0] = year_in
    out_array[1] = milage_in
    out_array[2] = mpg_in
    out_array[3] = engine_in
    out_array[(3 + trans_in)] = 1
    out_array[(3 + len(transmissions) + fuel_in)] = 1
    out_array[(3 + len(transmissions) + len(fuels) + model_in)] = 1

    return out_array


def main():

    main_loop = True

    while main_loop:

        response_first = False

        while not response_first:
            print('What would you like to do?')
            print('1 : leave')
            print('2 : train neural network for a brand of car')

            try:
                option1 = int(input())
                if option1 < 1 or option1 > 2:
                    print('invalid input')
                else:
                    response_first = True
            except:
                print('invalid input')

        if option1 == 1:
            quit()
        else:
            dataset = taking_manufacture_input()

        x = dataset.drop(columns = ['price', 'tax'])
        y = dataset[['price']]

        rid_output = get_rid_of_string(x)
        x = rid_output[0]
        models = rid_output[1]
        fuels = rid_output[2]
        transmissions = rid_output[3]

        x_array = x.to_numpy()
        test_data = np.array(x_array[0, :])

        model = make_model(x, y)
        model.fit(x, y, epochs = 200)

        loop_2 = True
        while loop_2:

            response_second = False
            while not response_second:
                print('What would you like to do?')
                print('1 : leave')
                print('2 : train new dataset')
                print('3 : input data for prediction')
                print('4 : use preset prediction')

                try:
                    option2 = int(input())
                    if option2 < 1 or option2 > 4:
                        print('invalid input')
                    else:
                        response_second = True
                except:
                    print('invalid input')

            if option2 == 1:
                quit()
            elif option2 == 2:
                loop_2 = False
            elif option2 == 3:
                predict_array = taking_inputs(models, fuels, transmissions, len(x.axes[1]))
                print(model.predict(predict_array.reshape(1, len(x.axes[1])), batch_size=1))
            else:
                print(model.predict(test_data.reshape(1, len(x.axes[1])), batch_size = 1))


if __name__ == '__main__':
    main()
