import sys
from scipy.stats import circmean
from scipy.stats import circstd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow
import keras
import keras.layers
import keras.models
from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras import optimizers
import keras.backend as K
from numpy.random import seed
from scipy.stats import mannwhitneyu


from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times New Roman']})
rc('text', usetex=True)
plt.rcParams['font.size'] = 16.5

N_runs = 50
#Number of neural networks


training_test_sizes1 = [10, 100, 300, 500, 1000, 1500, 2000, 3000, 4000, 5000]
#the training dataset sizes at which the neural networks are tested
#the networks are also tested with no training


Z_axis_to_x_values = [np.pi/2, np.pi/6, np.pi/36]
#vector of angles between axes for non-orthogonal comparison
#the acute angle between the axes along which the gradeints vary
#np.pi/2 for orthogonal axes; smaller for non-orthogonal axes

def neural_net_run(training_test_sizes, Z_axis_to_x, scramble):

    Total_number_points = training_test_sizes[len(training_test_sizes)-1]

    data_dist= np.random.normal(loc=0, size=Total_number_points)
    #Normal distribution of distances from home, mean=0
    distances = [np.abs(i)+0.001 for i in data_dist]
    #take absolute values from normal distribution and offset from 0

    angles = np.random.uniform(low = 0, high = 2*np.pi, size = Total_number_points)
    #uniform distribution of bearings from home in radians

    def cartesian(dist, ang):
        ang1 = ang - (np.pi/2)
        ang2 = 2*np.pi - ang1
        ang3 = np.where(ang2 < 0, 2*np.pi + ang2, ang2)
        x_y = dist * np.exp(1j * ang3)
        return x_y
    #function takes angle and distance from home and returns cartesian coordinates of training points


    home_dist = 0
    home_ang = 0
    home_coord = cartesian(home_dist, home_ang)


    coords = cartesian(distances, angles)
    X = [x.real + home_coord.real for x in coords]
    Y = [y.imag + home_coord.imag for y in coords]


    #produces Z coordinate; if axes are orthogonal, then Z and Y coordinates are identical.
    #Neural networks are trained with X and Z coordinates which are modelled environmental gradients
    Z = []
    Z_m = np.tan(Z_axis_to_x)
    Z_m_90 = -1/Z_m
    for i in range(Total_number_points):
        x_intersect = (Y[i] -Z_m_90*X[i])/(Z_m - Z_m_90)
        y_intersect = Z_m * x_intersect
        z_value = np.sqrt(x_intersect**2 + y_intersect**2)
        if x_intersect < 0:
            z_value = -z_value
        Z.append(z_value)

    if scramble==True:
        rand_order = np.random.permutation(len(X))
        X1 = np.array(X)[rand_order]
        Y1 = np.array(Y)[rand_order]
        Z1 = np.array(Z)[rand_order]
    else:
        X1 = X
        Y1 = Y
        Z1 = Z

    #'sensory' is the information about the modelled environmental gradients passed as input to the neural networks
    sensory = np.array([[X1], [Z1]])
    sensory = np.asarray(sensory, dtype=float)
    sensory = np.reshape(sensory, (2, len(X)))
    sensory = np.ndarray.transpose(sensory)

    #path_int is the distance/direction information passed as outputs to the neural network
    angles = angles/(2*np.pi)
    #change angle to vary between 0 and 1
    path_int = np.array([[angles, distances]])
    path_int = np.asarray(path_int, dtype=float)
    path_int = np.reshape(path_int, (2, len(angles)))
    path_int = np.ndarray.transpose(path_int)


    visible = Input(shape=(sensory.shape[1],))
    hidden1 = Dense(10, activation='relu')(visible)
    hidden2 = Dense(100, activation='relu')(hidden1)
    hidden3 = Dense(200, activation='relu')(hidden2)
    hidden4 = Dense(500, activation='relu')(hidden3)
    hidden5 = Dense(200, activation='relu')(hidden4)
    hidden6 = Dense(100, activation='relu')(hidden5)
    out1 = Dense(1, activation='sigmoid')(hidden6)
    out2 = Dense(1, activation='relu')(hidden6)

    model = Model(inputs=visible, outputs=[out1, out2])

    sgd = optimizers.SGD(lr=0.001, decay=0, momentum=0.9, nesterov=True)

    def customLoss(true, predict):
        diff = K.abs(true-predict)
        new_diff = K.switch(diff>0.5, 1-diff, diff)
        return(K.mean(K.square(new_diff), axis=-1))

    model.compile(
    optimizer = sgd,
    loss = [customLoss, "mse"]
    )


    #finds difference between predictions of model and correct answer
    #puts angle on 0-360 scale
    def test(model_pred, correct):
        angdiff = []
        distdiff = []
        absangdiff = []
        for (each1, each2) in zip(model_pred[0], correct):
            diff = each1 - each2[0]
            if (diff>0.5):
                diff_out = diff-1
            elif (diff<(-0.5)):
                diff_out = 1+diff
            else:
                diff_out = diff
            angdiff.append(diff_out*360)
            absangdiff.append(np.absolute(diff_out)*360)
        for (each1, each2) in zip(model_pred[1], correct):
            d_diff = np.absolute(each1 - each2[1])
            distdiff.append(d_diff)
        meanangdiff = circmean(angdiff, low=-180, high=180)
        meanabsang = np.mean(absangdiff)
        meandistdiff = np.mean(distdiff)
        to_return = np.array([[meanangdiff], [meandistdiff], [meanabsang]])
        to_return = np.asarray(to_return, dtype=float)
        to_return = np.reshape(to_return, (3, 1))
        to_return = np.ndarray.transpose(to_return)
        return(to_return)


    #prepares and processes network output to be tested
    def testing(test_size, model_to_test):
        test_dist_dist = np.random.normal(loc=0, size=test_size)
        test_dist = [np.abs(i)+0.001 for i in test_dist_dist]
        test_ang = np.random.uniform(low = 0, high = 2*np.pi, size = test_size)
        cart = cartesian(test_dist, test_ang)
        test_X = [x.real + home_coord.real for x in cart]
        test_Y = [x.imag + home_coord.imag for x in cart]
        test_Z = []
        for i in range(test_size):
            x_intersect = (test_Y[i] -Z_m_90*test_X[i])/(Z_m - Z_m_90)
            y_intersect = Z_m * x_intersect
            z_value = np.sqrt(x_intersect**2 + y_intersect**2)
            if x_intersect < 0:
                z_value = -z_value
            test_Z.append(z_value)
        test_sensory = np.array([[test_X], [test_Z]])
        test_sensory = np.asarray(test_sensory, dtype=float)
        test_sensory = np.reshape(test_sensory, (2, len(test_X)))
        test_sensory = np.ndarray.transpose(test_sensory)
        test_ang = test_ang/(2*np.pi)
        test_correct = np.array([[test_ang, test_dist]])
        test_correct = np.asarray(test_correct, dtype=float)
        test_correct = np.reshape(test_correct, (2, len(test_dist)))
        test_correct = np.ndarray.transpose(test_correct)
        pred = model_to_test.predict(test_sensory)
        out = test(pred, test_correct)
        return([out, test_X, test_Y])


    #Number of test datapoints in each region
    t_size = 1000

    test_out0 = testing(t_size, model)
    test_out = [test_out0[0]]
    test_X = test_out0[1]
    test_Y = test_out0[2]

    for i in range(1, len(training_test_sizes)):
        print("Training", i)
        model.fit(
        x = sensory[training_test_sizes[i-1]:training_test_sizes[i]],
        y = [path_int[training_test_sizes[i-1]:training_test_sizes[i],0], path_int[training_test_sizes[i-1]:training_test_sizes[i],1]],
        validation_split=0,
        batch_size = 1,
        epochs = 1,
        )
        new_test_out = testing(t_size, model)[0]
        test_out.append(new_test_out)

    test_out = np.reshape(test_out, (len(test_out), 3))

    run_return = test_out
    return(run_return)


training_test_szs = [0]
for each in training_test_sizes1:
    training_test_szs.append(each)


all_runs = []
all_null_runs = []

count = 0
for l in range(len(Z_axis_to_x_values)):
    Z_axis_to_x = Z_axis_to_x_values[l]
    runs = []
    null_runs = []
    for k in range(N_runs):
        print("\nAngle between axes:", Z_axis_to_x*360/(2*np.pi))
        print("Run", k)
        seed(count)
        run = neural_net_run(training_test_szs, Z_axis_to_x, False)
        runs.append(run)
        print("\nAngle between axes:", Z_axis_to_x*360/(2*np.pi))
        print("Null run", k)
        count += 1
        seed(count)
        null_run = neural_net_run(training_test_szs, Z_axis_to_x, True)
        null_runs.append(null_run)
        count += 1
    all_runs.append(runs)
    all_null_runs.append(null_runs)



colors = ["black", "blue", "green"]
axis_angles = ["90", "30", "5"]



plt.figure(1, figsize = (5,5))
plt.xlim(((-6),(6)))
plt.ylim(((-6),(6)))
plt.plot((-6500 , 6500), (0, 0), color='black', linewidth = 1.8)
for (each_ang, each_col) in zip(Z_axis_to_x_values, colors):
    plt.plot((-6500*np.cos(each_ang), 6500*np.cos(each_ang)), (-6500*np.sin(each_ang), 6500*np.sin(each_ang)), color=each_col, linewidth = 3, linestyle="dotted")
plt.axis('off')
#plt.savefig('/Users/joemorford/Desktop/Nav_NNs/ProcB_Submission/ProcB_Sub_Figs/Raw_Non_Orth1.jpg', dpi=600)
plt.show()


plt.figure(1, figsize = (7,7))
plt.ylim(0, 120)
plt.ylabel('Mean absolute angular error (degrees)')
plt.xlabel('Number of training datapoints')
final_training_angs_all = []
final_training_angs_all1000 = []
for j in range(len(Z_axis_to_x_values)):
    runs = all_runs[j]
    null_runs = all_null_runs[j]
    angs = np.zeros((len(runs),len(training_test_szs)))
    null_angs = np.zeros((len(null_runs),len(training_test_szs)))
    for i in range(len(runs)):
        angs[i, :] = np.absolute(runs[i][:,2])
        null_angs[i, :] = np.absolute(null_runs[i][:,2])
    print("MWU angle test for axes at", axis_angles[j], "degrees:")
    final_training_angs = angs[:,(-1)]
    final_training_null_angs = null_angs[:,(-1)]
    final_training_ang_test = mannwhitneyu(final_training_angs, final_training_null_angs)
    print(final_training_ang_test)
    final_training_angs_all.append(final_training_angs)
    final_training_angs_all1000.append(angs[:,5])
    plt.plot(training_test_szs, angs.mean(axis=0), color=colors[j], linewidth = 2.5, alpha = 0.8)
    plt.plot(training_test_szs, null_angs.mean(axis=0), color=colors[j], linewidth = 2.5, alpha = 0.5, linestyle="dashed")
    plt.fill_between(training_test_szs, angs.mean(axis=0) + angs.std(axis=0)/np.sqrt(len(runs)), angs.mean(axis=0) - angs.std(axis=0)/np.sqrt(len(runs)), color=colors[j], alpha=0.2)
    plt.fill_between(training_test_szs, null_angs.mean(axis=0) + null_angs.std(axis=0)/np.sqrt(len(runs)), null_angs.mean(axis=0) - null_angs.std(axis=0)/np.sqrt(len(runs)), color=colors[j], alpha=0.2)
#plt.savefig('/Users/joemorford/Desktop/Nav_NNs/ProcB_Submission/ProcB_Sub_Figs/Raw_Non_Orth2.jpg', dpi=600)
plt.show()

print("\n")
for i in range(len(final_training_angs_all)-1):
    for j in range(i, len(final_training_angs_all)-1):
        test1 = mannwhitneyu(final_training_angs_all[i], final_training_angs_all[j+1])
        print("Angle Test at end of training:", axis_angles[i], "degrees vs", axis_angles[j+1], "degrees")
        print(test1)
        test2 = mannwhitneyu(final_training_angs_all1000[i], final_training_angs_all1000[j+1])
        print("Angle Test after 1000 datapoints", axis_angles[i], "degrees vs", axis_angles[j+1], "degrees")
        print(test2)


print("\n")

plt.figure(1, figsize = (7,7))
plt.ylim(0, 1)
plt.ylabel('Mean absolute distance error (arbitrary units)')
plt.xlabel('Number of training datapoints')
final_training_dists_all = []
final_training_dists_all1000 = []
for j in range(len(Z_axis_to_x_values)):
    runs = all_runs[j]
    null_runs = all_null_runs[j]
    dists = np.zeros((len(runs),len(training_test_szs)))
    null_dists = np.zeros((len(null_runs),len(training_test_szs)))
    for i in range(len(runs)):
        dists[i, :] = np.absolute(runs[i][:,1])
        null_dists[i, :] = np.absolute(null_runs[i][:,1])
    print("MWU dists test for axes at", axis_angles[j], "degrees:")
    final_training_dists = dists[:,(-1)]
    final_training_null_dists = null_dists[:,(-1)]
    final_training_dist_test = mannwhitneyu(final_training_dists, final_training_null_dists)
    print(final_training_dist_test)
    final_training_dists_all.append(final_training_dists)
    final_training_dists_all1000.append(dists[:,5])
    plt.plot(training_test_szs, dists.mean(axis=0), color=colors[j], linewidth = 2.5, alpha = 0.8)
    plt.plot(training_test_szs, null_dists.mean(axis=0), color=colors[j], linewidth = 2.5, alpha = 0.5, linestyle="dashed")
    plt.fill_between(training_test_szs, dists.mean(axis=0) + dists.std(axis=0)/np.sqrt(len(runs)), dists.mean(axis=0) - dists.std(axis=0)/np.sqrt(len(runs)), color=colors[j], alpha=0.2)
    plt.fill_between(training_test_szs, null_dists.mean(axis=0) + null_dists.std(axis=0)/np.sqrt(len(runs)), null_dists.mean(axis=0) - null_dists.std(axis=0)/np.sqrt(len(runs)), color=colors[j], alpha=0.2)
#plt.savefig('/Users/joemorford/Desktop/Nav_NNs/ProcB_Submission/ProcB_Sub_Figs/Raw_Non_Orth3.jpg', dpi=600)
plt.show()


print("\n")
for i in range(len(final_training_dists_all)-1):
    for j in range(i, len(final_training_dists_all)-1):
        test1 = mannwhitneyu(final_training_dists_all[i], final_training_dists_all[j+1])
        print("Dist Test at end of training:", axis_angles[i], "degrees vs", axis_angles[j+1], "degrees")
        print(test1)
        test2 = mannwhitneyu(final_training_dists_all1000[i], final_training_dists_all1000[j+1])
        print("Dist Test after 1000 datapoints", axis_angles[i], "degrees vs", axis_angles[j+1], "degrees")
        print(test2)
