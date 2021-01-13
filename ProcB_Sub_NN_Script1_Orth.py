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

def neural_net_run(training_test_sizes, Figs, scramble):

    Total_number_points = training_test_sizes[-1]
    #total training dataset size

    data_dist= np.random.normal(loc=0, size=Total_number_points)
    #Normal distribution of distances from home, mean=0
    distances = [(np.abs(i)) + (0.001) for i in data_dist]
    #take absolute values from normal distribution and offset from 0
    familiar_area = np.percentile(distances, 95)
    d_range = max(distances)


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
    #Cartesian coordinates


    #Randomly determines the order of the coordinates such that they don't match the distance/direction information if scramble=True
    #Used to test the neural network performance relative to 'chance'
    if scramble==True:
        rand_order = np.random.permutation(len(X))
        X1 = np.array(X)[rand_order]
        Y1 = np.array(Y)[rand_order]
    else:
        X1 = X
        Y1 = Y
    #X1 and Y1 are to be passed to the NN

    #'sensory' is the information about the modelled environmental gradients passed as input to the neural networks
    sensory = np.array([[X1], [Y1]])
    sensory = np.asarray(sensory, dtype=float)
    sensory = np.reshape(sensory, (2, len(X)))
    sensory = np.ndarray.transpose(sensory)

    angles1 = angles/(2*np.pi)
    #change angle to vary between 0 and 1
    path_int = np.array([[angles1, distances]])
    path_int = np.asarray(path_int, dtype=float)
    path_int = np.reshape(path_int, (2, len(angles)))
    path_int = np.ndarray.transpose(path_int)
    #path_int is the distance/direction information passed as outputs to the neural network


    #NEURAL NETWORK
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
    def testing(test_size, dist_low, dist_high, model_to_test):
        test_dist = np.random.uniform(low = dist_low, high = dist_high, size = test_size)
        test_ang = np.random.uniform(low = 0, high = 2*np.pi, size = test_size)
        cart = cartesian(test_dist, test_ang)
        test_X = [x.real + home_coord.real for x in cart]
        test_Y = [x.imag + home_coord.imag for x in cart]
        test_sensory = np.array([[test_X], [test_Y]])
        test_sensory = np.asarray(test_sensory, dtype=float)
        test_sensory = np.reshape(test_sensory, (2, len(test_X)))
        test_sensory = np.ndarray.transpose(test_sensory)
        test_ang1 = test_ang/(2*np.pi)
        test_correct = np.array([[test_ang1, test_dist]])
        test_correct = np.asarray(test_correct, dtype=float)
        test_correct = np.reshape(test_correct, (2, len(test_dist)))
        test_correct = np.ndarray.transpose(test_correct)
        pred = model_to_test.predict(test_sensory)
        out = test(pred, test_correct)
        return([out, test_X, test_Y])


    #Number of test datapoints in each test region
    t_size = 1000

    #Testing prior to training:

    #high familiarity area testing
    familiar_out0 = testing(t_size, 0.001, familiar_area, model)
    familiar_out = [familiar_out0[0]]
    fam_test_X = familiar_out0[1]
    fam_test_Y = familiar_out0[2]

    #low familiarity area testing
    unfamiliar_out0 = testing(t_size, familiar_area, d_range, model)
    unfamiliar_out = [unfamiliar_out0[0]]
    unfam_test_X = unfamiliar_out0[1]
    unfam_test_Y = unfamiliar_out0[2]

    #novel area testing
    novel_out0 = testing(t_size, d_range, d_range*1.5, model)
    novel_out = [novel_out0[0]]
    novel_test_X = novel_out0[1]
    novel_test_Y = novel_out0[2]

    for i in range(1, len(training_test_sizes)):
        print("Training", i)
        model.fit(
        x = sensory[training_test_sizes[i-1]:training_test_sizes[i]],
        y = [path_int[training_test_sizes[i-1]:training_test_sizes[i],0], path_int[training_test_sizes[i-1]:training_test_sizes[i],1]],
        validation_split=0,
        batch_size = 1,
        epochs = 1,
        )
        new_fam_out = testing(t_size, 0.001, familiar_area, model)[0]
        familiar_out.append(new_fam_out)
        new_unfam_out = testing(t_size, familiar_area, d_range, model)[0]
        unfamiliar_out.append(new_unfam_out)
        new_novel_out = testing(t_size, d_range, d_range*1.5, model)[0]
        novel_out.append(new_novel_out)

    familiar_out = np.reshape(familiar_out, (len(familiar_out),3))
    unfamiliar_out = np.reshape(unfamiliar_out, (len(unfamiliar_out),3))
    novel_out = np.reshape(novel_out, (len(novel_out),3))

    run_return = [familiar_out, unfamiliar_out, novel_out]
    if Figs==True:
        plt.figure(1, figsize = (7,7))
        plt.xlim(((-1.5*d_range + home_coord.real),(1.5*d_range + home_coord.real)))
        plt.ylim(((-1.5*d_range + home_coord.imag),(1.5*d_range + home_coord.imag)))
        for pl in range(0, 9):
            plt.plot((10, -10), (pl, pl), color='dimgrey', linestyle='dashed', linewidth = 1.8, alpha = 0.7)
            plt.plot((10, -10), (-pl, -pl), color='dimgrey', linestyle='dashed', linewidth = 1.8, alpha = 0.7)
        for pl1 in range(0,8):
            plt.plot([i+pl1 for i in (0, 0)], (-6500 , 6500), color='dimgrey', linestyle='dotted', linewidth = 1.8, alpha = 0.7)
            plt.plot([i-pl1 for i in (0, 0)], (-6500 , 6500), color='dimgrey', linestyle='dotted', linewidth = 1.8, alpha = 0.7)
        plt.scatter([i for i in X], [i for i in Y], color='blue', s = 2.5, alpha=0.2)
        plt.scatter(home_coord.real, home_coord.imag, color='black', s = 50, alpha=1)
        circle1 = plt.Circle((home_coord.real, home_coord.imag), familiar_area, color='r', fill=False, lw=2.5)
        circle2 = plt.Circle((home_coord.real, home_coord.imag), d_range, color='blueviolet', fill=False, lw=2.5)
        circle3 = plt.Circle((home_coord.real, home_coord.imag), d_range*1.5, color='magenta', fill=False, lw=2.5)
        plt.gcf().gca().add_artist(circle1)
        plt.gcf().gca().add_artist(circle2)
        plt.gcf().gca().add_artist(circle3)
        plt.ylabel('Northings')
        plt.xlabel('Eastings')
        #plt.savefig('/Users/joemorford/Desktop/Nav_NNs/ProcB_Submission/ProcB_Sub_Figs/Raw_Orth1.jpg', dpi=600)
        plt.show()

        #Test dataset
        plt.figure(1, figsize = (7,7))
        plt.xlim(((-1.5*d_range + home_coord.real),(1.5*d_range + home_coord.real)))
        plt.ylim(((-1.5*d_range + home_coord.imag),(1.5*d_range + home_coord.imag)))
        plt.scatter([i for i in fam_test_X], [i for i in fam_test_Y], color='red', s = 2, alpha=0.4)
        plt.scatter([i for i in unfam_test_X], [i for i in unfam_test_Y], color='blueviolet', s = 2, alpha=0.4)
        plt.scatter([i for i in novel_test_X], [i for i in novel_test_Y], color='magenta', s = 2, alpha=0.4)
        plt.scatter(home_coord.real, home_coord.imag, color='black', s = 50, alpha=1)
        circle1 = plt.Circle((home_coord.real, home_coord.imag), familiar_area, color='red', fill=False, lw=2.5)
        circle2 = plt.Circle((home_coord.real, home_coord.imag), d_range, color='blueviolet', fill=False, lw=2.5)
        circle3 = plt.Circle((home_coord.real, home_coord.imag), d_range*1.5, color='magenta', fill=False, lw=2.5)
        plt.gcf().gca().add_artist(circle1)
        plt.gcf().gca().add_artist(circle2)
        plt.gcf().gca().add_artist(circle3)
        plt.ylabel('Northings')
        plt.xlabel('Eastings')
        #plt.savefig('/Users/joemorford/Desktop/Nav_NNs/ProcB_Submission/ProcB_Sub_Figs/Raw_Orth2.jpg', dpi=600)
        plt.show()

    return(run_return)


training_test_szs = [0]
for each in training_test_sizes1:
    training_test_szs.append(each)



runs = []
null_runs = []

for k in range(N_runs):
    seed(2*k)
    print("\nRun", k)
    run = neural_net_run(training_test_szs,Figs=False, scramble=False)
    runs.append(run)
    print("\nNull run", k)
    seed(2*k + 1)
    if k == (N_runs-1):
        null_run = neural_net_run(training_test_szs,True, True)
    else:
        null_run = neural_net_run(training_test_szs, False, True)
    null_runs.append(null_run)



colors = ["red", "blueviolet", "magenta"]
areas = ["high familiarity", "low familiarity", "novel"]


plt.figure(1, figsize = (7,7))
plt.ylim(0, 120)
plt.ylabel('Mean absolute angular error (degrees)')
plt.xlabel('Number of training datapoints')
for k in range(3):
    angs = np.zeros((len(runs),len(training_test_szs)))
    null_angs = np.zeros((len(null_runs),len(training_test_szs)))
    for i in range(len(runs)):
        angs[i, :] = np.absolute(runs[i][k][:,2])
        null_angs[i, :] = np.absolute(null_runs[i][k][:,2])
    print("\nMWU angle test in", areas[k], "area:")
    final_training_angs = angs[:,(-1)]
    final_training_null_angs = null_angs[:,(-1)]
    final_training_ang_test = mannwhitneyu(final_training_angs, final_training_null_angs)
    print(final_training_ang_test)
    plt.plot(training_test_szs, angs.mean(axis=0), color=colors[k], linewidth = 2.5, alpha = 0.8)
    plt.plot(training_test_szs, null_angs.mean(axis=0), color=colors[k], linewidth = 2.5, alpha = 0.5, linestyle="dashed")
    plt.fill_between(training_test_szs, angs.mean(axis=0) + angs.std(axis=0)/np.sqrt(len(runs)), angs.mean(axis=0) - angs.std(axis=0)/np.sqrt(len(runs)), color=colors[k], alpha=0.2)
    plt.fill_between(training_test_szs, null_angs.mean(axis=0) + null_angs.std(axis=0)/np.sqrt(len(runs)), null_angs.mean(axis=0) - null_angs.std(axis=0)/np.sqrt(len(runs)), color=colors[k], alpha=0.2)
#plt.savefig('/Users/joemorford/Desktop/Nav_NNs/ProcB_Submission/ProcB_Sub_Figs/Raw_Orth3.jpg', dpi=600)
plt.show()


plt.figure(1, figsize = (7,7))
plt.ylim(0, 5)
plt.ylabel('Mean absolute distance error (arbitrary units)')
plt.xlabel('Number of training datapoints')
for k in range(3):
    dists = np.zeros((len(runs),len(training_test_szs)))
    null_dists = np.zeros((len(null_runs),len(training_test_szs)))
    for i in range(len(runs)):
        dists[i, :] = np.absolute(runs[i][k][:,1])
        null_dists[i, :] = np.absolute(null_runs[i][k][:,1])
    print("\nMWU dists test in", areas[k], "area:")
    final_training_dists = dists[:,(-1)]
    final_training_null_dists = null_dists[:,(-1)]
    final_training_dist_test = mannwhitneyu(final_training_dists, final_training_null_dists)
    print(final_training_dist_test)
    plt.plot(training_test_szs, dists.mean(axis=0), color=colors[k], linewidth = 2.5, alpha = 0.8)
    plt.plot(training_test_szs, null_dists.mean(axis=0), color=colors[k], linewidth = 2.5, alpha = 0.5, linestyle="dashed")
    plt.fill_between(training_test_szs, dists.mean(axis=0) + dists.std(axis=0)/np.sqrt(len(runs)), dists.mean(axis=0) - dists.std(axis=0)/np.sqrt(len(runs)), color=colors[k], alpha=0.2)
    plt.fill_between(training_test_szs, null_dists.mean(axis=0) + null_dists.std(axis=0)/np.sqrt(len(runs)), null_dists.mean(axis=0) - null_dists.std(axis=0)/np.sqrt(len(runs)), color=colors[k], alpha=0.2)
#plt.savefig('/Users/joemorford/Desktop/Nav_NNs/ProcB_Submission/ProcB_Sub_Figs/Raw_Orth4.jpg', dpi=600)
plt.show()



angle_overall_results = np.zeros((len(runs),3))
for k in range(3):
    for i in range(len(runs)):
        angle_overall_results[i, k] = runs[i][k][(-1),0]*2*np.pi/360

for k in range(3):
    plt.figure(1, figsize = (4,4))
    theta = angle_overall_results[:,k]
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.grid(True)
    y, x, _ = ax.hist(theta, bottom=0, color=colors[k], alpha=1, bins='auto', density=True, rwidth=1, zorder=2)
    ax.bar(theta, (y.max()/2), width=0.1, bottom=(-y.max()/2), color="lightsteelblue", alpha=0.5, zorder=1)
    ax.set_ylim([(-y.max()/2),(6*y.max()/4)])
    ax.set_yticks([(-y.max()/4), 0, (y.max()/4), (2*y.max()/4), (3*y.max()/4), (4*y.max()/4), (5*y.max()/4), (6*y.max()/4)], minor=False)
    ax.set_axisbelow(True)
    ax.set_yticklabels([])
    out_file = '/Users/joemorford/Desktop/Nav_NNs/ProcB_Submission/ProcB_Sub_Figs/Raw_Orth' + str(5 + k) + '.jpg'
    #plt.savefig(out_file, dpi=600)
    plt.show()
